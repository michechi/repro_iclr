#!/usr/bin/env python3
"""Run the ordered KIP M4 data-efficiency queue, including detached supervision."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARM_ORDER = ("qwen27b_qlora", "qwen14b_qlora", "llama8b_lora", "llama8b_full")
FRACTIONS = (0.01, 0.10, 0.30, 0.50, 0.75, 1.0)
MODEL_PATHS = {
    "qwen27b_qlora": ROOT / Path("data/model_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "qwen14b_qlora": ROOT / "data/model_cache/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9",
    "llama8b_lora": ROOT / "data/model_cache/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "llama8b_full": ROOT / "data/model_cache/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
}
MICROBATCHES = {"qwen27b_qlora": 128, "qwen14b_qlora": 128, "llama8b_lora": 128, "llama8b_full": 16}
SOURCE_FILES = (
    "scripts/run_kip_data_efficiency.py",
    "scripts/audit_kip_efficiency_weights.py",
    "scripts/plot_kip_data_efficiency.py",
    "src/experiments/kip_data_efficiency.py",
    "src/experiments/kip_efficiency_models.py",
    "src/experiments/kip_llama_training.py",
    "src/experiments/kip_encoder_training.py",
    "src/experiments/kip_cpu_offload.py",
    "src/experiments/qwen38_fewshot_eval.py",
    "src/experiments/qwen_fewshot_eval.py",
)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/kip_m4_data_efficiency_20260920_seed9950")
    parser.add_argument("--arms", nargs="+", default=list(ARM_ORDER))
    parser.add_argument("--fractions", nargs="+", default=[str(value) for value in FRACTIONS])
    parser.add_argument("--seed", type=int, default=9950)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-perfect-test", action="store_true",
                        help="Stop an arm after a production fit reaches exact full-test AUROC 1.0")
    parser.add_argument("--batch-size", type=int, help="Override the initial microbatch of every selected arm")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    arms = {arm for group in args.arms for arm in group.split(",")}
    if arms - set(ARM_ORDER):
        parser.error(f"unknown arms: {sorted(arms - set(ARM_ORDER))}")
    args.arms = [arm for arm in ARM_ORDER if arm in arms]
    try:
        args.fractions = sorted({float(value) for group in args.fractions for value in str(group).split(",")})
    except ValueError:
        parser.error("fractions must be numbers")
    if not args.fractions or any(value not in FRACTIONS for value in args.fractions):
        parser.error(f"fractions must be selected from {FRACTIONS}")
    if args.batch_size is not None and (args.batch_size <= 0 or 256 % args.batch_size):
        parser.error("--batch-size must be a positive divisor of the effective batch size 256")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    args.run_root = args.run_root.resolve()
    return args


def training_command(args, arm, fraction, output_dir, batch_size, eval_batch_size, smoke=False):
    command = [sys.executable, "-B", "-u", "-m", "src.experiments.kip_data_efficiency",
               "--arm", arm, "--model-path", str(MODEL_PATHS[arm]),
               "--fraction", str(fraction), "--seed", str(args.seed),
               "--output-dir", str(output_dir), "--batch-size", str(batch_size),
               "--gradient-accumulation-steps", str(256 // batch_size),
               "--eval-batch-size", str(eval_batch_size)]
    if smoke:
        command.append("--smoke")
    elif args.stop_on_perfect_test:
        command.append("--stop-on-perfect-test")
    return command


def environment(args):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(args.seed),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               MKL_NUM_THREADS="8", TRITON_CACHE_DIR=str(args.run_root / "kernel_cache/triton"),
               MPLCONFIGDIR=str(args.run_root / "kernel_cache/matplotlib"),
               TORCHINDUCTOR_CACHE_DIR=str(args.run_root / "kernel_cache/inductor"),
               CUDA_CACHE_PATH=str(args.run_root / "kernel_cache/cuda"))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / ".deps/kip_eff"), str(ROOT)] +
                                       ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def build_plan(args):
    sources = {relative: sha256(ROOT / relative) for relative in SOURCE_FILES if (ROOT / relative).is_file()}
    missing = sorted(set(SOURCE_FILES) - set(sources))
    jobs = []
    for arm in args.arms:
        batch = args.batch_size or MICROBATCHES[arm]
        for smoke, fraction in [(True, args.fractions[0])] + [(False, value) for value in args.fractions]:
            key = f"{arm}/" + ("smoke" if smoke else f"fraction_{fraction:.2f}")
            output = args.run_root / key / "attempt_001"
            jobs.append({"key": key, "arm": arm, "fraction": fraction, "is_smoke": smoke,
                         "initial_batch_size": batch, "eval_batch_size": args.eval_batch_size,
                         "command": training_command(args, arm, fraction, output, batch, args.eval_batch_size, smoke)})
    protocol = {"schema_version": 1, "tag": "kip_m4", "mode": "ordered", "seed": args.seed,
                "arms": args.arms, "fractions": args.fractions, "effective_batch_size": 256,
                "epochs": 20, "patience": 3, "learning_rate": 2e-5,
                "validation_rows": 50000, "test_rows": 50000,
                "fraction_applies_to": "training_only", "selection": "validation_loss",
                "oom_policy": "halve microbatch and evaluation batch; keep effective batch 256; full optimizer OOM retries with CPU offload",
                "model_integrity": "SHA256 payload verification once per pinned checkpoint, metadata revalidated before arm smoke",
                "python": sys.executable, "models": {arm: str(MODEL_PATHS[arm]) for arm in args.arms},
                "source_sha256": sources, "missing_source_files": missing,
                "jobs": jobs, "run_root": str(args.run_root)}
    if args.stop_on_perfect_test:
        protocol["stop_on_perfect_test"] = True
        protocol["saturation_policy"] = "exact full-test AUROC 1.0 from a verified production completion skips larger fractions of that arm"
    protocol["plan_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def freeze_plan(args, plan):
    """Keep an immutable plan and source copies; refuse changing a resumed protocol."""
    if plan["missing_source_files"]:
        raise RuntimeError(f"missing trainer sources: {plan['missing_source_files']}")
    path = args.run_root / "launch_plan.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("run-root already contains a different plan or source revision; use a new --run-root")
        return
    for relative in plan["source_sha256"]:
        destination = args.run_root / "protocol" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    atomic_json(plan, path)


def verify_sources(plan):
    for relative, expected in plan["source_sha256"].items():
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed after queue launch: {relative}; queued training stopped to preserve the protocol")


def validated_metrics(path, job, seed):
    metrics = json.loads(Path(path).read_text())
    if metrics.get("status") != "completed":
        raise ValueError("metrics.json does not mark completed training")
    expected = {"arm": job["arm"], "fraction": job["fraction"], "seed": seed, "is_smoke": job["is_smoke"]}
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(f"metrics identity mismatch for {name}: expected {value!r}, found {metrics.get(name)!r}")
    for name in ("auc", "accuracy", "f1", "loss"):
        value = metrics.get("test", {}).get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"missing or nonfinite test metric: {name}")
    expected_rows = {"train": 256, "val": 128, "test": 128} if job["is_smoke"] else {
        "train": round(400000 * job["fraction"]), "val": 50000, "test": 50000}
    if metrics.get("rows") != expected_rows:
        raise ValueError(f"incorrect data split sizes: {metrics.get('rows')!r}; expected {expected_rows}")
    return metrics


def is_verified_success(record, job, seed):
    if record.get("status") != "completed" or not record.get("metrics_path") or not record.get("metrics_sha256"):
        return False
    try:
        validated_metrics(record["metrics_path"], job, seed)
        return sha256(record["metrics_path"]) == record["metrics_sha256"]
    except (OSError, ValueError, TypeError):
        return False


def apply_saturation(args, arm_jobs):
    """Infer intentional skips from verified full-test results, never smoke or rounded AUC."""
    if not args.stop_on_perfect_test:
        return None
    evidence = None
    for record in sorted(arm_jobs, key=lambda item: item["fraction"]):
        if record["is_smoke"] or not is_verified_success(record, record, args.seed):
            continue
        metrics = validated_metrics(record["metrics_path"], record, args.seed)
        auc = metrics["test"]["auc"]
        if isinstance(auc, bool) or auc != 1.0:
            continue
        evidence = {"job_key": record["key"], "arm": record["arm"], "fraction": record["fraction"],
                    "test_auc": auc, "test_rows": metrics["rows"]["test"],
                    "metrics_path": record["metrics_path"], "metrics_sha256": record["metrics_sha256"]}
        break
    if evidence is None:
        return None
    for record in arm_jobs:
        if (record["is_smoke"] or record["fraction"] <= evidence["fraction"]
                or is_verified_success(record, record, args.seed)):
            continue
        # Preserve existing attempt records and all completed results. Unrun
        # fractions have no performance measurements or completion artifacts.
        record.pop("metrics_path", None)
        record.pop("metrics_sha256", None)
        record.update(status="skipped_saturated", error=None,
                      skip_reason=f"{evidence['arm']} reached exact full-test AUROC 1.0 at fraction {evidence['fraction']}",
                      saturation_evidence=dict(evidence),
                      skipped_at_unix=record.get("skipped_at_unix", time.time()))
    return evidence


def update_summary(args, state):
    rows = []
    for job in state["jobs"]:
        if job["is_smoke"]:
            continue
        row = {"arm": job["arm"], "fraction": job["fraction"], "seed": args.seed,
               "status": job["status"], "train_rows": round(400000 * job["fraction"]),
               "test_auc": "", "test_accuracy": "", "test_f1": "", "test_loss": "",
               "best_epoch": "", "epochs_completed": "", "training_seconds": "",
               "metrics_path": job.get("metrics_path", ""), "error": job.get("error", "")}
        if args.stop_on_perfect_test:
            evidence = job.get("saturation_evidence", {})
            row.update(skip_reason=job.get("skip_reason", ""),
                       saturation_metrics_path=evidence.get("metrics_path", ""),
                       saturation_metrics_sha256=evidence.get("metrics_sha256", ""))
        if is_verified_success(job, job, args.seed):
            metrics = validated_metrics(job["metrics_path"], job, args.seed)
            row.update({f"test_{name}": metrics["test"][name] for name in ("auc", "accuracy", "f1", "loss")})
            row.update({name: metrics.get(name, "") for name in ("best_epoch", "epochs_completed", "training_seconds")})
        rows.append(row)
    if not rows:
        return
    temporary = args.run_root / f"summary.csv.{os.getpid()}.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(args.run_root / "summary.csv")


def publish(args, state):
    state["updated_at_unix"] = time.time()
    atomic_json(state, args.run_root / "queue_status.json")
    update_summary(args, state)


def update_plots(args):
    """Figure export is optional and cannot invalidate completed training."""
    log_path = args.run_root / "plot.log"
    try:
        with log_path.open("a") as log:
            result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/plot_kip_data_efficiency.py"),
                                     "--run-root", str(args.run_root)],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment(args), timeout=180)
        return {"status": "completed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode, "log": str(log_path), "updated_at_unix": time.time()}
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "log": str(log_path), "updated_at_unix": time.time()}


def valid_model_audit(model_path, evidence_path):
    try:
        evidence = json.loads(Path(evidence_path).read_text())
        if (evidence.get("status") != "verified" or evidence.get("verified") is not True
                or evidence.get("revision") != model_path.name
                or evidence.get("model_path") != str(model_path.resolve())):
            return False
        index_path = model_path / "model.safetensors.index.json"
        if sha256(index_path) != evidence.get("index_sha256"):
            return False
        names = set(json.loads(index_path.read_text())["weight_map"].values())
        if not names or names != set(evidence.get("weights", {})):
            return False
        for name in names:
            if Path(name).name != name:
                return False
            shard = (model_path / name).resolve(strict=True)
            stat = shard.stat()
            record = evidence["weights"][name]
            if (record.get("verified") is not True or record.get("stable_during_hash") is not True
                    or record.get("sha256") != record.get("expected_hf_blob_sha256")
                    or record.get("sha256") != shard.name
                    or record.get("blob_path") != str(shard)
                    or record.get("bytes") != stat.st_size or record.get("mtime_ns") != stat.st_mtime_ns):
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def ensure_model_audit(args, plan, state, arm, lock_fds):
    verify_sources(plan)
    model_path = MODEL_PATHS[arm]
    evidence_path = args.run_root / "protocol" / f"model_{model_path.name}_sha256.json"
    if valid_model_audit(model_path, evidence_path):
        return {"status": "verified", "evidence": str(evidence_path), "reused": True}
    log_path = evidence_path.with_suffix(".log")
    command = [sys.executable, "-B", "-u", str(ROOT / "scripts/audit_kip_efficiency_weights.py"),
               "--model-path", str(model_path), "--output", str(evidence_path)]
    state["current_job"] = f"{arm}/model_payload_audit"
    publish(args, state)
    with log_path.open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=environment(args), pass_fds=lock_fds)
        state["child_pid"] = child.pid
        publish(args, state)
        code = child.wait()
    state["child_pid"] = None
    if code or not valid_model_audit(model_path, evidence_path):
        raise RuntimeError(f"checkpoint payload audit failed (exit={code}); inspect {log_path}")
    return {"status": "verified", "evidence": str(evidence_path), "reused": False}


def is_oom(log_path):
    # Keep tracebacks from multi-hour logs bounded in RAM.
    with Path(log_path).open("rb") as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END) - 256 * 1024))
        tail = stream.read().decode(errors="replace").lower()
    return any(marker in tail for marker in ("cuda out of memory", "cuda error: out of memory", "torch.outofmemoryerror"))


def is_optimizer_oom(log_path):
    with Path(log_path).open("rb") as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END) - 256 * 1024))
        tail = stream.read().decode(errors="replace").lower()
    return any(marker in tail for marker in ("optimizer.step(", "adamw.py", "optim/adam.py"))


def run_job(args, plan, state, record, batch, eval_batch, lock_fds, optimizer_offload="none"):
    record.setdefault("attempts", [])
    while True:
        verify_sources(plan)
        job_dir = args.run_root / record["key"]
        job_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        while (job_dir / f"attempt_{index:03d}").exists():
            index += 1
        output = job_dir / f"attempt_{index:03d}"
        # Exclusive creation avoids confusing stale metrics with this process's output.
        output.mkdir()
        log_path = job_dir / f"attempt_{index:03d}.log"
        command = training_command(args, record["arm"], record["fraction"], output, batch, eval_batch, record["is_smoke"])
        if optimizer_offload == "cpu":
            command.extend(["--optimizer-offload", "cpu"])
        attempt = {"output_dir": str(output), "log": str(log_path), "command": command,
                   "batch_size": batch, "gradient_accumulation_steps": 256 // batch,
                   "optimizer_offload": optimizer_offload,
                   "eval_batch_size": eval_batch, "started_at_unix": time.time(), "status": "starting"}
        record["attempts"].append(attempt)
        record.update(status="running", error=None)
        state["current_job"] = record["key"]
        publish(args, state)
        try:
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, env=environment(args), pass_fds=lock_fds)
                attempt.update(status="running", child_pid=child.pid)
                state["child_pid"] = child.pid
                publish(args, state)
                code = child.wait()
            attempt.update(exit_code=code, finished_at_unix=time.time())
            if code:
                raise RuntimeError(f"training exited with code {code}; inspect {log_path}")
            metrics_path = output / "metrics.json"
            validated_metrics(metrics_path, record, args.seed)
            attempt.update(status="completed", metrics_path=str(metrics_path))
            record.update(status="completed", metrics_path=str(metrics_path), metrics_sha256=sha256(metrics_path),
                          batch_size=batch, eval_batch_size=eval_batch, optimizer_offload=optimizer_offload,
                          finished_at_unix=time.time())
            state["child_pid"] = None
            publish(args, state)
            if not record["is_smoke"]:
                state["plot_status"] = update_plots(args)
                publish(args, state)
            return True, batch, eval_batch, optimizer_offload
        except Exception as exc:
            attempt.update(status="failed", error=str(exc), finished_at_unix=time.time())
            record.update(status="failed", error=str(exc), finished_at_unix=time.time())
            state["child_pid"] = None
            oom = log_path.is_file() and is_oom(log_path)
            attempt["out_of_memory"] = oom
            atomic_json(attempt, output / "queue_attempt_error.json")
            use_offload = (oom and record["arm"] == "llama8b_full" and optimizer_offload == "none"
                           and (is_optimizer_oom(log_path) or batch == 1))
            retry = oom and (batch > 1 or eval_batch > 1 or use_offload)
            if use_offload:
                optimizer_offload = "cpu"
                batch = args.batch_size or MICROBATCHES[record["arm"]]
                record["status"] = "retrying_with_cpu_optimizer"
            elif retry:
                # For power-of-two defaults this preserves effective batch exactly.
                candidates = [value for value in range(1, batch) if 256 % value == 0]
                batch = max(candidates, default=1)
                eval_batch = max(1, eval_batch // 2)
                record["status"] = "retrying_after_oom"
            publish(args, state)
            if not retry:
                return False, batch, eval_batch, optimizer_offload


def supervise(args, plan):
    args.run_root.mkdir(parents=True, exist_ok=True)
    global_path = ROOT / "results/.kip_data_efficiency_gpu.lock"
    global_path.parent.mkdir(parents=True, exist_ok=True)
    with (args.run_root / "queue.lock").open("a") as local_lock, global_path.open("a") as gpu_lock:
        # Inherited by each trainer: a surviving orphan also prevents a duplicate queue.
        for lock in (local_lock, gpu_lock):
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("a KIP data-efficiency queue or its trainer is already running") from exc
        freeze_plan(args, plan)
        state_path = args.run_root / "queue_status.json"
        old = json.loads(state_path.read_text()) if state_path.exists() else {}
        if old and old.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("existing queue status belongs to a different plan")
        records = {record["key"]: record for record in old.get("jobs", [])}
        jobs = []
        for job in plan["jobs"]:
            record = {**job, **records.get(job["key"], {})}
            if not is_verified_success(record, job, args.seed):
                record["status"] = "pending"
                for field in ("skip_reason", "saturation_evidence", "skipped_at_unix"):
                    record.pop(field, None)
            jobs.append(record)
        state = {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "status": "running",
                 "supervisor_pid": os.getpid(), "child_pid": None,
                 "started_at_unix": old.get("started_at_unix", time.time()),
                 "resumed_at_unix": time.time() if old else None, "jobs": jobs,
                 "model_audits": old.get("model_audits", {})}
        (args.run_root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        publish(args, state)
        try:
            for arm in args.arms:
                batch = args.batch_size or MICROBATCHES[arm]
                eval_batch = args.eval_batch_size
                optimizer_offload = "none"
                arm_jobs = [job for job in jobs if job["arm"] == arm]
                evidence = apply_saturation(args, arm_jobs)
                if evidence:
                    state.setdefault("saturated_arms", {})[arm] = evidence
                    publish(args, state)
                if all(is_verified_success(job, job, args.seed) or job["status"] == "skipped_saturated"
                       for job in arm_jobs):
                    continue
                try:
                    audit = ensure_model_audit(args, plan, state, arm, (local_lock.fileno(), gpu_lock.fileno()))
                    state.setdefault("model_audits", {})[arm] = audit
                    publish(args, state)
                except Exception as exc:
                    state.setdefault("model_audits", {})[arm] = {"status": "failed", "error": str(exc)}
                    for record in arm_jobs:
                        if not is_verified_success(record, record, args.seed) and record["status"] != "skipped_saturated":
                            record.update(status="failed" if record["is_smoke"] else "skipped", error=str(exc))
                    publish(args, state)
                    continue
                smoke_success = False
                for record in arm_jobs:
                    if record["status"] == "skipped_saturated":
                        continue
                    if is_verified_success(record, record, args.seed):
                        batch = record.get("batch_size", batch)
                        eval_batch = record.get("eval_batch_size", eval_batch)
                        optimizer_offload = record.get("optimizer_offload", optimizer_offload)
                        if record["is_smoke"]:
                            smoke_success = True
                        continue
                    if not record["is_smoke"] and not smoke_success:
                        record.update(status="skipped", error="arm preflight smoke did not pass")
                        publish(args, state)
                        continue
                    success, batch, eval_batch, optimizer_offload = run_job(
                        args, plan, state, record, batch, eval_batch,
                        (local_lock.fileno(), gpu_lock.fileno()), optimizer_offload)
                    if record["is_smoke"]:
                        smoke_success = success
                    elif success:
                        evidence = apply_saturation(args, arm_jobs)
                        if evidence:
                            state.setdefault("saturated_arms", {})[arm] = evidence
                            publish(args, state)
            state.update(status="completed" if all(job["status"] in {"completed", "skipped_saturated"} for job in jobs)
                         else "completed_with_failures", current_job=None, child_pid=None)
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise
        finally:
            state["finished_at_unix"] = time.time()
            publish(args, state)
        return state


def background_command(args):
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--run-root", str(args.run_root), "--arms", *args.arms,
               "--fractions", *[str(value) for value in args.fractions],
               "--seed", str(args.seed), "--eval-batch-size", str(args.eval_batch_size)]
    if args.batch_size:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.stop_on_perfect_test:
        command.append("--stop-on-perfect-test")
    return command


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.background:
        args.run_root.mkdir(parents=True, exist_ok=True)
        # Existing plans are checked in the child while holding both locks.
        with (args.run_root / "supervisor.log").open("a") as log:
            child = subprocess.Popen(background_command(args), cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=environment(args),
                                     start_new_session=True)
        # Wait only for durable supervisor startup, never for GPU training.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"supervisor exited with code {child.returncode}; inspect {args.run_root / 'supervisor.log'}")
            state_path = args.run_root / "queue_status.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                if state.get("supervisor_pid") == child.pid and state.get("status") == "running":
                    print(json.dumps({"status": "running", "supervisor_pid": child.pid,
                                      "run_root": str(args.run_root), "first_arm": args.arms[0],
                                      "queue_status": str(state_path), "summary_csv": str(args.run_root / "summary.csv")}))
                    return
            time.sleep(0.1)
        raise RuntimeError(f"supervisor startup not confirmed after 10 seconds (PID {child.pid}); inspect supervisor.log before relaunching")
    state = supervise(args, plan)
    print(json.dumps({"status": state["status"], "run_root": str(args.run_root)}))


if __name__ == "__main__":
    main()
