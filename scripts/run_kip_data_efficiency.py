#!/usr/bin/env python3
"""Run the KIP-4 and KIP-6 data-efficiency sweep."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.paper_protocol import CONFIG, PROTOCOL_ID, SEEDS

ARM_ORDER = ("qwen27b_qlora", "qwen14b_lora", "llama8b_lora", "llama8b_full")
FRACTIONS = (0.01, 0.10, 0.30, 0.50, 0.75, 1.0)
MODEL_PATHS = {
    "qwen27b_qlora": ROOT / Path("data/model_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "qwen14b_lora": ROOT / "data/model_cache/models--Qwen--Qwen2.5-14B/snapshots/97e1e76335b7017d8f67c08a19d103c0504298c9",
    "llama8b_lora": ROOT / "data/model_cache/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "llama8b_full": ROOT / "data/model_cache/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
}
EFFECTIVE_BATCH_SIZE = CONFIG.effective_batch_size
MICROBATCHES = {arm: 16 for arm in ARM_ORDER}
TAGS = ("kip_m4", "kip_m6")
SOURCE_FILES = (
    "scripts/run_kip_data_efficiency.py",
    "scripts/plot_kip_data_efficiency.py",
    "src/experiments/kip_data_efficiency.py",
    "src/experiments/kip_efficiency_models.py",
    "src/experiments/paper_pretrained_training.py",
    "src/experiments/paper_protocol.py",
    "src/experiments/paper_data.py",
    "src/experiments/kip_llama_training.py",
    "src/experiments/kip_encoder_training.py",
    "src/experiments/kip_cpu_offload.py",
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
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/kip_paper_protocol")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data-root", type=Path, default=ROOT / "data/paper",
                        help="parent of kip_m4/ and kip_m6/ dataset directories")
    inputs.add_argument("--data-dir", type=Path,
                        help="shared CSV directory")
    parser.add_argument("--arms", nargs="+", default=list(ARM_ORDER))
    parser.add_argument("--fractions", nargs="+", default=[str(value) for value in FRACTIONS])
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Run one explicit seed instead of the three-seed default")
    seeds.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds (default: 9550 9551 9552)")
    parser.add_argument("--tags", nargs="+", choices=TAGS, default=list(TAGS))
    parser.add_argument("--model-path", action="append", default=[], metavar="ARM=PATH",
                        help="Local model snapshot for an arm; repeat for multiple arms")
    parser.add_argument("--background", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--dry-run", action="store_true", help="print the KIP configuration (the default)")
    execution.add_argument("--execute", action="store_true", help="start the queue")
    parser.add_argument("--batch-size", type=int, help="Initial microbatch of every selected arm")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    args.seeds = sorted(set(args.seeds or ([args.seed] if args.seed is not None else SEEDS)))
    args.tags = [tag for tag in TAGS if tag in args.tags]
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
    if args.batch_size is not None and (args.batch_size <= 0 or EFFECTIVE_BATCH_SIZE % args.batch_size):
        parser.error("--batch-size must be a positive divisor of the effective batch size 16")
    if args.eval_batch_size < 1:
        parser.error("--eval-batch-size must be positive")
    args.model_paths = dict(MODEL_PATHS)
    for override in args.model_path:
        arm, separator, path = override.partition("=")
        if not separator or arm not in ARM_ORDER or not path:
            parser.error("--model-path requires a known ARM=PATH")
        args.model_paths[arm] = Path(path).expanduser().resolve()
    args.run_root = args.run_root.resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.expanduser().resolve()
    return args


def training_command(args, arm, fraction, output_dir, batch_size, eval_batch_size, smoke=False, *, tag, seed):
    command = [sys.executable, "-B", "-u", "-m", "src.experiments.kip_data_efficiency",
               "--arm", arm, "--model-path", str(args.model_paths[arm]),
               "--data-dir", str(args.data_dir if args.data_dir is not None else args.data_root / tag),
               "--fraction", str(fraction), "--seed", str(seed), "--tag", tag,
               "--output-dir", str(output_dir), "--batch-size", str(batch_size),
               "--gradient-accumulation-steps", str(EFFECTIVE_BATCH_SIZE // batch_size),
               "--eval-batch-size", str(eval_batch_size)]
    if smoke:
        command.append("--smoke")
    return command


def environment(args, seed=None):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=str(args.seeds[0] if seed is None else seed),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
               MKL_NUM_THREADS="8", TRITON_CACHE_DIR=str(args.run_root / "kernel_cache/triton"),
               MPLCONFIGDIR=str(args.run_root / "kernel_cache/matplotlib"),
               TORCHINDUCTOR_CACHE_DIR=str(args.run_root / "kernel_cache/inductor"),
               CUDA_CACHE_PATH=str(args.run_root / "kernel_cache/cuda"))
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def build_plan(args):
    sources = {relative: sha256(ROOT / relative) for relative in SOURCE_FILES if (ROOT / relative).is_file()}
    missing = sorted(set(SOURCE_FILES) - set(sources))
    jobs = []
    for arm in args.arms:
        batch = args.batch_size or MICROBATCHES[arm]
        for tag in args.tags:
            for seed in args.seeds:
                for smoke, fraction in [(True, args.fractions[0])] + [(False, value) for value in args.fractions]:
                    key = f"{arm}/{tag}/seed{seed}/" + ("smoke" if smoke else f"fraction_{fraction:.2f}")
                    output = args.run_root / key / "attempt_001"
                    jobs.append({"key": key, "arm": arm, "tag": tag, "seed": seed,
                                 "fraction": fraction, "is_smoke": smoke,
                                 "initial_batch_size": batch, "eval_batch_size": args.eval_batch_size,
                                 "command": training_command(args, arm, fraction, output, batch,
                                                             args.eval_batch_size, smoke, tag=tag, seed=seed)})
    protocol = {"schema_version": 2, "protocol": "paper_v2", "protocol_id": PROTOCOL_ID, "tags": args.tags,
                "mode": "ordered", "seeds": args.seeds,
                "arms": args.arms, "fractions": args.fractions,
                "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                "epochs": 20, "patience": 3, "learning_rate": 2e-5, "max_length": 200,
                "loss": "classification_cross_entropy_plus_prompt_next_token_cross_entropy",
                "validation_rows": 50000, "test_rows": 50000,
                "fraction_applies_to": "training_only", "selection": "validation_loss",
                "stopping_rule": "validation_loss_patience",
                "oom_policy": "halve microbatch and evaluation batch; keep effective batch 16; full optimizer OOM retries with CPU offload",
                "python": sys.executable, "models": {arm: str(args.model_paths[arm]) for arm in args.arms},
                "data_directories": {tag: str(args.data_dir if args.data_dir is not None else args.data_root / tag)
                                     for tag in args.tags},
                "source_sha256": sources, "missing_source_files": missing,
                "jobs": jobs, "run_root": str(args.run_root)}
    protocol["plan_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


def freeze_plan(args, plan):
    if plan["missing_source_files"]:
        raise RuntimeError(f"missing trainer sources: {plan['missing_source_files']}")
    path = args.run_root / "launch_plan.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old.get("plan_sha256") != plan["plan_sha256"]:
            raise RuntimeError("run-root already contains a different plan; use a new --run-root")
        return
    for relative in plan["source_sha256"]:
        destination = args.run_root / "protocol" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    atomic_json(plan, path)


def verify_sources(plan):
    for relative, expected in plan["source_sha256"].items():
        if not (ROOT / relative).is_file() or sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed after queue launch: {relative}")


def validated_metrics(path, job, seed=None):
    metrics = json.loads(Path(path).read_text())
    if metrics.get("status") != "completed":
        raise ValueError("metrics.json does not mark completed training")
    expected = {"arm": job["arm"], "fraction": job["fraction"], "seed": job["seed"] if seed is None else seed, "tag": job["tag"], "is_smoke": job["is_smoke"]}
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(f"metrics identity mismatch for {name}: expected {value!r}, found {metrics.get(name)!r}")
    if (metrics.get("protocol") != "paper_v2" or metrics.get("selected_by") != "validation_loss"
            or metrics.get("stopping_rule") != "validation_loss_patience"):
        raise ValueError("metrics do not match the validation-loss stopping protocol")
    for name in ("auc", "accuracy", "f1", "loss"):
        value = metrics.get("test", {}).get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"missing or nonfinite test metric: {name}")
    expected_rows = {"train": 256, "val": 128, "test": 128} if job["is_smoke"] else {
        "train": round(400000 * job["fraction"]), "val": 50000, "test": 50000}
    if metrics.get("rows") != expected_rows:
        raise ValueError(f"incorrect data split sizes: {metrics.get('rows')!r}; expected {expected_rows}")
    return metrics


def is_verified_success(record, job, seed=None):
    if record.get("status") != "completed" or not record.get("metrics_path") or not record.get("metrics_sha256"):
        return False
    try:
        validated_metrics(record["metrics_path"], job, seed)
        return sha256(record["metrics_path"]) == record["metrics_sha256"]
    except (OSError, ValueError, TypeError):
        return False


def update_summary(args, state):
    rows = []
    for job in state["jobs"]:
        if job["is_smoke"]:
            continue
        row = {"arm": job["arm"], "fraction": job["fraction"], "seed": job["seed"], "tag": job["tag"],
               "status": job["status"], "train_rows": round(400000 * job["fraction"]),
               "test_auc": "", "test_accuracy": "", "test_f1": "", "test_loss": "",
               "best_epoch": "", "epochs_completed": "", "training_seconds": "",
               "metrics_path": job.get("metrics_path", ""), "error": job.get("error", "")}
        if is_verified_success(job, job):
            metrics = validated_metrics(job["metrics_path"], job)
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


def is_oom(log_path):
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
        output.mkdir()
        log_path = job_dir / f"attempt_{index:03d}.log"
        command = training_command(args, record["arm"], record["fraction"], output, batch, eval_batch, record["is_smoke"],
                                   tag=record["tag"], seed=record["seed"])
        if optimizer_offload == "cpu":
            command.extend(["--optimizer-offload", "cpu"])
        attempt = {"output_dir": str(output), "log": str(log_path), "command": command,
                   "batch_size": batch, "gradient_accumulation_steps": EFFECTIVE_BATCH_SIZE // batch,
                   "optimizer_offload": optimizer_offload,
                   "eval_batch_size": eval_batch, "started_at_unix": time.time(), "status": "starting"}
        record["attempts"].append(attempt)
        record.update(status="running", error=None)
        state["current_job"] = record["key"]
        publish(args, state)
        try:
            with log_path.open("x") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, env=environment(args, record["seed"]), pass_fds=lock_fds)
                attempt.update(status="running", child_pid=child.pid)
                state["child_pid"] = child.pid
                publish(args, state)
                code = child.wait()
            attempt.update(exit_code=code, finished_at_unix=time.time())
            if code:
                raise RuntimeError(f"training exited with code {code}; inspect {log_path}")
            metrics_path = output / "metrics.json"
            validated_metrics(metrics_path, record)
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
                candidates = [value for value in range(1, batch) if EFFECTIVE_BATCH_SIZE % value == 0]
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
            if not is_verified_success(record, job):
                record["status"] = "pending"
            jobs.append(record)
        state = {"schema_version": 1, "plan_sha256": plan["plan_sha256"], "status": "running",
                 "supervisor_pid": os.getpid(), "child_pid": None,
                 "started_at_unix": old.get("started_at_unix", time.time()),
                 "resumed_at_unix": time.time() if old else None, "jobs": jobs}
        (args.run_root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        publish(args, state)
        try:
            for arm in args.arms:
                arm_jobs = [job for job in jobs if job["arm"] == arm]
                if all(is_verified_success(job, job) for job in arm_jobs):
                    continue
                for tag in args.tags:
                    for seed in args.seeds:
                        batch = args.batch_size or MICROBATCHES[arm]
                        eval_batch = args.eval_batch_size
                        optimizer_offload = "none"
                        smoke_success = False
                        group = [job for job in arm_jobs if job["tag"] == tag and job["seed"] == seed]
                        for record in group:
                            if is_verified_success(record, record):
                                batch = record.get("batch_size", batch)
                                eval_batch = record.get("eval_batch_size", eval_batch)
                                optimizer_offload = record.get("optimizer_offload", optimizer_offload)
                                if record["is_smoke"]:
                                    smoke_success = True
                                continue
                            if not record["is_smoke"] and not smoke_success:
                                record.update(status="skipped", error="arm/task/seed preflight smoke did not pass")
                                publish(args, state)
                                continue
                            success, batch, eval_batch, optimizer_offload = run_job(
                                args, plan, state, record, batch, eval_batch,
                                (local_lock.fileno(), gpu_lock.fileno()), optimizer_offload)
                            if record["is_smoke"]:
                                smoke_success = success
            state.update(status="completed" if all(job["status"] == "completed" for job in jobs)
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
               "--seeds", *[str(seed) for seed in args.seeds], "--tags", *args.tags,
               "--eval-batch-size", str(args.eval_batch_size), "--execute"]
    command.extend(["--data-dir", str(args.data_dir)] if args.data_dir is not None else
                   ["--data-root", str(args.data_root)])
    for arm in args.arms:
        command.extend(["--model-path", f"{arm}={args.model_paths[arm]}"])
    if args.batch_size:
        command.extend(["--batch-size", str(args.batch_size)])
    return command


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if args.background:
        args.run_root.mkdir(parents=True, exist_ok=True)
        with (args.run_root / "supervisor.log").open("a") as log:
            child = subprocess.Popen(background_command(args), cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=environment(args),
                                     start_new_session=True)
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
        raise RuntimeError(f"supervisor startup not confirmed after 10 seconds (PID {child.pid}); inspect supervisor.log")
    state = supervise(args, plan)
    print(json.dumps({"status": state["status"], "run_root": str(args.run_root)}))


if __name__ == "__main__":
    main()
