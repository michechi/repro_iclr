#!/usr/bin/env python3
"""Extend the existing attention queue with CKD attention and Qwen OC fits/audits."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_kip_checkpoint_attention as prior
from src.analysis.kip_checkpoint_attention import digest, read_json, write_json
from src.analysis.oc_attention_core import KEYS, LAG, ORACLE_SOURCE, label

RUN_ROOT = ROOT / "results/qwen_oc_clinical_attention_20260922_seed9950"
ATTENTION_ROOT = ROOT / "results/kip_checkpoint_attention_20260922"
CKD_ROOT = ROOT / "results/qwen38_27b_ckd_20260921_seed42"
OC_DATA = ROOT / "results/qwen25_14b_fewshot_20260918/data/oc_corrected"
CLINICAL_DATA = ROOT / "data/processed/mimic_training"
SOURCES = (
    "scripts/run_qwen_oc_clinical_audits.py", "src/experiments/qwen_oc_training.py",
    "src/analysis/qwen_oc_audit.py", "src/analysis/oc_attention_core.py", "src/analysis/ckd_attention_audit.py",
    "docs/qwen_oc_clinical_attention_20260922.md", "scripts/run_kip_checkpoint_attention.py",
    "src/analysis/kip_checkpoint_attention.py", "src/analysis/kip_attention_core.py",
    "src/experiments/kip_efficiency_models.py", "src/experiments/kip_encoder_training.py",
    "src/experiments/kip_llama_training.py", "src/experiments/kip_data_efficiency.py",
    "src/experiments/qwen38_fewshot_eval.py", "src/mimic/qwen_ckd_training.py",
    "repro/src/analysis/mechanism_id/scripts/common.py",
)


def prepare(args):
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer
    from src.mimic.qwen_ckd_training import sequence_prompt
    from src.analysis.ckd_attention_audit import clinical_groups
    root = args.run_root
    if (root / "launch_plan.json").exists():
        raise FileExistsError("extension plan already exists; resume instead")
    root.mkdir(parents=True, exist_ok=True)
    export = read_json(OC_DATA / "export_manifest.json")
    if export["mechanism"]["key_letters"] != list(KEYS) or export["mechanism"]["lag"] != LAG:
        raise ValueError("canonical OC parameters differ from audit implementation")
    plan = dict(schema_version=1, task="qwen_oc_and_clinical_attention", seed=9950, tags=args.tags,
        data_dir=str(OC_DATA), clinical_data_dir=str(CLINICAL_DATA), ckd_root=str(CKD_ROOT),
        attention_samples=1000, clinical_samples=1000, perturbations=200,
        original_attention_root=str(ATTENTION_ROOT), original_attention_plan_sha256=digest(ATTENTION_ROOT / "launch_plan.json"),
        ckd_plan_sha256=digest(CKD_ROOT / "launch_plan.json"),
        model_source=read_json(ROOT / "results/kip_m4_data_efficiency_20260920_seed9950/qwen27b_qlora/fraction_0.30/attempt_001/manifest.json")["config"]["model_source"],
        recipe=dict(arm="qwen27b_qlora", epochs=20, patience=3, learning_rate=2e-5, weight_decay=.01,
                    warmup_ratio=.06, batch_size=128, accumulation=2, effective_batch=256, eval_batch_size=128,
                    selection="minimum_validation_loss", test_informed_stopping=False,
                    fresh_pretrained_initialization=True, train_rows=400000, val_rows=50000, test_rows=50000),
        schedule="finish CKD -> 22 existing KIP analyses -> paired clinical attention -> fresh OC fit and attention -> Llama70",
        jobs=[dict(key="clinical_attention", kind="clinical_attention", status="pending")],
        input_sha256={str(OC_DATA / "export_manifest.json"): digest(OC_DATA / "export_manifest.json")},
        created_at_unix=time.time())
    tokenizer_path = ROOT / "results/kip_m4_data_efficiency_20260920_seed9950/qwen27b_qlora/fraction_0.30/attempt_001/tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    evidence = dict(oc={}, clinical={})
    all_sequences = set()
    for split, count in (("train", 400000), ("val", 50000), ("test", 50000)):
        xp, yp = OC_DATA / f"X_{split}_ocdet.csv", OC_DATA / f"y_{split}_ocdet.csv"
        x = pd.read_csv(xp, dtype=str, keep_default_na=False)["Sequences"].tolist()
        y = pd.read_csv(yp)["Outcome"].to_numpy(dtype=int)
        if len(x) != count or len(y) != count or len(set(x)) != count or all_sequences.intersection(x):
            raise ValueError("OC split size, uniqueness, or split-disjointness failure")
        all_sequences.update(x)
        predicted = np.array([label(s) for s in x])
        if not np.array_equal(predicted, y):
            raise ValueError("canonical OC oracle does not match every clean label")
        minimum, maximum = 10**9, 0
        for start in range(0, count, 2048):
            texts = ["Sequential events: " + s.replace("\x1f", " ") for s in x[start:start+2048]]
            tokens = tokenizer(texts, truncation=False, return_attention_mask=False)["input_ids"]
            minimum = min(minimum, min(map(len, tokens))); maximum = max(maximum, max(map(len, tokens)))
        if maximum > 64:
            raise ValueError("OC training would require truncation")
        for tag in set(args.tags) | {"ocdet"}:
            for kind in ("X", "y"):
                p = OC_DATA / f"{kind}_{split}_{tag}.csv"
                actual = digest(p)
                if export["splits"][split]["outputs"][p.name]["sha256"] != actual:
                    raise ValueError("OC export checksum changed")
                plan["input_sha256"][str(p)] = actual
        evidence["oc"][split] = dict(rows=count, positives=int(y.sum()), oracle_mismatches=0,
            min_tokens=minimum, max_tokens=maximum, duplicate_sequences=0, cross_split_overlaps=0)
        print(f"OC {split}: {count} labels verified, tokens {minimum}..{maximum}", flush=True)
    # Complete test-set tokenizer/paired-code audit; source inputs stay private on this VM.
    previous, previous_labels = None, None
    for condition in ("ordered", "shuffled"):
        xp, yp = CLINICAL_DATA / f"X_test_{condition}.csv", CLINICAL_DATA / f"y_test_{condition}.csv"
        x = pd.read_csv(xp, dtype=str, keep_default_na=False)["Sequences"].tolist()
        y = pd.read_csv(yp)["Outcome"].to_numpy(dtype=int)
        if len(x) != 2963 or len(y) != 2963:
            raise ValueError("clinical test count differs from CKD protocol")
        minimum, maximum = 10**9, 0
        multisets = []
        for s in x:
            events = s.split("\x1f"); multisets.append(sorted(events))
            encoded = tokenizer(sequence_prompt(s), truncation=False, return_offsets_mapping=True)
            clinical_groups(encoded["offset_mapping"], events)
            minimum = min(minimum, len(encoded["input_ids"])); maximum = max(maximum, len(encoded["input_ids"]))
        if maximum > 16384:
            raise ValueError("clinical attention would require truncation")
        if previous is not None and (previous != multisets or not np.array_equal(y, previous_labels)):
            raise ValueError("ordered/shuffled test patient rows are not paired")
        previous, previous_labels = multisets, y
        for p in (xp, yp):
            actual = digest(p)
            if read_json(CKD_ROOT / "launch_plan.json")["data_sha256"][p.name] != actual:
                raise ValueError("clinical test inputs differ from the frozen CKD training plan")
            plan["input_sha256"][str(p)] = actual
        evidence["clinical"][condition] = dict(rows=len(x), positives=int(y.sum()), min_tokens=minimum,
                                              max_tokens=maximum, unmapped_events=0)
    for tag in args.tags:
        plan["jobs"] += [dict(key=f"fit_{tag}", kind="fit", tag=tag, status="pending"),
                         dict(key=f"attention_{tag}", kind="oc_attention", tag=tag, status="pending")]
    plan["source_sha256"] = {s: digest(ROOT / s) for s in SOURCES}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    write_json(root / "preflight.json", evidence)
    write_json(root / "launch_plan.json", plan)
    write_json(root / "queue_status.json", dict(status="prepared", jobs=plan["jobs"], plan_sha256=plan["plan_sha256"],
                supervisor_pid=None, child_pid=None, current_job=None, updated_at_unix=time.time()))
    print(json.dumps(dict(plan_sha256=plan["plan_sha256"], jobs=[j["key"] for j in plan["jobs"]])), flush=True)


def verify_plan(plan):
    for relative, h in plan["source_sha256"].items():
        if digest(ROOT / relative) != h:
            raise ValueError(f"frozen source changed: {relative}")
    for path, h in plan["input_sha256"].items():
        if digest(path) != h:
            raise ValueError(f"frozen input changed: {path}")
    if digest(ATTENTION_ROOT / "launch_plan.json") != plan["original_attention_plan_sha256"]:
        raise ValueError("original attention plan changed")
    if digest(CKD_ROOT / "launch_plan.json") != plan["ckd_plan_sha256"]:
        raise ValueError("CKD training plan changed")


def command_for(job, root, plan, batch=128, attempt=1):
    if job["kind"] == "fit":
        output = root / "fits" / job["tag"] / f"attempt_{attempt:03d}"
        command = [sys.executable, "-B", "-u", "-m", "src.experiments.qwen_oc_training", "--tag", job["tag"],
            "--model-path", plan["model_source"]["snapshot"], "--output-dir", str(output), "--data-dir", plan["data_dir"],
            "--seed", str(plan["seed"]), "--batch-size", str(batch), "--gradient-accumulation-steps", str(256//batch),
            "--eval-batch-size", str(batch), "--threads", "4"]
    elif job["kind"] == "clinical_attention":
        output = root / "clinical_attention"
        command = [sys.executable, "-B", "-u", "-m", "src.analysis.ckd_attention_audit", "--run-root", str(root)]
    else:
        output = root / "attention" / job["tag"]
        command = [sys.executable, "-B", "-u", "-m", "src.analysis.qwen_oc_audit", "--run-root", str(root), "--tag", job["tag"]]
    return command, output


def inherited_priority_lock_fds():
    """Keep the original attention lock alive if this supervisor dies during a new worker."""
    target = (ATTENTION_ROOT / "queue.lock").stat()
    result = []
    for p in Path("/proc/self/fd").iterdir():
        try:
            st = os.fstat(int(p.name))
            if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
                result.append(int(p.name))
        except (OSError, ValueError):
            pass
    return result


def run_additional(root, outer_fd):
    plan = read_json(root / "launch_plan.json")
    verify_plan(plan)
    state = read_json(root / "queue_status.json")
    state.update(supervisor_pid=os.getpid(), status="waiting_for_gpu", current_job=None, child_pid=None, error=None)

    def progress(status=None, **details):
        if status: state["status"] = status
        state.update(details, updated_at_unix=time.time()); write_json(root / "queue_status.json", state)

    with prior.GPU_LOCK.open("a+") as gpu:
        while True:
            try:
                fcntl.flock(gpu.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except BlockingIOError:
                progress("waiting_for_gpu"); time.sleep(15)
        if subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip():
            raise RuntimeError("unexpected compute process on GPU")
        for job in state["jobs"]:
            if job["status"] == "completed":
                metrics = read_json(Path(job["output_dir"]) / "metrics.json")
                if metrics.get("status") != "completed": raise ValueError("completed job lacks valid metrics")
                continue
            verify_plan(plan)
            if shutil.disk_usage(root).free < 256 * 1024**2:
                raise OSError("less than 256 MiB free; preserving existing checkpoints")
            batch, attempt = 128, 1
            if job.get("attempts"):
                attempt = len(job["attempts"]) + 1
            while True:
                command, output = command_for(job, root, plan, batch, attempt)
                job.update(status="running", output_dir=str(output))
                log = root / "logs" / f"{job['key']}_attempt{attempt:03d}.log"
                log.parent.mkdir(exist_ok=True)
                job.setdefault("attempts", []).append(dict(command=command, output_dir=str(output), log=str(log), batch=batch))
                with log.open("a") as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=prior.environment(root), stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT,
                        pass_fds=tuple(set([outer_fd, gpu.fileno()] + inherited_priority_lock_fds())))
                    progress("running", current_job=job["key"], child_pid=child.pid)
                    while child.poll() is None:
                        time.sleep(15); progress()
                job["attempts"][-1]["exit_code"] = child.returncode
                if child.returncode == 0 and (output / "metrics.json").exists() and read_json(output / "metrics.json").get("status") == "completed":
                    job["status"] = "completed"; progress(child_pid=None); break
                tail = log.read_text(errors="replace")[-16000:].lower()
                if job["kind"] == "fit" and "out of memory" in tail and batch > 1:
                    batch //= 2; attempt += 1; progress("retrying_oom", child_pid=None); continue
                job["status"] = "failed"; progress("failed", child_pid=None, error=f"{job['key']} failed; see {log}")
                raise RuntimeError(f"extension job failed: {job['key']}")
        progress("completed", current_job=None, child_pid=None)


def supervise(args):
    root = args.run_root
    with (root / "queue.lock").open("a+") as outer:
        fcntl.flock(outer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_json(root / "launch_plan.json"); verify_plan(plan)
        state = read_json(root / "queue_status.json")
        state.update(status="waiting_for_existing_attention", supervisor_pid=os.getpid(), error=None, updated_at_unix=time.time(),
                     upstream_status_path=str(ATTENTION_ROOT / "queue_status.json"))
        write_json(root / "queue_status.json", state)
        original_resume = prior.resume_waiter

        def extended_resume(record, attention_root):
            run_additional(root, outer.fileno())
            original_resume(record, attention_root)
            state = read_json(root / "queue_status.json")
            state.update(llama70_resumed=True, updated_at_unix=time.time()); write_json(root / "queue_status.json", state)

        # Explicit scheduling extension in this process only. Original on-disk code
        # and all 22 jobs/provenance remain byte-identical.
        prior.resume_waiter = extended_resume
        try:
            code = prior.run_queue(argparse.Namespace(run_root=ATTENTION_ROOT))
            if code and read_json(ATTENTION_ROOT / "queue_status.json").get("status") == "blocked_analysis_failures":
                run_additional(root, outer.fileno())
                state = read_json(root / "queue_status.json")
                state.update(status="completed_but_prior_attention_failed", llama70_resumed=False,
                             updated_at_unix=time.time()); write_json(root / "queue_status.json", state)
            return code
        except BaseException as error:
            state = read_json(root / "queue_status.json")
            state.update(status="failed", error=f"{type(error).__name__}: {error}", updated_at_unix=time.time())
            write_json(root / "queue_status.json", state); raise
        finally:
            prior.resume_waiter = original_resume


def validate_idle_attention(state, identity):
    if state.get("status") != "waiting_for_ckd" or state.get("child_pid") or state.get("current_job"):
        raise RuntimeError("attention supervisor is no longer idle; refusing to replace active work")
    if identity["pid"] != state.get("supervisor_pid") or identity["cwd"] != str(ROOT):
        raise RuntimeError("attention supervisor identity mismatch")
    command = [x.decode() if isinstance(x, bytes) else x for x in identity["command"]]
    if str(ROOT / "scripts/run_kip_checkpoint_attention.py") not in command and "scripts/run_kip_checkpoint_attention.py" not in command:
        raise RuntimeError("expected the original attention supervisor")


def background(args):
    plan = read_json(args.run_root / "launch_plan.json"); verify_plan(plan)
    state = read_json(ATTENTION_ROOT / "queue_status.json")
    identity = prior.process_identity(state["supervisor_pid"])
    validate_idle_attention(state, identity)
    os.kill(identity["pid"], signal.SIGSTOP)
    try:
        for _ in range(100):
            actual = prior.process_identity(identity["pid"])
            if actual["state"] in ("T", "t"): break
            time.sleep(.02)
        if actual["start_ticks"] != identity["start_ticks"] or actual["state"] not in ("T", "t"):
            raise RuntimeError("attention process identity changed")
        validate_idle_attention(read_json(ATTENTION_ROOT / "queue_status.json"), actual)
        children = (Path("/proc") / str(identity["pid"]) / "task" / str(identity["pid"]) / "children").read_text().strip()
        if children: raise RuntimeError("attention waiter spawned a child")
    except Exception:
        os.kill(identity["pid"], signal.SIGCONT); raise
    write_json(args.run_root / "supervisor_replacement.json", dict(old_pid=identity["pid"], old_start_ticks=identity["start_ticks"],
        old_command=[x.decode() for x in identity["command"]], reason="User adds clinical attention and OC fit/audit before Llama70",
        replaced_at_unix=time.time(), original_attention_plan_sha256=plan["original_attention_plan_sha256"]))
    os.kill(identity["pid"], signal.SIGTERM)
    os.kill(identity["pid"], signal.SIGCONT)
    for _ in range(100):
        try:
            if prior.process_identity(identity["pid"])["state"] == "Z": break
        except (FileNotFoundError, ProcessLookupError): break
        time.sleep(.05)
    else: raise RuntimeError("idle attention supervisor did not exit")
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--run-root", str(args.run_root)]
    with (args.run_root / "supervisor.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=prior.environment(args.run_root), start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    (args.run_root / "supervisor.pid").write_text(str(child.pid) + "\n")
    (ATTENTION_ROOT / "supervisor.pid").write_text(str(child.pid) + "\n")
    for _ in range(150):
        time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError("extended supervisor exited; Llama70 stays held; inspect supervisor.log")
        active = read_json(ATTENTION_ROOT / "queue_status.json")
        if active.get("supervisor_pid") == child.pid and active.get("status") == "waiting_for_ckd":
            print(json.dumps(dict(supervisor_pid=child.pid, status="waiting_for_ckd", schedule=plan["schedule"],
                                 extension_root=str(args.run_root))))
            return
    raise RuntimeError("extended startup not acknowledged; inspect status before retrying")


def validate_resume_states(extension, attention):
    if extension.get("status") not in {"failed", "completed_but_prior_attention_failed"}:
        raise RuntimeError("only a failed extension queue can be resumed")
    if attention.get("status") not in {"failed", "blocked_analysis_failures"}:
        raise RuntimeError("the original analysis queue is not in a retryable state")
    if extension.get("child_pid") or attention.get("child_pid"):
        raise RuntimeError("a queue still reports an active child")


def resume_background(args):
    """Resume failed analyses without replacing a live supervisor or redoing successes."""
    root = args.run_root
    plan = read_json(root / "launch_plan.json")
    verify_plan(plan)
    with (root / "queue.lock").open("a+") as outer, (ATTENTION_ROOT / "queue.lock").open("a+") as original:
        for lock in (outer, original):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        extension = read_json(root / "queue_status.json")
        attention = read_json(ATTENTION_ROOT / "queue_status.json")
        validate_resume_states(extension, attention)
        for pid in {extension.get("supervisor_pid"), attention.get("supervisor_pid")} - {None}:
            try:
                identity = prior.process_identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if identity["state"] != "Z":
                raise RuntimeError(f"prior supervisor PID {pid} is still alive")
        hold = read_json(ATTENTION_ROOT / "llama70_priority_hold.json")
        identity = prior.process_identity(hold["pid"])
        prior.validate_waiting_llama(read_json(prior.LLAMA), identity)
        if (identity["start_ticks"] != hold["start_ticks"] or identity["state"] not in {"T", "t"}
                or digest(prior.LLAMA.parent / "launch_plan.json") != hold["llama_plan_sha256"]):
            raise RuntimeError("original Llama70 priority hold no longer matches")
        if not prior.ckd_complete(read_json(prior.CKD)):
            raise RuntimeError("CKD has not completed successfully")
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--run-root", str(root)]
    with (root / "supervisor.log").open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=prior.environment(root), start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    for queue_root in (root, ATTENTION_ROOT):
        (queue_root / "supervisor.pid").write_text(str(child.pid) + "\n")
    for _ in range(150):
        time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError("resumed supervisor exited; inspect supervisor.log")
        active = read_json(ATTENTION_ROOT / "queue_status.json")
        if active.get("supervisor_pid") == child.pid and active.get("status") in {
                "running", "waiting_for_ckd", "waiting_for_gpu_lock", "analysis_completed"}:
            print(json.dumps(dict(supervisor_pid=child.pid, status=active["status"],
                                  schedule=plan["schedule"], extension_root=str(root))))
            return
    raise RuntimeError("resumed startup not acknowledged; inspect status before retrying")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--tags", nargs="+", choices=("ocdet", "ocnoise"), default=["ocdet"])
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--resume-background", action="store_true")
    args = parser.parse_args(); args.run_root = args.run_root.resolve()
    parser.error("This historical supervisor controls other queues and is disabled in the public release. "
                 "Use the direct training and analysis modules documented in REPRODUCING.md.")
    if len(set(args.tags)) != len(args.tags): parser.error("duplicate OC tags")
    if sum([args.prepare, args.background, args.resume_background]) > 1:
        parser.error("choose only one of --prepare, --background, --resume-background")
    if args.prepare: prepare(args)
    elif args.resume_background: resume_background(args)
    elif args.background: background(args)
    else: raise SystemExit(supervise(args))


if __name__ == "__main__":
    main()
