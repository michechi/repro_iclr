#!/usr/bin/env python3
"""Detached priority queue: CKD -> saved-checkpoint attention -> existing Llama70 queue."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analysis.kip_checkpoint_attention import DEFAULT_RUN, build_plan, digest, read_json, write_json

CKD = ROOT / "results/qwen38_27b_ckd_20260921_seed42/queue_status.json"
LLAMA = ROOT / "results/kip_m4_llama70b_qlora_20260921_seed9950/queue_status.json"
GPU_LOCK = ROOT / "results/.kip_data_efficiency_gpu.lock"


def process_identity(pid):
    base = Path("/proc") / str(int(pid))
    fields = (base / "stat").read_text().rpartition(")")[2].split()
    return dict(pid=int(pid), start_ticks=fields[19], state=fields[0],
                command=(base / "cmdline").read_bytes().split(b"\0")[:-1],
                cwd=os.readlink(base / "cwd"))


def validate_waiting_llama(state, identity):
    if state.get("status") != "waiting_for_ckd" or state.get("child_pid") or state.get("current_job"):
        raise RuntimeError("Llama70 is no longer an idle prerequisite waiter; refusing to interrupt active work")
    if identity["pid"] != state.get("supervisor_pid") or identity["cwd"] != str(ROOT):
        raise RuntimeError("Llama70 supervisor identity mismatch")
    command = [x.decode() if isinstance(x, bytes) else x for x in identity["command"]]
    if not any(x in ("scripts/run_kip_llama70.py", str(ROOT / "scripts/run_kip_llama70.py")) for x in command):
        raise RuntimeError("PID does not belong to the expected Llama70 queue")
    if any(j.get("status") != "pending" for j in state.get("jobs", [])):
        raise RuntimeError("Llama70 has already begun a job")


def pause_waiter(run_root, saved=None):
    state = read_json(LLAMA)
    identity = process_identity(state["supervisor_pid"])
    validate_waiting_llama(state, identity)
    if saved:
        if saved["pid"] != identity["pid"] or saved["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("saved Llama70 PID was replaced")
    elif identity["state"] in ("T", "t"):
        raise RuntimeError("Llama70 was already suspended by another owner")
    serializable = {**identity, "command": [x.decode() for x in identity["command"]]}
    record = dict(**serializable, llama_plan_sha256=digest(LLAMA.parent / "launch_plan.json"),
                  authorization="available-checkpoint attention prioritized over the Llama-70B queue", paused_at_unix=time.time())
    # Record identity first so a restart can recover even if interrupted just after SIGSTOP.
    write_json(Path(run_root) / "llama70_priority_hold.json", record)
    os.kill(identity["pid"], signal.SIGSTOP)
    for _ in range(50):
        actual = process_identity(identity["pid"])
        if actual["start_ticks"] != identity["start_ticks"]:
            raise RuntimeError("Llama70 PID changed during suspension")
        if actual["state"] in ("T", "t"):
            break
        time.sleep(.02)
    else:
        raise RuntimeError("Llama70 did not enter stopped state")
    try:
        validate_waiting_llama(read_json(LLAMA), actual)
        children = (Path("/proc") / str(actual["pid"]) / "task" / str(actual["pid"]) / "children").read_text().strip()
        if children:
            raise RuntimeError("Llama70 spawned a child while the priority hold was being applied")
    except Exception:
        os.kill(actual["pid"], signal.SIGCONT)
        raise
    return record


def resume_waiter(record, run_root):
    identity = process_identity(record["pid"])
    if (identity["start_ticks"] != record["start_ticks"] or identity["state"] not in ("T", "t")
            or digest(LLAMA.parent / "launch_plan.json") != record["llama_plan_sha256"]):
        raise RuntimeError("cannot safely resume the original Llama70 waiter")
    validate_waiting_llama(read_json(LLAMA), identity)
    os.kill(record["pid"], signal.SIGCONT)
    write_json(Path(run_root) / "llama70_resumed.json", dict(pid=record["pid"], resumed_at_unix=time.time(),
                                                            original_plan_sha256=record["llama_plan_sha256"]))


def ckd_complete(state):
    return (state.get("status") == "completed" and not state.get("child_pid")
            and bool(state.get("jobs")) and all(j.get("status") == "completed" for j in state["jobs"]))


def environment(run_root):
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4")
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / ".deps/kip_eff"), str(ROOT)))
    for variable, folder in (("MPLCONFIGDIR", "matplotlib"), ("TRITON_CACHE_DIR", "triton"),
                             ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("CUDA_CACHE_PATH", "cuda")):
        env[variable] = str(Path(run_root) / "kernel_cache" / folder)
    return env


def run_queue(args):
    root = args.run_root
    root.mkdir(parents=True, exist_ok=True)
    with (root / "queue.lock").open("a+") as local:
        fcntl.flock(local.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read_json(root / "launch_plan.json")
        for name, expected in plan["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise RuntimeError(f"frozen source changed: {name}")
        old = read_json(root / "queue_status.json") if (root / "queue_status.json").exists() else {}
        if old.get("status") == "completed":
            raise RuntimeError("queue is already completed")
        state = dict(status="initializing", supervisor_pid=os.getpid(), child_pid=None, current_job=None,
                     plan_sha256=plan["plan_sha256"], jobs=[dict(key=j["key"], status="pending") for j in plan["jobs"]])
        for job in state["jobs"]:
            p = root / "jobs" / job["key"] / "metrics.json"
            if p.exists() and read_json(p).get("status") == "completed":
                job["status"] = "completed"

        def progress(status=None, **kwargs):
            if status:
                state["status"] = status
            state.update(kwargs, updated_at_unix=time.time())
            write_json(root / "queue_status.json", state)

        progress()
        try:
            prior_hold = read_json(root / "llama70_priority_hold.json") if (root / "llama70_priority_hold.json").exists() else None
            hold = pause_waiter(root, prior_hold)
            progress("waiting_for_ckd", llama70_held_pid=hold["pid"], llama70_priority="after_attention")
            while not ckd_complete(read_json(CKD)):
                progress("waiting_for_ckd", ckd_status=read_json(CKD).get("status"))
                time.sleep(15)
            with GPU_LOCK.open("a+") as gpu:
                while True:
                    try:
                        fcntl.flock(gpu.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        progress("waiting_for_gpu_lock")
                        time.sleep(15)
                processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
                if processes:
                    raise RuntimeError("unmanaged GPU process is active; priority analysis will not overlap it")
                for job in state["jobs"]:
                    if job["status"] == "completed":
                        continue
                    job["status"] = "running"
                    command = [sys.executable, "-B", "-u", "-m", "src.analysis.kip_checkpoint_attention",
                               "--run-root", str(root), "--job", job["key"]]
                    log_dir = root / "logs"
                    log_dir.mkdir(exist_ok=True)
                    # Both locks survive if the supervisor exits while its worker is alive.
                    with (log_dir / (job["key"] + ".log")).open("a") as log:
                        child = subprocess.Popen(command, cwd=ROOT, env=environment(root),
                                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                                 pass_fds=(local.fileno(), gpu.fileno()))
                        progress("running", current_job=job["key"], child_pid=child.pid)
                        while child.poll() is None:
                            time.sleep(15)
                            progress()
                        job["exit_code"] = child.returncode
                    metrics = root / "jobs" / job["key"] / "metrics.json"
                    success = child.returncode == 0 and metrics.exists() and read_json(metrics).get("status") == "completed"
                    job["status"] = "completed" if success else "failed"
                    progress(child_pid=None)
                if any(j["status"] != "completed" for j in state["jobs"]):
                    progress("blocked_analysis_failures", current_job=None,
                             error="Some analyses failed; Llama70 remains held to preserve Qwen weights for repair")
                    return 1
                progress("analysis_completed", current_job=None, child_pid=None)
            resume_waiter(hold, root)
            progress("completed", llama70_priority="resumed_after_attention")
            return 0
        except BaseException as error:
            progress("failed", error=f"{type(error).__name__}: {error}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--perturbations", type=int, default=200)
    args = parser.parse_args()
    parser.error("This historical supervisor controls other queues and is disabled in the public release. "
                 "Use python -m src.analysis.kip_checkpoint_attention --help for direct audit entry points.")
    args.run_root = args.run_root.resolve()
    if args.prepare:
        plan = build_plan(args.run_root, args.samples, args.perturbations)
        print(json.dumps(dict(jobs=len(plan["jobs"]), plan_sha256=plan["plan_sha256"], run_root=str(args.run_root))))
        return
    if args.background:
        args.run_root.mkdir(parents=True, exist_ok=True)
        # Read-only identity check before the detached process performs the priority mutation.
        state = read_json(LLAMA)
        validate_waiting_llama(state, process_identity(state["supervisor_pid"]))
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--run-root", str(args.run_root)]
        with (args.run_root / "supervisor.log").open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=environment(args.run_root), start_new_session=True,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        (args.run_root / "supervisor.pid").write_text(str(child.pid) + "\n")
        for _ in range(100):
            time.sleep(.1)
            if child.poll() is not None:
                raise RuntimeError("priority queue exited; inspect supervisor.log")
            status = args.run_root / "queue_status.json"
            if status.exists():
                info = read_json(status)
                if info.get("supervisor_pid") == child.pid and info.get("status") == "waiting_for_ckd":
                    print(json.dumps(info))
                    return
        raise RuntimeError("priority startup not yet acknowledged; inspect queue_status.json before relaunching")
    raise SystemExit(run_queue(args))


if __name__ == "__main__":
    main()
