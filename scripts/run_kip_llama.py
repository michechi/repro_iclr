#!/usr/bin/env python3
"""Launch four Llama 3.2 1B full KIP runs, seed 9950, using local weights."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_kip_encoders import summarize, write_json

DEFAULT_RUN_ROOT = ROOT / "results/kip_llama1b_full_20260916_seed9950"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
MODEL_PATH = ROOT / Path("data/model_cache/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08")
WEIGHT_BYTES = 1_235_818_496 * 4


def matrix(smoke=False):
    return [dict(tuning="full", model="llama1b", tag=tag, mode=mode, seed=9950,
                 name=f"full/llama1b/{tag}/{mode}/seed9950")
            for tag in (("kip_m4",) if smoke else ("kip_m4", "kip_m6"))
            for mode in (("ordered",) if smoke else ("ordered", "shuffled_train"))]


def command(job, args):
    cmd = [sys.executable, "-B", "-u", "-m", "src.experiments.kip_llama_training",
           "--model-name", MODEL_NAME, "--model-path", str(MODEL_PATH),
           "--tag", job["tag"], "--mode", job["mode"], "--seed", "9950",
           "--ordered-dir", str(ROOT / "data/simulation/kip_ordered_recovered"),
           "--shuffled-dir", str(ROOT / "data/simulation/kip_shuffled"),
           "--output-dir", str(args.run_root / "runs" / job["name"]),
           "--cache-dir", str(ROOT / "data/model_cache"),
           "--batch-size", str(args.batch_size),
           "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
           "--eval-batch-size", "256", "--threads", "4",
           "--lr", "2e-5", "--epochs", "1" if args.smoke else "20",
           "--patience", "3", "--max-length", "64", "--skip-completed"]
    if args.smoke:
        cmd.extend(["--smoke", "--no-save-model"])
    return cmd


def supervise(args):
    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = matrix(args.smoke)
        commands = [command(job, args) for job in jobs]
        sources = [ROOT / "src/experiments/kip_llama_training.py",
                   ROOT / "src/experiments/kip_encoder_training.py",
                   ROOT / "scripts/run_kip_encoders.py", Path(__file__).resolve()]
        manifest = {
            "seed": 9950, "smoke": args.smoke, "jobs": jobs, "commands": commands,
            "model_name": MODEL_NAME, "model_path": str(MODEL_PATH),
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
            "objective": "binary outcome classification only",
            "initialization": "fresh pretrained checkpoint for each run",
            "checkpoint_policy": "selected FP32 model only; unfinished runs must restart",
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sources},
        }
        manifest_path = args.run_root / "matrix.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing matrix differs; use a new run root")
        else:
            if not args.smoke and shutil.disk_usage(args.run_root).free < len(jobs) * WEIGHT_BYTES + 1024**3:
                raise RuntimeError("Insufficient disk for four selected FP32 Llama checkpoints")
            write_json(manifest_path, manifest)
            protocol = args.run_root / "protocol"
            protocol.mkdir(exist_ok=True)
            for path in sources:
                shutil.copyfile(path, protocol / path.name)
        state = dict(status="running", pid=os.getpid(), started_unix=time.time(), total=len(jobs), completed=0)
        state_path = args.run_root / "matrix_status.json"
        (args.run_root / "logs").mkdir(exist_ok=True)
        (args.run_root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        write_json(state_path, state)
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required")
            print(f"GPU: {torch.cuda.get_device_name(0)}; {len(jobs)} Llama FULL runs", flush=True)
            for index, (job, cmd) in enumerate(zip(jobs, commands, strict=True)):
                state.update(current=job["name"], current_index=index + 1,
                             completed=summarize(args.run_root, jobs))
                write_json(state_path, state)
                logfile = args.run_root / "logs" / (job["name"].replace("/", "_") + ".log")
                print(f"[{index+1}/{len(jobs)}] {job['name']} -> {logfile}", flush=True)
                with logfile.open("a", buffering=1) as handle:
                    process = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                    state.update(child_pid=process.pid, log=str(logfile))
                    write_json(state_path, state)
                    returncode = process.wait()
                if returncode:
                    raise RuntimeError(f"{job['name']} failed with exit {returncode}; see {logfile}")
                completed = summarize(args.run_root, jobs)
                if completed < index + 1:
                    raise RuntimeError(f"{job['name']} returned without completed metrics")
                state.update(completed=completed)
                write_json(state_path, state)
            state.update(status="completed", finished_unix=time.time())
        except BaseException as exc:
            state.update(status="failed", error=str(exc), finished_unix=time.time())
            raise
        finally:
            write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("batch size and accumulation must be positive")
    args.run_root = args.run_root.resolve()
    if args.smoke and args.run_root == DEFAULT_RUN_ROOT:
        args.run_root = DEFAULT_RUN_ROOT / "smoke"
    if args.status:
        path = args.run_root / "matrix_status.json"
        state = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
        print(json.dumps(state, indent=2))
        if state.get("current"):
            progress = args.run_root / "runs" / state["current"] / "status.json"
            if progress.exists():
                print(progress.read_text())
        return
    os.environ.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="9950",
                      OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      NUMEXPR_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    if args.background:
        args.run_root.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--run-root", str(args.run_root), "--batch-size", str(args.batch_size),
               "--gradient-accumulation-steps", str(args.gradient_accumulation_steps)]
        if args.smoke:
            cmd.append("--smoke")
        with (args.run_root / "supervisor.log").open("a") as handle:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "run_root": str(args.run_root)}))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
