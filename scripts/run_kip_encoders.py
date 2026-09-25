#!/usr/bin/env python3
"""Run the seed-9950 KIP encoder matrix, all LoRA jobs before full tuning.

Run --background to detach, --status to inspect, or --smoke for four small
end-to-end checks. Restarting the same matrix resumes its unfinished run.
"""
from __future__ import annotations

import argparse
import csv
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
DEFAULT_RUN_ROOT = ROOT / "results/kip_encoders_20260916_seed9950"
MODELS = {
    "bert": {
        "name": "bert-base-uncased",
        "path": str(ROOT / "data/model_cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"),
    },
    "roberta": {
        "name": "FacebookAI/roberta-base",
        "path": str(ROOT / "data/model_cache/models--FacebookAI--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"),
    },
}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def matrix(smoke: bool = False) -> list[dict]:
    return [
        dict(tuning=tuning, model=model, tag=tag, mode=mode, seed=9950,
             name=f"{tuning}/{model}/{tag}/{mode}/seed9950")
        for tuning in ("lora", "full")
        for model in MODELS
        for tag in (("kip_m4",) if smoke else ("kip_m4", "kip_m6"))
        for mode in (("ordered",) if smoke else ("ordered", "shuffled_train"))
    ]


def command(job: dict, args: argparse.Namespace) -> list[str]:
    model = MODELS[job["model"]]
    cmd = [
        sys.executable, "-B", "-u", "-m", "src.experiments.kip_encoder_training",
        "--model-name", model["name"], "--model-path", model["path"],
        "--tag", job["tag"], "--mode", job["mode"], "--tuning", job["tuning"],
        "--seed", str(job["seed"]),
        "--ordered-dir", str(ROOT / "data/simulation/kip_ordered_recovered"),
        "--shuffled-dir", str(ROOT / "data/simulation/kip_shuffled"),
        "--cache-dir", str(ROOT / "data/model_cache"),
        "--output-dir", str(args.run_root / "runs" / job["name"]),
        "--batch-size", str(args.batch_size), "--lr", "2e-5",
        "--eval-batch-size", "512", "--threads", "4",
        "--epochs", "1" if args.smoke else "20", "--patience", "3",
        "--max-length", "64", "--resume",
    ]
    if args.smoke:
        cmd.append("--smoke")
    return cmd


def summarize(run_root: Path, jobs: list[dict]) -> int:
    rows = []
    for job in jobs:
        metrics_path = run_root / "runs" / job["name"] / "metrics.json"
        status_path = metrics_path.with_name("status.json")
        if not metrics_path.exists() or not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed":
            continue
        metrics = json.loads(metrics_path.read_text())
        row = {**job}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if not isinstance(subvalue, (dict, list)):
                        row[f"{key}_{subkey}"] = subvalue
            elif not isinstance(value, list):
                row[key] = value
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        temporary = run_root / "summary.csv.tmp"
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(run_root / "summary.csv")
    return len(rows)


def supervise(args: argparse.Namespace) -> None:
    args.run_root.mkdir(parents=True, exist_ok=True)
    with (args.run_root / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = matrix(args.smoke)
        commands = [command(job, args) for job in jobs]
        manifest = {
            "seed": 9950, "smoke": args.smoke, "batch_size": args.batch_size,
            "stage_order": ["lora", "full"],
            "full_initialization": "fresh pretrained weights, independent of LoRA",
            "models": MODELS, "jobs": jobs, "commands": commands,
        }
        protocol_sources = [ROOT / "src/experiments/kip_encoder_training.py",
                            ROOT / "scripts/recover_kip_ordered.py", Path(__file__).resolve()]
        manifest["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in protocol_sources}
        manifest_path = args.run_root / "matrix.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Existing matrix configuration differs; use a new run root")
        else:
            write_json(manifest_path, manifest)
            protocol_dir = args.run_root / "protocol"
            protocol_dir.mkdir(exist_ok=True)
            for path in protocol_sources:
                shutil.copyfile(path, protocol_dir / path.name)
        state = dict(status="running", pid=os.getpid(), started_unix=time.time(),
                     total=len(jobs), completed=0)
        state_path = args.run_root / "matrix_status.json"
        (args.run_root / "logs").mkdir(exist_ok=True)
        (args.run_root / "supervisor.pid").write_text(str(os.getpid()) + "\n")
        write_json(state_path, state)
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this training matrix")
            print(f"GPU: {torch.cuda.get_device_name(0)}; {len(jobs)} runs; batch size {args.batch_size}", flush=True)
            for index, (job, cmd) in enumerate(zip(jobs, commands, strict=True)):
                # Enough room for an atomic full-model optimizer checkpoint.
                if shutil.disk_usage(args.run_root).free < 5 * 1024**3:
                    raise RuntimeError("Less than 5 GiB free for training checkpoints")
                state.update(current=job["name"], current_index=index + 1,
                             completed=summarize(args.run_root, jobs))
                write_json(state_path, state)
                logfile = args.run_root / "logs" / (job["name"].replace("/", "_") + ".log")
                print(f"[{index+1}/{len(jobs)}] {job['name']} -> {logfile}", flush=True)
                with logfile.open("a", buffering=1) as handle:
                    handle.write(f"\nSupervisor start: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
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
            state.update(status="completed", completed=len(jobs), finished_unix=time.time())
        except BaseException as exc:
            state.update(status="failed", error=str(exc), finished_unix=time.time())
            raise
        finally:
            write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.smoke and args.run_root == DEFAULT_RUN_ROOT:
        args.run_root = DEFAULT_RUN_ROOT / "smoke"
    if args.status:
        state_path = args.run_root / "matrix_status.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "not_started"}
        print(json.dumps(state, indent=2))
        if state.get("current"):
            progress = args.run_root / "runs" / state["current"] / "status.json"
            if progress.exists():
                print(progress.read_text())
        return
    os.environ.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                      PYTHONHASHSEED="9950", OMP_NUM_THREADS="4",
                      OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
                      NUMEXPR_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.background:
        args.run_root.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--run-root", str(args.run_root), "--batch-size", str(args.batch_size)]
        if args.smoke:
            cmd.append("--smoke")
        with (args.run_root / "supervisor.log").open("a") as handle:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        print(json.dumps({"pid": child.pid, "run_root": str(args.run_root)}))
    else:
        supervise(args)


if __name__ == "__main__":
    main()
