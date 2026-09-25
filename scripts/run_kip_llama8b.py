#!/usr/bin/env python3
"""Prepare or launch full FP32 Llama 3.1 8B training on ordered KIP M6."""
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
from src.experiments.kip_encoder_training import atomic_json

MODEL = "meta-llama/Llama-3.1-8B"
REVISION = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
SOURCE_BYTES = 16_060_556_376
# Conservative budget including the selected FP32 classifier and export overhead.
EXPORT_BYTES = 33_000_000_000
MARGIN_BYTES = 2 * 1024**3
DEFAULT_ROOT = ROOT / "results/kip_llama8b_full_m6_20260917_seed9950"
CACHE = ROOT / "data/model_cache"
DEFAULT_SOURCE = CACHE / "models--meta-llama--Llama-3.1-8B/snapshots" / REVISION


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.download and args.model_path != DEFAULT_SOURCE:
        parser.error("--download uses the default cache; omit it for a supplied local model")
    args.run_root = args.run_root.resolve()
    args.model_path = args.model_path.resolve()
    if args.smoke:
        args.run_root = args.run_root / "smoke"
    return args


def source_ready(path):
    config_path, index_path = path / "config.json", path / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file() or not (path / "tokenizer.json").is_file():
        return False
    config = json.loads(config_path.read_text())
    if (config.get("model_type"), config.get("hidden_size"), config.get("num_hidden_layers")) != ("llama", 4096, 32):
        raise ValueError("expected a base Llama 3.1 8B snapshot (Llama, hidden size 4096, 32 layers)")
    shards = set(json.loads(index_path.read_text())["weight_map"].values())
    return bool(shards) and all((path / shard).is_file() for shard in shards)


def training_command(args):
    cmd = [sys.executable, "-B", "-u", "-m", "src.experiments.kip_llama_training",
           "--model-name", MODEL, "--model-path", str(args.model_path),
           "--tag", "kip_m6", "--mode", "ordered", "--seed", "9950",
           "--ordered-dir", str(ROOT / "data/simulation/kip_ordered_recovered"),
           "--shuffled-dir", str(ROOT / "data/simulation/kip_shuffled"),
           "--output-dir", str(args.run_root / "run"),
           "--batch-size", "16", "--gradient-accumulation-steps", "16",
           "--eval-batch-size", "16", "--threads", "8", "--progress-steps", "16",
           "--optimizer-offload", "cpu", "--gradient-checkpointing",
           "--precision", "bf16", "--lr", "2e-5", "--weight-decay", "0.01",
           "--warmup-ratio", "0.06", "--epochs", "1" if args.smoke else "20",
           "--patience", "3", "--max-length", "64"]
    if args.smoke:
        cmd.extend(["--smoke", "--no-save-model"])
    return cmd


def preparation(args):
    args.run_root.mkdir(parents=True, exist_ok=True)
    ready = source_ready(args.model_path)
    free = shutil.disk_usage(args.run_root).free
    needed = (0 if args.smoke else EXPORT_BYTES) + MARGIN_BYTES
    # With the default download cache, source and run reside on the same volume.
    if not ready and args.model_path == DEFAULT_SOURCE:
        needed += SOURCE_BYTES
    blockers = []
    if not ready:
        blockers.append("8B source weights are missing; provide --model-path or use --download with approved Hugging Face access")
    if free < needed:
        blockers.append(f"insufficient output storage: {free} free bytes, at least {needed} required")
    plan = {"model": MODEL, "revision": REVISION if args.model_path == DEFAULT_SOURCE else None,
            "source": str(args.model_path), "source_ready": ready,
            "command": training_command(args), "status": "blocked" if blockers else "ready",
            "blockers": blockers, "free_bytes": free, "required_free_bytes": needed,
            "effective_batch_size": 256, "parameter_dtype": "float32",
            "optimizer_state_dtype": "float32", "optimizer_device": "cpu",
            "compute_precision": "bf16", "gradient_checkpointing": True,
            "updated_at_unix": time.time()}
    atomic_json(plan, args.run_root / "launch_plan.json")
    return plan


def download(args):
    from huggingface_hub import snapshot_download

    required = SOURCE_BYTES + (0 if args.smoke else EXPORT_BYTES) + MARGIN_BYTES
    CACHE.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(CACHE).free < required:
        raise RuntimeError(f"download and export require at least {required} free bytes on {CACHE}")
    snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE,
                      allow_patterns=["*.json", "*.safetensors", "LICENSE", "USE_POLICY.md"],
                      max_workers=2)


def supervise(args, plan):
    with (args.run_root / "launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.run_root / "run").exists():
            raise FileExistsError("run output already exists; use a new --run-root")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; launch outside the restricted sandbox")
        if torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
            raise RuntimeError("this configuration requires an 80GB GPU")
        protocol = args.run_root / "protocol"
        protocol.mkdir(exist_ok=True)
        sources = [Path(__file__), ROOT / "src/experiments/kip_llama_training.py",
                   ROOT / "src/experiments/kip_encoder_training.py",
                   ROOT / "src/experiments/kip_cpu_offload.py"]
        plan["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources}
        for source in sources:
            shutil.copyfile(source, protocol / source.name)
        state = {**plan, "status": "starting", "supervisor_pid": os.getpid(),
                 "started_at_unix": time.time(), "gpu": torch.cuda.get_device_name(0)}
        status_path = args.run_root / "launcher_status.json"
        atomic_json(state, status_path)
        try:
            with (args.run_root / "train.log").open("a") as log:
                child = subprocess.Popen(plan["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state.update(status="running", child_pid=child.pid)
                atomic_json(state, status_path)
                code = child.wait()
            if code or not (args.run_root / "run/metrics.json").is_file():
                raise RuntimeError(f"training did not complete (exit={code}); inspect train.log")
            state["status"] = "completed"
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            raise
        finally:
            state["finished_at_unix"] = time.time()
            atomic_json(state, status_path)


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="9950",
                      OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
                      TOKENIZERS_PARALLELISM="false")
    plan = preparation(args)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    if args.download and not plan["source_ready"]:
        download(args)
        plan = preparation(args)
    if plan["blockers"]:
        raise RuntimeError("; ".join(plan["blockers"]))
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.background:
        # args.run_root already includes the smoke suffix, so pass its parent.
        launch_root = args.run_root.parent if args.smoke else args.run_root
        cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--run-root", str(launch_root), "--model-path", str(args.model_path)]
        if args.smoke:
            cmd.append("--smoke")
        with (args.run_root / "supervisor.log").open("a") as log:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        (args.run_root / "supervisor.pid").write_text(str(child.pid) + "\n")
        print(json.dumps({"supervisor_pid": child.pid, "run_root": str(args.run_root)}))
    else:
        supervise(args, plan)


if __name__ == "__main__":
    main()
