"""Launch pinned Qwen3.8-27B with the previous Qwen test rows and demonstrations."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL = Path("data/model_cache/models--Qwen--Qwen3.8-27B/snapshots") / REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "results/qwen38_27b_fewshot_20260918")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    root = args.output_root.resolve()
    stage = "pilot" if args.pilot else "run"
    reference = REPO / "results/qwen25_14b_fewshot_20260918"
    data = reference / "data/oc_corrected"
    hashes = root / "evidence/model_weight_hashes.json"
    if not args.dry_run:
        if not (MODEL / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("download the pinned Qwen snapshot into host RAM cache first")
        if not hashes.is_file():
            raise FileNotFoundError("complete the checkpoint SHA256 audit first")
    if not (data / "export_manifest.json").is_file():
        raise FileNotFoundError("verified corrected OC exports are required")
    command = [sys.executable, "-B", "-u", "-m", "src.experiments.qwen38_fewshot_eval",
               "--model-path", str(MODEL), "--oc-data-dir", str(data),
               "--oc-det-tag", "ocdet", "--oc-noise-tag", "ocnoise",
               "--tasks", "oc_det,oc_noise,gpc,kip4,kip6", "--shots", "0,5,20",
               "--demo-seeds", "0" if args.pilot else "0,1,2",
               "--test-seed", "9950", "--n-test", "32" if args.pilot else "1000",
               "--batch-size", str(args.batch_size), "--max-length", "4096",
               "--reference-run", str(reference / stage), "--weight-hash-manifest", str(hashes),
               "--output-dir", str(root / stage)]
    if args.dry_run:
        command.append("--dry-run")
    logs, protocol = root / "logs", root / "protocol"
    logs.mkdir(parents=True, exist_ok=True)
    protocol.mkdir(parents=True, exist_ok=True)
    sources = {}
    if not args.dry_run:
        for name in ("qwen38_fewshot_eval.py", "qwen_fewshot_eval.py"):
            source = (REPO / "src/experiments" / name).read_bytes()
            frozen = protocol / f"{stage}_{name}"
            if frozen.exists() and frozen.read_bytes() != source:
                raise ValueError(f"{name} differs from frozen stage source; use a new output root")
            frozen.write_bytes(source)
            sources[name] = hashlib.sha256(source).hexdigest()
    lock_path = root / stage / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"{stage} already has a running evaluator") from error
    environment = os.environ.copy()
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_DEBUG="0", HF_HUB_DISABLE_PROGRESS_BARS="1",
                       TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8",
                       MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                       NUMEXPR_NUM_THREADS="8", PYTHONUNBUFFERED="1")
    if not args.dry_run:
        environment["TRITON_CACHE_DIR"] = "runs/cache/triton"
    suffix = "_dry" if args.dry_run else ""
    log_path = logs / f"{stage}{suffix}.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=args.background)
    record = {"stage": stage, "pid": process.pid, "command": command,
              "started_at_unix": time.time(), "background": args.background,
              "log": str(log_path), "script_sha256": sources,
              "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_revision": REVISION, "status": "launched",
              "thinking_enabled": False, "ram_cache_volatile": True}
    path = root / f"{stage}{suffix}_launcher.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    if not args.background:
        code = process.wait()
        record.update(status="completed" if code == 0 else "failed",
                      exit_code=code, ended_at_unix=time.time())
        path.write_text(json.dumps(record, indent=2) + "\n")
        raise SystemExit(code)


if __name__ == "__main__":
    main()
