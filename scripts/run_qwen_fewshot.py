"""Launch the requested fixed Qwen few-shot grid with persistent logs."""
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
REVISION = "97e1e76335b7017d8f67c08a19d103c0504298c9"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "results/qwen25_14b_fewshot_20260918")
    args = parser.parse_args()
    root = args.output_root.resolve()
    stage = "pilot" if args.pilot else "run"
    model = REPO / "data/model_cache/models--Qwen--Qwen2.5-14B/snapshots" / REVISION
    data = root / "data/oc_corrected"
    if not (model / "model.safetensors.index.json").exists():
        raise FileNotFoundError("download the pinned Qwen snapshot first")
    if not (data / "export_manifest.json").exists():
        raise FileNotFoundError("prepare and verify corrected OC exports first")
    command = [sys.executable, "-B", "-u", "-m", "src.experiments.qwen_fewshot_eval",
               "--model-path", str(model), "--oc-data-dir", str(data),
               "--oc-det-tag", "ocdet", "--oc-noise-tag", "ocnoise",
               "--tasks", "oc_det,oc_noise,gpc,kip4,kip6", "--shots", "0,5,20",
               "--demo-seeds", "0" if args.pilot else "0,1,2",
               "--test-seed", "9950", "--n-test", "32" if args.pilot else "1000",
               "--batch-size", str(args.batch_size), "--max-length", "4096",
               "--output-dir", str(root / stage)]
    logs, protocol = root / "logs", root / "protocol"
    logs.mkdir(parents=True, exist_ok=True)
    protocol.mkdir(parents=True, exist_ok=True)
    source = (REPO / "src/experiments/qwen_fewshot_eval.py").read_bytes()
    frozen = protocol / f"{stage}_qwen_fewshot_eval.py"
    if frozen.exists() and frozen.read_bytes() != source:
        raise ValueError("evaluator differs from the frozen stage source; use a new output root")
    frozen.write_bytes(source)
    # Do not enqueue a second process behind an already running evaluator.
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
    log_path = logs / f"{stage}.log"
    with log_path.open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=args.background)
    record = {"stage": stage, "pid": process.pid, "command": command,
              "started_at_unix": time.time(), "background": args.background,
              "log": str(log_path), "script_sha256": hashlib.sha256(source).hexdigest(),
              "model_revision": REVISION, "status": "launched"}
    path = root / f"{stage}_launcher.json"
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
