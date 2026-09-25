"""Launch and supervise the Qwen3.8 40-shot thinking experiment and its report."""
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
SOURCE_NAMES = ("qwen_fewshot_eval.py", "qwen38_fewshot_eval.py", "qwen_thinking_generation.py",
                "qwen_thinking_fewshot_eval.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--max-thinking-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--output-root", type=Path,
                        default=REPO / "results/qwen38_27b_fewshot40_thinking_20260918")
    args = parser.parse_args()
    if min(args.batch_size, args.score_batch_size) < 1 or args.max_thinking_tokens < 1 or (args.n_test is not None and not 2 <= args.n_test <= 1000):
        parser.error("positive batch size/thinking budget and 2..1000 test rows required")
    root = args.output_root.resolve()
    stage = "pilot" if args.pilot else "run"
    suffix = "_dry" if args.dry_run else ""
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "protocol").mkdir(parents=True, exist_ok=True)
    if args.background:
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                   *[v for v in sys.argv[1:] if v != "--background"]]
        with (root / "logs" / f"{stage}{suffix}.supervisor.log").open("a") as log:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"status": "supervisor_started", "pid": process.pid, "stage": stage,
                  "command": command, "started_at_unix": time.time()}
        (root / f"{stage}{suffix}_supervisor.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        return
    previous = REPO / "results/qwen38_27b_fewshot_20260918"
    hashes = previous / "evidence/model_weight_hashes.json"
    data = REPO / "results/qwen25_14b_fewshot_20260918/data/oc_corrected"
    if not args.dry_run and not (MODEL / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("verified pinned checkpoint must be present in the host RAM cache")
    sources = {}
    if not args.dry_run:
        for name in SOURCE_NAMES:
            source = (REPO / "src/experiments" / name).read_bytes()
            frozen = root / "protocol" / f"{stage}_{name}"
            if frozen.exists() and frozen.read_bytes() != source:
                raise ValueError(f"{name} differs from the frozen stage source; use a new output root")
            frozen.write_bytes(source)
            sources[name] = hashlib.sha256(source).hexdigest()
    command = [sys.executable, "-B", "-u", "-m", "src.experiments.qwen_thinking_fewshot_eval",
               "--model-path", str(MODEL), "--oc-data-dir", str(data),
               "--oc-det-tag", "ocdet", "--oc-noise-tag", "ocnoise",
               "--tasks", "kip6" if args.pilot else "oc_det,oc_noise,gpc,kip4,kip6",
               "--shots", "40", "--demo-seeds", "0" if args.pilot else "0,1,2",
               "--test-seed", "9950", "--generation-seed", "9951",
               "--n-test", str(args.n_test if args.n_test is not None else 32 if args.pilot else 1000),
               "--batch-size", str(args.batch_size), "--max-length", "8192",
               "--score-batch-size", str(args.score_batch_size),
               "--max-thinking-tokens", str(args.max_thinking_tokens),
               "--reasoning-effort", args.reasoning_effort,
               "--reference-run", str(previous / "run"), "--weight-hash-manifest", str(hashes),
               "--output-dir", str(root / stage)]
    if args.dry_run:
        command.append("--dry-run")
    lock_path = root / f".{stage}{suffix}_launcher.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = os.environ.copy()
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DEBUG="0",
            HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="8",
            MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", NUMEXPR_NUM_THREADS="8", PYTHONUNBUFFERED="1")
        if not args.dry_run:
            environment["TRITON_CACHE_DIR"] = "runs/cache/triton"
        log_path = root / "logs" / f"{stage}{suffix}.log"
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
        record = {"status": "launched", "pid": child.pid, "supervisor_pid": os.getpid(), "stage": stage,
            "command": command, "started_at_unix": time.time(), "log": str(log_path),
            "source_sha256": sources, "thinking_enabled": True, "reasoning_effort": args.reasoning_effort,
            "max_thinking_tokens": args.max_thinking_tokens, "model_revision": REVISION}
        record_path = root / f"{stage}{suffix}_launcher.json"
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)
        code = child.wait()
        record.update(status="completed" if code == 0 else "failed", exit_code=code, ended_at_unix=time.time())
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        if code == 0 and not args.dry_run:
            report_script = REPO / "scripts/summarize_qwen_thinking.py"
            if report_script.is_file():
                report = subprocess.run([sys.executable, "-B", str(report_script), "--run-dir", str(root/stage),
                                         "--output-dir", str(root/f"{stage}_analysis"), "--reference-run", str(previous/"run")],
                                        cwd=REPO, env=environment)
                record["report_exit_code"] = report.returncode
            else:
                record["report_status"] = "report script unavailable; evaluation artifacts completed"
            record_path.write_text(json.dumps(record, indent=2) + "\n")
        raise SystemExit(code)


if __name__ == "__main__":
    main()
