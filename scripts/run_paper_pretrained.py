#!/usr/bin/env python3
"""Run one Appendix B pretrained fit from an explicit local snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_protocol import CONFIG, MODELS, SEEDS, TRAIN_SIZES, model_spec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--tuning", choices=("lora", "qlora", "full"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="CSV suffix (OC/GPC/KIP)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--train-size", type=int, choices=TRAIN_SIZES, default=400000)
    parser.add_argument("--microbatch", type=int, default=CONFIG.effective_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=CONFIG.effective_batch_size)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--smoke", action="store_true", help="run a small smoke check")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--dry-run", action="store_true", help="print protocol (the default)")
    execution.add_argument("--execute", action="store_true", help="start this fit")
    args = parser.parse_args(argv)
    if args.microbatch < 1 or CONFIG.effective_batch_size % args.microbatch:
        parser.error(f"microbatch must divide effective batch size {CONFIG.effective_batch_size}")
    if args.eval_batch_size < 1:
        parser.error("evaluation batch size must be positive")
    try:
        model_spec(args.model, args.tuning)
    except ValueError as error:
        parser.error(str(error))
    if args.device == "cpu" and not args.smoke:
        parser.error("CPU is supported only for --smoke; full runs use BF16 CUDA")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({**CONFIG.to_dict(), "model": model_spec(args.model, args.tuning),
                          "run": {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()}}, indent=2))
        return
    from src.experiments.paper_pretrained_training import train_paper
    train_paper(args)


if __name__ == "__main__":
    main()
