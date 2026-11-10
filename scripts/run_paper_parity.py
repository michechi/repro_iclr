#!/usr/bin/env python3
"""Appendix E.1-E.5 parity experiments. --execute runs one job."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_parity_protocol import build_plan, summarize_run_root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("all", "E1", "E2", "E3", "E4", "E5"), default="all")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/paper_parity")
    parser.add_argument("--job", help="Exact job ID from plan")
    parser.add_argument("--bert-path", type=Path, help="Local BERT-base snapshot for BERT jobs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and (not args.job or args.dry_run or args.summarize):
        parser.error("--execute requires --job and cannot be combined with --dry-run")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args.section)
    if args.job:
        selected = [job for job in plan["jobs"] if job["id"] == args.job]
        if len(selected) != 1:
            raise ValueError("job not in selected section plan")
    if args.output_plan:
        args.output_plan.parent.mkdir(parents=True, exist_ok=True)
        args.output_plan.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(summarize_run_root(plan, args.run_root) if args.summarize else plan, indent=2))
        return
    from src.experiments.paper_parity_training import run_job
    run_job(selected[0], args.run_root, bert_path=args.bert_path, device=args.device)


if __name__ == "__main__":
    main()
