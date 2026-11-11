#!/usr/bin/env python3
"""Appendix B scratch, tree, k-gram, and Appendix D feature baselines.

Printing the configuration is the default. --execute fits an estimator.
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_baselines import SCRATCH_CONFIGS, FEATURES, configuration, run_baseline


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=list(SCRATCH_CONFIGS) + ["xgboost-basic", "xgboost-llm", "xgboost-feature", "mlp-feature", "logreg-l1", "logreg-l2", "kgram"], required=True)
    p.add_argument("--variant", choices=["D", "O"], default="D")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=9550)
    p.add_argument("--train-size", type=int, choices=[4000, 40000, 100000, 120000, 200000, 300000, 400000], default=400000)
    p.add_argument("--feature", choices=FEATURES, default="count26")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--shuffle-control", choices=["none", "train_shuffled", "test_shuffled"], default="none",
                   help="Section 5 order controls: retrain on shuffled inputs or evaluate test-only shuffles, keeping original labels")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--sensitivity", action="store_true")
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true")
    execution.add_argument("--dry-run", action="store_true", help="print the configuration (the default)")
    args = p.parse_args(argv)
    if args.threads < 1 or not 1 <= args.k <= 20:
        p.error("positive threads and 1<=k<=20 required")
    if args.model == "xgboost-llm" and args.execute and not args.model_path:
        p.error("--model-path required for frozen Llama embeddings")
    if args.learning_rate is not None and args.learning_rate <= 0:
        p.error("positive learning rate required")
    try:
        configuration(args.model, args.variant, learning_rate=args.learning_rate, sensitivity=args.sensitivity)
    except (ValueError, KeyError) as e:
        p.error(str(e))
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.execute:
        print(json.dumps(run_baseline(args), indent=2))
    else:
        print(json.dumps({"protocol": "paper_core_baselines_v1",
                          "configuration": configuration(args.model, args.variant, learning_rate=args.learning_rate, sensitivity=args.sensitivity),
                          "run": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, indent=2))


if __name__ == "__main__":
    main()
