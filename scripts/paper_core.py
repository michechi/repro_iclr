#!/usr/bin/env python3
"""Experiment schedule for Section 4.3, Appendix B, and Appendix D.1-D.4 baselines."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_catalog import DATA_DIRS
from src.experiments.paper_protocol import SEEDS, TRAIN_SIZES
from src.experiments.paper_data import oc_prevalence, oracle_metrics
from src.experiments.paper_baselines import FEATURES


def make_plan(args):
    jobs = []

    def add(section, tag, model, size, seed, extra=(), data_dir=None, feature=None, k=None, shuffle_control=None):
        control = shuffle_control or "none"
        suffix = f"{model}/{feature}" if feature else f"{model}/k{k}" if k else model
        if control != "none":
            suffix = f"{suffix}/{control}"
        identity = f"{section}/{tag}/{suffix}/n{size}/seed{seed}"
        command = ["python", "-B", "scripts/run_paper_baselines.py", "--model", model,
                   "--data-dir", str(data_dir or DATA_DIRS[tag]), "--tag", tag,
                   "--train-size", str(size), "--seed", str(seed),
                   "--output-dir", str(args.output_root / identity)] + list(extra)
        if feature:
            command += ["--feature", feature]
        if k:
            command += ["--k", str(k)]
        if control != "none":
            command += ["--shuffle-control", control]
        jobs.append({"id": identity, "section": section, "tag": tag, "model": model,
                     "seed": seed, "train_rows": size, "val_rows": 50000, "test_rows": 50000,
                     "feature": feature, "k": k, "shuffle_control": control, "status": section,
                     "inspect_command": command + ["--dry-run"],
                     "execute_command": command + ["--execute"]})
        return jobs[-1]

    if args.scope in ("all", "main"):
        for tag in DATA_DIRS:
            for seed in SEEDS:
                for size in TRAIN_SIZES:
                    for model in ("lstm", "transformer", "rnntransformer", "llama1m", "xgboost-basic", "xgboost-llm"):
                        extra = ["--variant", "D" if model == "llama1m" else args.variant]
                        if model == "llama1m" and args.llama1m_learning_rate:
                            extra += ["--learning-rate", str(args.llama1m_learning_rate)]
                        if model == "xgboost-llm":
                            extra += ["--model-path", str(args.llama1b_snapshot or "<LOCAL_LLAMA1B_SNAPSHOT>")]
                        add("main", tag, model, size, seed, extra)
                    for k in ((1, 2) if tag == "test_just_pair" else (2,)):
                        add("main", tag, "kgram", size, seed, k=k)

    if args.scope in ("all", "sensitivity"):
        for m in (3, 6, 10):
            for lag in (1, 3, 5, 7, 9, 10):
                tag = f"oc_m{m}_lag{lag}"
                for seed in SEEDS:
                    for model in ("lstm", "transformer", "bert", "llama1b"):
                        if model in ("lstm", "transformer"):
                            job = add("D1", tag, model, 400000, seed, ["--sensitivity"],
                                      data_dir=args.sensitivity_data / tag)
                        else:
                            identity = f"D1/{tag}/{model}/n400000/seed{seed}"
                            snapshot = args.bert_snapshot if model == "bert" else args.llama1b_snapshot
                            command = ["python", "-B", "scripts/run_paper_pretrained.py", "--model", model,
                                       "--tuning", "lora", "--model-path", str(snapshot or f"<LOCAL_{model.upper()}_SNAPSHOT>"),
                                       "--data-dir", str(args.sensitivity_data / tag), "--tag", tag,
                                       "--seed", str(seed), "--train-size", "400000", "--output-dir", str(args.output_root / identity)]
                            job = {"id": identity, "section": "D1", "tag": tag, "model": model,
                                   "seed": seed, "train_rows": 400000, "val_rows": 50000, "test_rows": 50000,
                                   "status": "D1", "inspect_command": command + ["--dry-run"],
                                   "execute_command": command + ["--execute"]}
                            jobs.append(job)
                        job["oracle_auc"] = oracle_metrics(oc_prevalence(20, m, lag), .3)["auc"]

    if args.scope in ("all", "ladder"):
        ladder_features = ("count26", "count_keys", "residue", "lag_pairs", "key_lag_pairs", "position_pairs", "key_trigrams")
        for tag in ("ocdet", "ocnoise"):
            for feature in ladder_features:
                for model in ("logreg-l2", "xgboost-feature"):
                    add("D2", tag, model, 200000, SEEDS[0], feature=feature)
        for tag in ("kip_m4", "kip_m6"):
            add("D2", tag, "mlp-feature", 400000, SEEDS[0], feature="key_order_pairs")

    if args.scope in ("all", "heldout"):
        # Published Appendix D.4 AUC=1.00/0.50 requires pi=0. Optional noisy check uses pi=0.3
        # and should be compared to the Theorem A.1 oracle, not to 1.00.
        for tag, pi, published in (("ocdet", 0.0, True), ("ocnoise", 0.3, False)):
            identity = f"D4/{tag}/heldout-rule"
            command = ["python", "-B", "scripts/run_paper_heldout_rule.py",
                       "--n_rules", "6", "--n_per_rule", "500000", "--n", "20", "--m", "6",
                       "--noise-probability", str(pi), "--families", "A1_count26", "C1_lagpair",
                       "--out", str(args.output_root / "D4" / tag / "heldout_rule.csv")]
            jobs.append({"id": identity, "section": "D4", "tag": tag, "model": "logreg-l2",
                         "status": "D4", "n_rules": 6, "n_keys": 6, "sequence_length": 20,
                         "train_rows": 400000, "val_rows": 50000, "test_rows": 50000,
                         "noise_probability": pi, "published_table_numbers": published,
                         "within_rule_ceiling": 1.0 if pi == 0.0 else "theorem_A1_oracle",
                         "lag_domain": list(range(1, 20)),
                         "rule_parameters": ["S", "kappa", "lambda"],
                         "features": ["letter_counts", "lambda_aware_pair_counts"],
                         "evaluation": ["within_rule", "across_rule"],
                         "inspect_command": command, "execute_command": command + ["--execute"]})

    if args.scope in ("all", "all-lag"):
        train_size = args.all_lag_train_size or 200000
        for tag in ("ocdet", "ocnoise"):
            for model in ("logreg-l1", "xgboost-feature"):
                add("D3", tag, model, train_size, SEEDS[0], feature="all_lag_pairs")
        # Section 5 / Appendix D.3: λ-agnostic all-lag pair counts on KIP-4 (AUC 1.000 in the paper).
        for model in ("logreg-l1", "xgboost-feature"):
            add("D3", "kip_m4", model, 400000, SEEDS[0], feature="all_lag_pairs")

    if args.scope in ("all", "main", "shuffle-controls"):
        for seed in SEEDS:
            for control in ("train_shuffled", "test_shuffled"):
                # Keep the same Table-2 variant as the ordered LSTM run (D or O).
                add("shuffle", "kip_m4", "lstm", 400000, seed,
                    extra=["--variant", args.variant], shuffle_control=control)

    return {"schema_version": 1,
            "scope": args.scope, "job_count": len(jobs), "status": "paper_core_baselines_v1",
            "seeds": list(SEEDS),
            "jobs": jobs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", choices=["all", "main", "sensitivity", "ladder", "all-lag", "heldout", "shuffle-controls"], default="all")
    p.add_argument("--output-root", type=Path, default=Path("results/paper_core_v1"))
    p.add_argument("--sensitivity-data", type=Path, default=Path("data/paper_sensitivity_v1"))
    p.add_argument("--variant", choices=["D", "O"], default="D")
    p.add_argument("--llama1m-learning-rate", type=float)
    p.add_argument("--llama1b-snapshot", type=Path)
    p.add_argument("--bert-snapshot", type=Path)
    p.add_argument("--all-lag-train-size", type=int, choices=list(TRAIN_SIZES) + [100000])
    args = p.parse_args(argv)
    if args.llama1m_learning_rate is not None and args.llama1m_learning_rate <= 0:
        p.error("positive learning rate required")
    print(json.dumps(make_plan(args), indent=2))


if __name__ == "__main__":
    main()
