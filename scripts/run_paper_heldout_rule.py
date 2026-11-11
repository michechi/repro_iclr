#!/usr/bin/env python3
"""Appendix D.4 held-out-rule generalization.

Six OC datasets (n=20, 26 letters, m=6, 400K/50K/50K) draw random key sets S,
rankings kappa, and lags lambda in {1, ..., n-1} from the same configuration
family as the main OC experiments. The published Appendix D.4 numbers
(within-rule AUC 1.00 / across-rule AUC 0.50 for lambda-aware pairs) use
deterministic labels (pi=0): with pi=0.3 the within-rule ceiling is the
Theorem A.1 oracle (~0.67), not 1.00. Pass --noise-probability 0.3 only for
the optional noisy transfer check.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import os
import string
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("REPRO_ROOT", Path(__file__).resolve().parents[1]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.paper_data import oc_label, tokens

ALPHABET = list(string.ascii_uppercase)
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results/paper_heldout_rule"))
SEED_BASE = 20260401
PAPER_ROWS = {"train": 400_000, "val": 50_000, "test": 50_000}
FEATURE_NAMES = {
    "A1_count26": "letter_counts",
    "C1_lagpair": "lambda_aware_pair_counts",
    "B_residue": "residue_class_counts",
    "C2_lagpair_pos": "position_aware_lag_pairs",
}


def feat_count26(toks):
    import numpy as np
    c = Counter(toks)
    return np.array([c.get(letter, 0) for letter in ALPHABET], dtype=np.float32)


def feat_residue(toks, lag: int):
    import numpy as np
    feats = np.zeros((26, lag), dtype=np.float32)
    for i, token in enumerate(toks):
        if len(token) == 1 and "A" <= token <= "Z":
            feats[ord(token) - ord("A"), i % lag] += 1
    return feats.reshape(-1)


def feat_lag_pair(toks, lag: int):
    import numpy as np
    n = len(toks)
    feats = np.zeros((26, 26), dtype=np.float32)
    for t in range(n - lag):
        a, b = toks[t], toks[t + lag]
        if len(a) == 1 and len(b) == 1 and "A" <= a <= "Z" and "A" <= b <= "Z":
            feats[ord(a) - ord("A"), ord(b) - ord("A")] += 1
    return feats.reshape(-1)


def feat_lag_pair_position(toks, lag: int, n: int):
    import numpy as np
    seq_len = len(toks)
    nstart = n - lag
    feats = np.zeros((nstart, 26, 26), dtype=np.float32)
    for t in range(min(nstart, seq_len - lag)):
        a, b = toks[t], toks[t + lag]
        if len(a) == 1 and len(b) == 1 and "A" <= a <= "Z" and "A" <= b <= "Z":
            feats[t, ord(a) - ord("A"), ord(b) - ord("A")] = 1.0
    return feats.reshape(-1)


def sample_rule(seed: int, alphabet=ALPHABET, m: int = 6, lag_choices=tuple(range(1, 20))):
    import numpy as np
    rng = np.random.default_rng(seed)
    S = tuple(sorted(rng.choice(alphabet, size=m, replace=False).tolist()))
    kappa_perm = rng.permutation(m)
    kappa = {s: int(kappa_perm[i]) for i, s in enumerate(S)}
    lag = int(rng.choice(lag_choices))
    return {"S": S, "kappa": kappa, "lag": lag, "seed": seed}


def ordered_keys(rule: dict) -> list[str]:
    if not set(rule["S"]).issubset(ALPHABET) or set(rule["kappa"]) != set(rule["S"]):
        raise ValueError("S must belong to ALPHABET and kappa must rank exactly S")
    if len(set(rule["kappa"].values())) != len(rule["S"]):
        raise ValueError("kappa must be a bijective ranking")
    return [key for key, _ in sorted(rule["kappa"].items(), key=lambda item: item[1])]


def generate_rule_dataset(
    rule: dict, n_sequences: int, n: int = 20, seed: int = 0,
    noise_probability: float = 0.0, noise_seed: int | None = None,
):
    """Draw iid uniform inputs, latent Y*, and observed Y = Y* XOR E."""
    import numpy as np
    import pandas as pd
    if n_sequences < 1 or n < 2 or not 1 <= rule["lag"] < n:
        raise ValueError("require positive n_sequences, n >= 2, and 1 <= lag < n")
    if not 0 <= noise_probability < 0.5:
        raise ValueError("noise_probability must satisfy 0 <= pi < 0.5")
    keys_in_rank_order = ordered_keys(rule)
    noise_seed = seed + 1_000_000 if noise_seed is None else noise_seed
    rng = np.random.default_rng(seed)
    seqs_arr = np.array(ALPHABET)[rng.integers(0, len(ALPHABET), size=(n_sequences, n))]
    latent = np.array([
        oc_label(row, keys_in_rank_order, rule["lag"]) for row in seqs_arr.tolist()
    ], dtype=int)
    flips = (np.random.default_rng(noise_seed).random(n_sequences) < noise_probability).astype(int)
    df = pd.DataFrame({
        "Sequences": ["\x1f".join(row) for row in seqs_arr.tolist()],
        "LatentOutcome": latent,
        "LabelFlipped": flips,
        "Outcome": latent ^ flips,
    })
    df.attrs["generation"] = {
        "noise_probability": noise_probability,
        "sequence_seed": seed, "noise_seed": noise_seed,
        "input_distribution": "iid_uniform_alphabet", "alphabet": ALPHABET,
        "n_sequences": n_sequences, "sequence_length": n,
        "latent_prevalence": float(latent.mean()),
        "observed_prevalence": float(df["Outcome"].mean()),
        "realized_flip_rate": float(flips.mean()),
        "dataset_sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
    }
    return df


def featurise(df, fam: str, lag: int, n: int):
    import numpy as np
    seqs = df["Sequences"].tolist()
    if fam == "A1_count26":
        ex = feat_count26
    elif fam == "B_residue":
        ex = lambda t: feat_residue(t, lag)
    elif fam == "C1_lagpair":
        ex = lambda t: feat_lag_pair(t, lag)
    elif fam == "C2_lagpair_pos":
        ex = lambda t: feat_lag_pair_position(t, lag, n)
    else:
        raise ValueError(fam)
    return np.stack([ex(tokens(s)) for s in seqs], 0)


def split_rule_dataset(df) -> dict:
    """Deterministic 80/10/10 split of iid rows."""
    train_end, val_end = int(0.8 * len(df)), int(0.9 * len(df))
    if not 0 < train_end < val_end < len(df):
        raise ValueError("n_per_rule must permit nonempty 80/10/10 splits")
    return {"train": df.iloc[:train_end], "val": df.iloc[train_end:val_end],
            "test": df.iloc[val_end:]}


def select_threshold(y_validation, scores) -> float:
    """Select F1 threshold on validation labels over {0.10, ..., 0.89}."""
    import numpy as np
    thresholds = np.arange(10, 90) / 100
    truth = np.asarray(y_validation, dtype=bool)
    values = []
    for threshold in thresholds:
        positive = scores >= threshold
        denominator = positive.sum() + truth.sum()
        values.append(2 * (positive & truth).sum() / denominator if denominator else 0.)
    return float(thresholds[np.argmax(values)])


def evaluate_model(train_x, train_y, val_x, val_y, test_x, test_y, seed):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    if len(np.unique(train_y)) < 2:
        raise ValueError("training split has only one class; increase n_per_rule")
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    clf.fit(train_x, train_y)
    threshold = select_threshold(val_y, clf.predict_proba(val_x)[:, 1])
    scores = clf.predict_proba(test_x)[:, 1]
    auc = float(roc_auc_score(test_y, scores)) if len(np.unique(test_y)) >= 2 else float("nan")
    return auc, float(f1_score(test_y, scores >= threshold, zero_division=0)), threshold


def appendix_d4_spec(args):
    train_end, val_end = int(0.8 * args.n_per_rule), int(0.9 * args.n_per_rule)
    return {
        "protocol": "appendix_d4", "status": "D4", "paper": "Appendix D.4",
        "family": "OC-Noisy", "n_rules": args.n_rules,
        "rule_parameters": ["S", "kappa", "lambda"],
        "sequence_length": args.n, "alphabet_size": len(ALPHABET), "n_keys": args.m,
        "noise_probability": args.noise_probability,
        "rows": {"train": train_end, "val": val_end - train_end, "test": args.n_per_rule - val_end},
        "features": [FEATURE_NAMES[name] for name in args.families],
        "model": "l2_logistic_regression",
        "evaluation": ["within_rule", "across_rule"],
        "lag_domain": [args.fixed_lag] if args.fixed_lag else list(range(1, args.n)),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_rules", type=int, default=6)
    p.add_argument("--n_per_rule", type=int, default=sum(PAPER_ROWS.values()))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--m", type=int, default=6)
    p.add_argument("--families", nargs="+",
                   choices=["A1_count26", "B_residue", "C1_lagpair", "C2_lagpair_pos"],
                   default=["A1_count26", "C1_lagpair"])
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "heldout_rule_oc_noisy.csv")
    p.add_argument("--fixed_lag", type=int, default=0,
                   help="if >0, all rules use this lag (pure S/kappa variation)")
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--noise-probability", type=float, default=0.0,
                   help="Published D.4 table uses 0.0; 0.3 matches OC-Noisy label noise but cannot reach AUC 1.00")
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true", help="generate data and fit models")
    execution.add_argument("--dry-run", action="store_true", help="print the Appendix D.4 specification (the default)")
    args = p.parse_args(argv)
    if args.n_rules < 2 or args.n_per_rule < 10:
        p.error("at least two rules and ten examples per rule are required")
    if args.n < 2 or not 1 <= args.m <= len(ALPHABET) or args.seed_base < 0:
        p.error("require n >= 2, 1 <= m <= 26, and a nonnegative seed")
    if not 0 <= args.noise_probability < 0.5:
        p.error("noise probability must satisfy 0 <= pi < 0.5")
    if args.fixed_lag < 0 or args.fixed_lag >= args.n:
        p.error("fixed_lag must be zero (random over 1..n-1) or in [1, n-1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps(appendix_d4_spec(args), indent=2))
        return
    import numpy as np
    import pandas as pd
    import sklearn
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or args.out.with_suffix(".manifest.json").exists():
        raise FileExistsError("output path already exists")
    lag_choices = (args.fixed_lag,) if args.fixed_lag else tuple(range(1, args.n))
    rules = [sample_rule(args.seed_base + i, m=args.m, lag_choices=lag_choices)
             for i in range(args.n_rules)]
    datasets, generation = [], []
    for i, rule in enumerate(rules):
        ds = generate_rule_dataset(
            rule, args.n_per_rule, n=args.n, seed=args.seed_base + 1000 + i,
            noise_probability=args.noise_probability,
        )
        generation.append({"rule_index": i, "rule": rule, **ds.attrs["generation"]})
        datasets.append(split_rule_dataset(ds))

    manifest = {
        **appendix_d4_spec(args),
        "status": "running",
        "noise_probability": args.noise_probability,
        "evaluation_target": "observed_noisy_labels" if args.noise_probability else "latent_labels",
        "split_proportions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_row_boundaries": [0, int(.8 * args.n_per_rule), int(.9 * args.n_per_rule), args.n_per_rule],
        "threshold_selection": "validation F1 on {0.10, 0.11, ..., 0.89}",
        "offrule_validation": "validation rows from source rules only",
        "feature_lag_policy": "each dataset uses its own rule lag",
        "families": args.families, "rules": generation,
        "estimator": {"class": "sklearn.linear_model.LogisticRegression",
                      "solver": "lbfgs", "C": 1.0, "max_iter": 2000,
                      "random_state": args.seed_base},
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "src/experiments/paper_data.py")
        },
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }
    if len({rule["lag"] for rule in rules}) > 1 and any(
        fam in args.families for fam in ("B_residue", "C2_lagpair_pos")
    ):
        raise ValueError("B_residue/C2_lagpair_pos require --fixed_lag for a shared feature space")
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    out_rows = []
    for fam in args.families:
        t0 = time.time()
        features = [{split: featurise(df, fam, rules[i]["lag"], args.n)
                     for split, df in ds.items()} for i, ds in enumerate(datasets)]
        build_t = time.time() - t0
        for heldout, rule in enumerate(rules):
            other = [i for i in range(len(rules)) if i != heldout]
            source_x = {split: np.concatenate([features[i][split] for i in other])
                        for split in ("train", "val")}
            source_y = {split: np.concatenate([datasets[i][split]["Outcome"].values for i in other])
                        for split in ("train", "val")}
            target_y = {split: datasets[heldout][split]["Outcome"].values
                        for split in ("train", "val", "test")}
            auc_off, f1_off, threshold_off = evaluate_model(
                source_x["train"], source_y["train"], source_x["val"], source_y["val"],
                features[heldout]["test"], target_y["test"], args.seed_base,
            )
            auc_on, f1_on, threshold_on = evaluate_model(
                features[heldout]["train"], target_y["train"],
                features[heldout]["val"], target_y["val"],
                features[heldout]["test"], target_y["test"], args.seed_base,
            )
            out_rows.append({
                "heldout_rule_idx": heldout, "heldout_S": "".join(rule["S"]),
                "heldout_kappa": json.dumps(rule["kappa"], sort_keys=True),
                "heldout_lag": rule["lag"], "family": fam, "model": "logreg_L2",
                "noise_probability": args.noise_probability,
                "AUC_offrule": auc_off, "F1_offrule": f1_off, "AUC_onrule": auc_on,
                "F1_onrule": f1_on, "threshold_offrule": threshold_off,
                "threshold_onrule": threshold_on, "threshold_source": "validation",
                "n_train_offrule": len(source_y["train"]), "n_val_offrule": len(source_y["val"]),
                "n_test_offrule": len(target_y["test"]),
                "rho_test_offrule": float(target_y["test"].mean()),
                "rho_latent_test_offrule": float(datasets[heldout]["test"]["LatentOutcome"].mean()),
                "build_sec": build_t,
            })
            pd.DataFrame(out_rows).to_csv(args.out, index=False)
    manifest["status"] = "completed"
    manifest["result_rows"] = len(out_rows)
    manifest["results_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
