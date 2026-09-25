#!/usr/bin/env python3
"""Fit the three unreported OC/GPC stacked-pair cells using phase2c settings.

Run with .venv/bin/python -B scripts/complete_feature_table.py.
The first 100,000 training and 50,000 validation/test rows are used, as in
phase2c_lag_agnostic.py. XGBoost gets dense float32 data so zero counts are
observed zeros, not absent sparse entries interpreted as missing values.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost as xgb
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from audit_kip_feature_table import check_features, stacked_pairs
from audit_kip_ngrams import ROOT, digest

DATA = ROOT / "data/simulation/tested"
OUT = ROOT / "results/feature_table_completed_20260918"
RECIPE = ROOT / "repro/src/analysis/mechanism_id/scripts/phase2c_lag_agnostic.py"
SEED = 20260421


def load(tag, split):
    limit = 100000 if split == "train" else 50000
    xp, yp = DATA / f"X_{split}_{tag}.csv", DATA / f"y_{split}_{tag}.csv"
    x, y = pd.read_csv(xp), pd.read_csv(yp)["Outcome"]
    assert len(x) == len(y) and len(x) >= limit
    total = len(x)
    sequences = x.Sequences.iloc[:limit].tolist()
    tokens = [sequence.split("\x1f") for sequence in sequences]
    assert all(len(row) == 20 for row in tokens)
    assert all(len(token) == 1 and "A" <= token <= "Z" for row in tokens for token in row)
    encoded = np.array([[ord(token) - 65 for token in row] for row in tokens], dtype=np.int64)
    outcomes = y.iloc[:limit].to_numpy()
    assert np.isin(outcomes, [0, 1]).all()
    outcomes = outcomes.astype(np.int8)
    assert len(np.unique(outcomes)) == 2
    checks = {
        "split": split, "available_rows": total, "used_rows": limit,
        "row_selection": f"first {limit} rows, no shuffle or subsampling",
        "positive_rate": float(outcomes.mean()),
        "within_used_split_duplicates": len(sequences) - len(set(sequences)),
        "x_sha256": digest(xp), "y_sha256": digest(yp),
    }
    if tag == "test_just_pair":
        keys = [ord(key) - 65 for key in "WDQJXN"]
        oracle = (np.isin(encoded, keys).sum(axis=1) % 2 == 0).astype(np.int8)
        np.testing.assert_array_equal(oracle, outcomes)
        checks["all_labels_match_gpc_rule"] = True
    return encoded, outcomes, sequences, checks


def run(task, threads):
    tag, dataset, model = {
        "gpc_l1": ("test_just_pair", "GPC", "logreg_L1"),
        "gpc_xgb": ("test_just_pair", "GPC", "xgboost"),
        "oc_det_xgb": ("6", "OC-Det.", "xgboost"),
    }[task]
    task_dir = OUT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"{task}: loading and checking input splits", flush=True)
    split_names = ("train", "val", "test") if model == "xgboost" else ("train", "test")
    arrays, labels, sequence_sets, checks = {}, {}, {}, []
    for split in split_names:
        tokens, labels[split], sequences, check = load(tag, split)
        sequence_sets[split] = set(sequences)
        checks.append(check)
        features = stacked_pairs(tokens)
        check_features(tokens, features)
        arrays[split] = features.astype(np.float32).toarray() if model == "xgboost" else features
        print(f"{task}: {split} feature shape {arrays[split].shape}; dtype {arrays[split].dtype}", flush=True)
        del features, tokens, sequences
    overlaps = {f"{a}/{b}": len(sequence_sets[a] & sequence_sets[b]) for i, a in enumerate(split_names) for b in split_names[i + 1:]}
    assert all(value == 0 for value in overlaps.values()), overlaps
    del sequence_sets
    gc.collect()
    manifest = {
        "task": task, "dataset": dataset, "tag": tag,
        "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "xgboost": xgb.__version__},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), RECIPE, ROOT / "scripts/audit_kip_feature_table.py", ROOT / "scripts/audit_kip_ngrams.py"]},
        "data": checks, "split_overlaps": overlaps,
        "features": "19 separate gaps x 26 x 26 directed raw pair counts; no key/rank input, no scaling",
        "feature_dimensions": 12844, "literal_feature_check": "passed for first 20 rows in every used split",
        "feature_storage": "dense float32; zeros are observed zeros" if model == "xgboost" else "CSR float64; zeros are numeric zeros",
        "validation_use": "logloss monitoring only; no early stopping or selection" if model == "xgboost" else "none",
        "hyperparameter_selection": "Fixed phase2c settings; no test-based tuning",
    }
    if model == "xgboost":
        classifier = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, tree_method="hist", n_jobs=threads, random_state=SEED, eval_metric="logloss")
    else:
        classifier = LogisticRegression(l1_ratio=1.0, C=0.5, max_iter=3000, solver="liblinear", random_state=SEED, tol=1e-4)
    manifest["parameters"] = classifier.get_params()
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{task}: fitting {model}", flush=True)
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if model == "xgboost":
            classifier.fit(arrays["train"], labels["train"], eval_set=[(arrays["val"], labels["val"])], verbose=50)
        else:
            classifier.fit(arrays["train"], labels["train"])
    fit_seconds = time.monotonic() - started
    scores = classifier.predict_proba(arrays["test"])[:, 1]
    auc = float(roc_auc_score(labels["test"], scores))
    np.savez_compressed(task_dir / "predictions.npz", test_probabilities=scores, test_labels=labels["test"])
    if model == "xgboost":
        assert classifier.get_booster().num_boosted_rounds() == 300
        classifier.save_model(task_dir / "model.ubj")
        (task_dir / "booster_config.json").write_text(classifier.get_booster().save_config() + "\n")
        (task_dir / "validation_history.json").write_text(json.dumps(classifier.evals_result(), indent=2) + "\n")
        reloaded = xgb.XGBClassifier(n_jobs=threads)
        reloaded.load_model(task_dir / "model.ubj")
        np.testing.assert_array_equal(scores[:256], reloaded.predict_proba(arrays["test"][:256])[:, 1])
        manifest["model_reload_check"] = "passed on first 256 test rows"
        manifest["boosting_rounds"] = 300
    else:
        np.savez_compressed(task_dir / "coefficients.npz", coefficients=classifier.coef_, intercept=classifier.intercept_)
        manifest["iterations"] = classifier.n_iter_.tolist()
        manifest["nonzero_coefficients"] = int(np.count_nonzero(classifier.coef_))
    result = {
        "task": task, "dataset": dataset, "tag": tag, "family": "stacked_all_lag", "model": model,
        "test_auc": auc, "n_train": len(labels["train"]), "n_test": len(labels["test"]),
        "seed": SEED, "feature_dimensions": 12844, "fit_seconds": fit_seconds,
    }
    manifest.update({"result": result, "completed_utc": datetime.now(timezone.utc).isoformat(), "convergence_warning": False})
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (task_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def finish_table():
    tasks = ("gpc_l1", "gpc_xgb", "oc_det_xgb")
    records = [json.loads((OUT / task / "metrics.json").read_text()) for task in tasks if (OUT / task / "metrics.json").exists()]
    pd.DataFrame(records).to_csv(OUT / "new_metrics.csv", index=False)
    if len(records) != len(tasks):
        return
    result = {row["task"]: row["test_auc"] for row in records}
    source_table = ROOT / "results/kip_feature_table_20260918/table.tex"
    text = source_table.read_text()
    (OUT / "previous_table.tex").write_text(text)
    formatted = lambda value: f"{value:.3f}".removeprefix("0")
    text = text.replace(r"All-lag pair LR ($L_1$) & .997 & .632 & --- & .500 & .502", rf"All-lag pair LR ($L_1$) & .997 & .632 & {formatted(result['gpc_l1'])} & .500 & .502")
    text = text.replace(r"All-lag pair XGBoost & --- & .667 & --- & .590 & .501", rf"All-lag pair XGBoost & {formatted(result['oc_det_xgb'])} & .667 & {formatted(result['gpc_xgb'])} & .590 & .501")
    text = text.replace("A dash denotes an unreported cell; N/A indicates that no generating lag exists.\nAll-lag rows use\n", "N/A indicates that no generating lag exists. All-lag rows use\n")
    text = text.replace("KIP $L_1$ results use 100,000 training and 50,000\ntest sequences.", "New OC/GPC all-lag results and KIP $L_1$ results use the first\n100,000 training and 50,000 test sequences.")
    assert "---" not in text
    (OUT / "table.tex").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=("gpc_l1", "gpc_xgb", "oc_det_xgb"), default=["gpc_l1", "gpc_xgb", "oc_det_xgb"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        run(task, args.threads)
        gc.collect()
        finish_table()


if __name__ == "__main__":
    main()
