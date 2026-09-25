#!/usr/bin/env python3
"""Complete KIP feature-family cells and fit the missing stacked-pair L1 LR.

Uses the L1 recipe in phase2c_lag_agnostic.py: first 100,000 training rows,
all 50,000 test rows, C=0.5, liblinear, 3,000 iterations, seed 20260421.
Only sparse storage differs from that script's dense feature construction.
"""
from __future__ import annotations

import csv
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
from scipy.sparse import csr_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from audit_kip_ngrams import ROOT, digest, load_split

OUT = ROOT / "results/kip_feature_table_20260918"
DATA = ROOT / "data/simulation/kip_ordered_recovered"
CONTROL = ROOT / "repro/src/analysis/mechanism_id/results/kip_audit.csv"
REVEAL = ROOT / "repro/src/analysis/mechanism_id/results/kip_oracle_reveal.csv"
RECIPE = ROOT / "repro/src/analysis/mechanism_id/scripts/phase2c_lag_agnostic.py"
REVIEW = ROOT / "results/kip_results_review_20260918"
NGRAMS = ROOT / "results/kip_ngram_audit_20260918/metrics.csv"
SEED = 20260421


def stacked_pairs(tokens):
    """Counts of every directed A-Z pair at each of 19 separate gaps."""
    n = tokens.shape[1]
    indices = np.concatenate([
        (lag - 1) * 676 + tokens[:, :-lag] * 26 + tokens[:, lag:]
        for lag in range(1, n)
    ], axis=1).astype(np.int32)
    entries = indices.shape[1]
    features = csr_matrix((
        np.ones(indices.size, dtype=np.float64),
        indices.ravel(),
        np.arange(len(tokens) + 1, dtype=np.int64) * entries,
    ), shape=(len(tokens), (n - 1) * 676))
    features.sum_duplicates()
    features.sort_indices()
    assert np.all(np.asarray(features.sum(axis=1)).ravel() == n * (n - 1) // 2)
    return features


def check_features(tokens, features):
    for i, sequence in enumerate(tokens[:20]):
        literal = np.zeros((19, 26, 26))
        for gap in range(1, 20):
            for start in range(20 - gap):
                literal[gap - 1, sequence[start], sequence[start + gap]] += 1
        np.testing.assert_array_equal(features[i].toarray().ravel(), literal.ravel())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    control, reveal, grams = pd.read_csv(CONTROL), pd.read_csv(REVEAL), pd.read_csv(NGRAMS)
    neural_paths = [REVIEW / f"{cohort}_neural_evaluations.csv" for cohort in ("latest", "historical")]
    neural = pd.concat([pd.read_csv(path) for path in neural_paths], ignore_index=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__), ROOT / "scripts/audit_kip_ngrams.py", CONTROL, REVEAL, RECIPE, NGRAMS, *neural_paths]},
        "input_sha256": {}, "new_fits": [], "derived_checks": [],
        "best_neural_definition": "Maximum recorded test AUC across completed ordered-trained, ordered-evaluated supervised raw-sequence neural runs; descriptive observed maximum, not a seed mean or validation-selected architecture.",
        "known_lag": "Not applicable: KIP has no fixed generating lag.",
    }
    rows = []

    def record(method, m, auc, status, source, **extra):
        rows.append({"method": method, "m": m, "test_auc": auc, "status": status, "source": source, **extra})

    for m in (4, 6):
        tag = f"kip_m{m}"
        train, labels = load_split(DATA, tag, "train")
        test, test_labels = load_split(DATA, tag, "test")
        rule_path = DATA / f"{tag}_rule.json"
        rule = json.loads(rule_path.read_text())
        keys = [ord(key) - ord("A") for key in rule["key_letters_in_kappa_order"]]
        for split in ("train", "test"):
            for prefix in ("X", "y"):
                path = DATA / f"{prefix}_{split}_{tag}.csv"
                manifest["input_sha256"][str(path.relative_to(ROOT))] = digest(path)
        manifest["input_sha256"][str(rule_path.relative_to(ROOT))] = digest(rule_path)

        for tokens, outcomes in ((train, labels), (test, test_labels)):
            key_counts = np.stack([(tokens == key).sum(axis=1) for key in keys], axis=1)
            assert np.all(key_counts == 1)
            positions = np.stack([(tokens == key).argmax(axis=1) for key in keys], axis=1)
            inversions = sum(positions[:, i] > positions[:, j] for i in range(m) for j in range(i + 1, m))
            oracle = (inversions % 2 == 0).astype(np.int8)
            np.testing.assert_array_equal(oracle, outcomes)
        constant_auc = float(roc_auc_score(test_labels, np.full(len(test), labels.mean())))
        oracle_auc = float(roc_auc_score(test_labels, oracle))
        assert constant_auc == 0.5 and oracle_auc == 1.0
        historical_oracle = reveal[(reveal.m == m) & (reveal.rung == "oracle_closed_form")].iloc[0]
        assert historical_oracle.AUC == oracle_auc
        manifest["derived_checks"].append({"m": m, "train_rows": len(train), "test_rows": len(test), "all_key_counts_one": True, "all_train_and_test_labels_match_oracle": True})
        record("Chance", m, 0.5, "reference", "constant score")
        record("Key-count baseline", m, constant_auc, "derived_and_data_verified", "Every sequence has the identical m-dimensional all-ones key-count vector.")
        record("Known-lag pair LR", m, None, "not_applicable", "KIP has no fixed generating lag.")
        record("Oracle AUC", m, oracle_auc, "recomputed_and_matches_retained", str(REVEAL.relative_to(ROOT)))

        for method, family, model in [
            ("Full-count XGBoost", "unigram_counts", "xgboost"),
            ("All-lag pair XGBoost (stacked)", "stacked_all_lag", "xgboost"),
            ("All-lag pair LR (L2; supplemental)", "stacked_all_lag", "logreg_L2"),
            ("Pair XGBoost (max over lags; supplemental)", "max_over_lag", "xgboost"),
            ("Pair XGBoost (sum over lags; supplemental)", "sum_over_lag", "xgboost"),
        ]:
            entry = control[(control.m == m) & (control.family == family) & (control.model == model)].iloc[0]
            record(method, m, float(entry.test_auc), "retained", str(CONTROL.relative_to(ROOT)), n_train=int(entry.n_train))
        subset = grams[grams.dataset == tag]
        best_gram = subset.loc[subset.test_auc_unseen_half.idxmax()]
        record("Best contiguous k-gram", m, float(best_gram.test_auc_unseen_half), "retained_previous_turn_audit", str(NGRAMS.relative_to(ROOT)), best_k=int(best_gram.k), n_train=int(best_gram.n_train))
        subset = neural[(neural.dataset == tag) & (neural.evaluation == "ordered")]
        best_neural = subset.loc[subset.test_auc.idxmax()]
        record("Best neural model", m, float(best_neural.test_auc), "retained_observed_maximum", str(best_neural.source), model=best_neural.model, seed=int(best_neural.seed), n_train=int(best_neural.n_train))

        print(f"{tag}: building stacked pair features (100,000 train / {len(test):,} test)", flush=True)
        features_train = stacked_pairs(train[:100000])
        features_test = stacked_pairs(test)
        check_features(train, features_train)
        check_features(test, features_test)
        classifier = LogisticRegression(l1_ratio=1.0, C=0.5, solver="liblinear", max_iter=3000, random_state=SEED, tol=1e-4)
        start = time.monotonic()
        print(f"{tag}: fitting L1 LR", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            classifier.fit(features_train, labels[:100000])
        probabilities = classifier.predict_proba(features_test)[:, 1]
        auc = float(roc_auc_score(test_labels, probabilities))
        evidence = {
            "m": m, "n_train": 100000, "n_test": len(test), "feature_dimensions": features_train.shape[1],
            "feature_storage": "CSR, raw unscaled counts; equivalent to dense phase2c representation",
            "parameters": classifier.get_params(), "l1_ratio_note": "l1_ratio=1.0 is the current sklearn API for the historical penalty='l1'.",
            "iterations": classifier.n_iter_.tolist(), "fit_seconds": time.monotonic() - start,
            "nonzero_coefficients": int(np.count_nonzero(classifier.coef_)), "test_auc": auc,
            "convergence_warning": False, "literal_feature_check": "passed",
            "validation_used": False, "selection": "Fixed existing phase2c hyperparameters; no test-based tuning.",
        }
        manifest["new_fits"].append(evidence)
        np.savez_compressed(OUT / f"{tag}_l1_fit.npz", coefficients=classifier.coef_, intercept=classifier.intercept_, test_probabilities=probabilities, test_labels=test_labels)
        record("All-lag pair LR (L1)", m, auc, "new_fit", f"{tag}_l1_fit.npz", n_train=100000, seed=SEED)
        print(json.dumps(evidence), flush=True)
        del features_train, features_test
        pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="method", columns="m", values="test_auc").to_string(), flush=True)


if __name__ == "__main__":
    main()
