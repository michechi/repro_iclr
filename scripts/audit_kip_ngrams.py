#!/usr/bin/env python3
"""Compute the contiguous k-gram appendix entries from recovered ordered KIP data.

Every training occurrence contributes its sequence label. Scores average
unsmoothed positive fractions over all contiguous windows. Both unseen-window
conventions are recorded: 0.5 (simulation/stat_test.py) and the training positive
rate (paper/paper.tex). Coverage is recorded over both distinct gram types and
gram occurrences; the original OC/GPC coverage table counts distinct types.

Run from the repository root:
    .venv/bin/python -B scripts/audit_kip_ngrams.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_split(directory, tag, split):
    sequences = pd.read_csv(directory / f"X_{split}_{tag}.csv")["Sequences"]
    labels = pd.read_csv(directory / f"y_{split}_{tag}.csv")["Outcome"].to_numpy()
    tokens = [sequence.split("\x1f") for sequence in sequences]
    assert len(tokens) == len(labels) and len(tokens) > 0
    assert all(len(row) == 20 for row in tokens)
    assert all(len(token) == 1 and "A" <= token <= "Z" for row in tokens for token in row)
    assert np.isin(labels, [0, 1]).all()
    encoded = np.array([[ord(token) - ord("A") for token in row] for row in tokens], dtype=np.int64)
    return encoded, labels.astype(np.int8)


def gram_codes(tokens, k):
    width = tokens.shape[1] - k + 1
    codes = np.zeros((len(tokens), width), dtype=np.int64)
    for offset in range(k):
        codes *= 26
        codes += tokens[:, offset:offset + width]
    return codes


def estimate(train, labels, test, k):
    train_codes = gram_codes(train, k)
    vocabulary, inverse, counts = np.unique(train_codes, return_inverse=True, return_counts=True)
    positives = np.bincount(inverse.ravel(), weights=np.repeat(labels, train_codes.shape[1]))
    probabilities = positives / counts
    test_codes = gram_codes(test, k)
    indices = np.searchsorted(vocabulary, test_codes)
    safe_indices = np.minimum(indices, len(vocabulary) - 1)
    seen = (indices < len(vocabulary)) & (vocabulary[safe_indices] == test_codes)
    scores = {}
    for name, fallback in (("half", 0.5), ("prior", float(labels.mean()))):
        values = np.where(seen, probabilities[safe_indices], fallback)
        scores[name] = values.mean(axis=1)
    distinct_test = np.unique(test_codes)
    distinct_seen = len(np.intersect1d(vocabulary, distinct_test, assume_unique=True))
    coverage = {
        "unique_train_grams": len(vocabulary),
        "unique_test_grams": len(distinct_test),
        "unique_test_grams_seen": distinct_seen,
        "coverage_unique": distinct_seen / len(distinct_test),
        "test_gram_occurrences": int(test_codes.size),
        "test_gram_occurrences_seen": int(seen.sum()),
        "coverage_occurrences": float(seen.mean()),
    }
    return scores, coverage


def verify_against_literal(train, labels, test):
    """Check counts, repeated windows, unseen windows and averaging independently."""
    train, labels, test = train[:300], labels[:300], test[:100]
    for k in range(1, 8):
        scores, coverage = estimate(train, labels, test, k)
        counts = defaultdict(lambda: [0, 0])
        for sequence, label in zip(train, labels, strict=True):
            for start in range(len(sequence) - k + 1):
                counts[tuple(sequence[start:start + k])][label] += 1
        test_types = set()
        for name, fallback in (("half", 0.5), ("prior", float(labels.mean()))):
            reference = []
            for sequence in test:
                windows = [tuple(sequence[start:start + k]) for start in range(len(sequence) - k + 1)]
                test_types.update(windows)
                reference.append(np.mean([counts[gram][1] / sum(counts[gram]) if gram in counts else fallback for gram in windows]))
            np.testing.assert_allclose(scores[name], reference, rtol=0, atol=1e-15)
        assert coverage["unique_train_grams"] == len(counts)
        assert coverage["unique_test_grams"] == len(test_types)
        assert coverage["unique_test_grams_seen"] == len(test_types.intersection(counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/simulation/kip_ordered_recovered")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/kip_ngram_audit_20260918")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    historical_path = ROOT / "repro/src/analysis/mechanism_id/results/kip_audit.csv"
    historical = pd.read_csv(historical_path)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "script_sha256": digest(Path(__file__)),
        "historical_csv_sha256": digest(historical_path),
        "data_directory": str(args.data_dir.resolve()),
        "estimator": "Unsmoothed occurrence-level P(Y=1|gram), averaged over all contiguous test windows; no position conditioning, fitting on all training rows only.",
        "coverage_unique": "Number of distinct test grams observed in training divided by number of distinct test grams.",
        "coverage_occurrences": "Number of test window occurrences observed in training divided by total number of test windows.",
        "validation_used": False,
        "input_sha256": {}, "historical_comparison": [],
    }
    rows = []
    for m in (4, 6):
        tag = f"kip_m{m}"
        train, labels = load_split(args.data_dir, tag, "train")
        test, test_labels = load_split(args.data_dir, tag, "test")
        verify_against_literal(train, labels, test)
        for split in ("train", "test"):
            for prefix in ("X", "y"):
                filename = f"{prefix}_{split}_{tag}.csv"
                manifest["input_sha256"][filename] = digest(args.data_dir / filename)
        for k in range(1, 8):
            scores, coverage = estimate(train, labels, test, k)
            row = {
                "dataset": tag, "k": k, "n_train": len(train), "n_test": len(test),
                "train_positive_rate": float(labels.mean()),
                "test_auc_unseen_half": float(roc_auc_score(test_labels, scores["half"])),
                "test_auc_unseen_prior": float(roc_auc_score(test_labels, scores["prior"])),
                **coverage,
            }
            rows.append(row)
            prior = historical[(historical["m"] == m) & (historical["family"] == f"kgram_k{k}")]
            if len(prior):
                previous = prior.iloc[0]
                manifest["historical_comparison"].append({
                    "dataset": tag, "k": k,
                    "historical_auc": float(previous["test_auc"]),
                    "historical_coverage": float(previous["coverage"]),
                    "auc_delta_half": row["test_auc_unseen_half"] - float(previous["test_auc"]),
                    "auc_delta_prior": row["test_auc_unseen_prior"] - float(previous["test_auc"]),
                    "coverage_occurrences_delta": row["coverage_occurrences"] - float(previous["coverage"]),
                })
            print(json.dumps(row), flush=True)
    manifest["literal_reference_check"] = "passed for k=1..7 for both KIP datasets and both unseen-gram fallbacks"
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
