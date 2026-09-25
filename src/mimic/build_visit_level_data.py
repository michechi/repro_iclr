"""
Build visit-level dataset from CKD cohort for temporal analysis.

Groups diagnosis codes by admission (hadm_id), represents each visit as a
multi-hot vector of CCS codes, and keeps only patients with >=3 admissions
to ensure genuine multi-step temporal sequences.

Output: data/processed/mimic_training/visit_level_{train,val,test}.pkl
"""

import json
import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data" / "processed" / "ckd_cohort_ccs.csv"
OUT_DIR = ROOT / "data" / "processed" / "mimic_training"

SEED = 42
MIN_VISITS = 1  # keep all patients to match code-level cohort exactly


def main():
    df = pd.read_csv(COHORT)
    print(f"Loaded cohort: {len(df)} patients")

    # Build global vocabulary (same as training pipeline)
    all_codes = set()
    for c in df["codes"]:
        all_codes.update(json.loads(c))
    vocab = sorted(all_codes)
    code_to_idx = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)
    print(f"Vocabulary: {vocab_size} CCS codes")

    # Build visit-level sequences
    records = []
    for _, row in df.iterrows():
        codes = json.loads(row["codes"])
        hadms = json.loads(row["hadm_ids"])

        # Group codes by hadm_id, preserving admission order
        visits = OrderedDict()
        for c, h in zip(codes, hadms):
            visits.setdefault(h, set()).add(c)

        if len(visits) < MIN_VISITS:
            continue

        # Convert each visit to multi-hot vector
        visit_vectors = []
        for hadm_id, code_set in visits.items():
            v = np.zeros(vocab_size, dtype=np.float32)
            for c in code_set:
                if c in code_to_idx:
                    v[code_to_idx[c]] = 1.0
            visit_vectors.append(v)

        records.append({
            "subject_id": row["subject_id"],
            "label": row["label"],
            "num_visits": len(visit_vectors),
            "visit_sequence": np.stack(visit_vectors),  # shape: (num_visits, vocab_size)
        })

    print(f"\nFiltered to >= {MIN_VISITS} visits: {len(records)} patients")

    labels = [r["label"] for r in records]
    n1 = sum(labels)
    n0 = len(labels) - n1
    print(f"  Y=1: {n1} ({100*n1/len(records):.1f}%)")
    print(f"  Y=0: {n0}")

    num_visits = [r["num_visits"] for r in records]
    print(f"  Visits: median={np.median(num_visits):.0f}, mean={np.mean(num_visits):.1f}, "
          f"max={np.max(num_visits)}")

    # Stratified 60/20/20 split (same ratios as existing pipeline)
    indices = list(range(len(records)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=SEED
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=SEED
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for name, idx in splits.items():
        subset = [records[i] for i in idx]
        pos = sum(r["label"] for r in subset)
        print(f"  {name}: {len(subset)} patients, {pos} positive ({100*pos/len(subset):.1f}%)")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "vocab": vocab,
        "code_to_idx": code_to_idx,
        "vocab_size": vocab_size,
        "min_visits": MIN_VISITS,
        "seed": SEED,
    }

    for name, idx in splits.items():
        subset = [records[i] for i in idx]
        path = OUT_DIR / f"visit_level_{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(subset, f)
        print(f"Saved {path} ({len(subset)} patients)")

    meta_path = OUT_DIR / "visit_level_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved {meta_path}")


if __name__ == "__main__":
    main()
