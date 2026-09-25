from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


def make_patient_splits(
    subjects: Sequence[int],
    strata: Sequence[int],
    seed: int = 2026,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
) -> pd.DataFrame:
    """Patient-level stratified 70/15/15 split.

    Returns a DataFrame with columns [subject_id, split] where split is
    one of {"train", "val", "test"}.

    Falls back to unstratified ShuffleSplit if any stratum has fewer than 2
    members (e.g., tiny smoke-test slices).
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1 (got {train + val + test})")

    subjects = np.asarray(subjects)
    strata = np.asarray(strata)
    if len(subjects) != len(strata):
        raise ValueError("subjects and strata must have same length")

    split_labels = np.full(len(subjects), "train", dtype=object)
    # For tiny samples (smoke tests), a proper 70/15/15 is meaningless;
    # assign the first patient to val, second to test, rest to train.
    if len(subjects) < 10:
        if len(subjects) >= 3:
            split_labels[0] = "val"
            split_labels[1] = "test"
        return pd.DataFrame({"subject_id": subjects, "split": split_labels})

    min_class = int(pd.Series(strata).value_counts().min())
    stratified = min_class >= 2

    remainder = val + test
    val_size = val / remainder
    if stratified:
        first = StratifiedShuffleSplit(n_splits=1, test_size=remainder, random_state=seed)
        train_idx, temp_idx = next(first.split(np.zeros_like(subjects), strata))
        temp_strata = strata[temp_idx]
        second_min = int(pd.Series(temp_strata).value_counts().min())
        if second_min >= 2:
            second = StratifiedShuffleSplit(
                n_splits=1, test_size=1.0 - val_size, random_state=seed
            )
            val_local_idx, test_local_idx = next(
                second.split(np.zeros_like(temp_idx), temp_strata)
            )
        else:
            second = ShuffleSplit(n_splits=1, test_size=1.0 - val_size, random_state=seed)
            val_local_idx, test_local_idx = next(
                second.split(np.zeros_like(temp_idx))
            )
    else:
        first = ShuffleSplit(n_splits=1, test_size=remainder, random_state=seed)
        train_idx, temp_idx = next(first.split(np.zeros_like(subjects)))
        second = ShuffleSplit(n_splits=1, test_size=1.0 - val_size, random_state=seed)
        val_local_idx, test_local_idx = next(second.split(np.zeros_like(temp_idx)))

    split_labels[temp_idx[val_local_idx]] = "val"
    split_labels[temp_idx[test_local_idx]] = "test"
    split_labels[train_idx] = "train"

    return pd.DataFrame({"subject_id": subjects, "split": split_labels})


def make_shuffle_index(
    subjects: Sequence[int], n_per_patient: Sequence[int], seed: int = 2026
) -> pd.DataFrame:
    """Deterministic within-patient permutation index.

    Returns long-format (subject_id, seq_idx, shuffled_idx) — `seq_idx` is the
    original order, `shuffled_idx` is where that note maps after shuffling.
    """
    rows = []
    for subj, n in zip(subjects, n_per_patient):
        rng = np.random.default_rng(seed + (hash(int(subj)) & 0x7FFFFFFF))
        perm = rng.permutation(int(n))
        for i, j in enumerate(perm):
            rows.append((int(subj), i, int(j)))
    return pd.DataFrame(rows, columns=["subject_id", "seq_idx", "shuffled_idx"])
