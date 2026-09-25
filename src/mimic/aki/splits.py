"""Leakage-safe patient-level split utilities for AKI episodes."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _section(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        value = config.get("splitting", config)
    else:
        value = getattr(config, "splitting", None)
    if not isinstance(value, Mapping):
        raise ValueError("configuration section 'splitting' is required")
    return value


def _required(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"splitting.{key} is required and has no default")
    return cfg[key]


def _subject_strata(episodes: pd.DataFrame, strategy: str) -> pd.DataFrame:
    if strategy == "none":
        return episodes[["subject_id"]].drop_duplicates().assign(stratum="all")
    if "label" not in episodes:
        raise ValueError("stratified splitting requires a label column")
    if strategy == "single_label":
        counts = episodes.groupby("subject_id")["label"].nunique()
        bad = counts[counts != 1]
        if not bad.empty:
            raise ValueError(
                "single_label stratification received subjects with multiple labels: "
                f"{bad.index.tolist()[:10]}"
            )
        return (
            episodes[["subject_id", "label"]]
            .drop_duplicates("subject_id")
            .rename(columns={"label": "stratum"})
        )
    if strategy == "phenotype_signature":
        rows = []
        for subject_id, group in episodes.groupby("subject_id", sort=True):
            signature = "|".join(sorted({str(v) for v in group["label"] if pd.notna(v)}))
            rows.append({"subject_id": subject_id, "stratum": signature})
        return pd.DataFrame(rows)
    raise ValueError(
        "splitting.stratification must be 'none', 'single_label', or " "'phenotype_signature'"
    )


def _can_stratify(values: pd.Series, first_test_fraction: float) -> bool:
    counts = values.value_counts()
    if counts.empty or int(counts.min()) < 2:
        return False
    n_test = int(np.ceil(len(values) * first_test_fraction))
    n_train = len(values) - n_test
    return n_test >= len(counts) and n_train >= len(counts)


def _split_indices(
    frame: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    rare_policy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = frame.index.to_numpy()
    remainder = validation_fraction + test_fraction
    strata: pd.Series | None = frame["stratum"]
    if not _can_stratify(frame["stratum"], remainder):
        if rare_policy == "error":
            raise ValueError("subject strata are too sparse for the requested split fractions")
        if rare_policy == "unstratified":
            strata = None
        else:
            raise ValueError("splitting.rare_stratum_policy must be 'error' or 'unstratified'")

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=remainder,
        random_state=seed,
        stratify=None if strata is None else strata.loc[indices],
    )
    temp = frame.loc[temp_idx]
    test_share = test_fraction / remainder
    temp_strata: pd.Series | None = temp["stratum"]
    if strata is None or not _can_stratify(temp["stratum"], test_share):
        if rare_policy == "error" and strata is not None:
            raise ValueError("temporary subject strata are too sparse for validation/test split")
        temp_strata = None
    val_idx, test_idx = train_test_split(
        temp.index.to_numpy(),
        test_size=test_share,
        random_state=seed,
        stratify=temp_strata,
    )
    return train_idx, val_idx, test_idx


def make_patient_splits(episodes: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Split unique subjects once and return one row per subject.

    The function rejects duplicated subject inputs only indirectly by reducing
    episodes to a configured subject-level stratum before invoking sklearn.
    This avoids the leakage bug caused by passing episode rows to a row-level
    splitter.
    """

    if "subject_id" not in episodes:
        raise ValueError("episodes must contain subject_id")
    cfg = _section(config)
    train_fraction = float(_required(cfg, "train_fraction"))
    validation_fraction = float(_required(cfg, "validation_fraction"))
    test_fraction = float(_required(cfg, "test_fraction"))
    total = train_fraction + validation_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to one, got {total}")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("all split fractions must be positive")
    strategy = str(_required(cfg, "stratification"))
    seed = int(_required(cfg, "split_seed"))
    rare_policy = str(_required(cfg, "rare_stratum_policy"))

    subjects = _subject_strata(episodes, strategy).sort_values("subject_id").reset_index(drop=True)
    if subjects.empty:
        return pd.DataFrame(columns=["subject_id", "split", "stratum", "split_seed"])
    if subjects["subject_id"].duplicated().any():
        raise AssertionError("internal error: subject strata are not unique")
    if len(subjects) < 3:
        raise ValueError("at least three unique subjects are required for train/val/test")

    train_idx, val_idx, test_idx = _split_indices(
        subjects,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        rare_policy=rare_policy,
    )
    result = subjects.copy()
    result["split"] = pd.NA
    result.loc[train_idx, "split"] = "train"
    result.loc[val_idx, "split"] = "validation"
    result.loc[test_idx, "split"] = "test"
    result["split_seed"] = seed
    validate_patient_splits(result)
    return result


def validate_patient_splits(splits: pd.DataFrame) -> None:
    required = {"subject_id", "split"}
    missing = required.difference(splits.columns)
    if missing:
        raise AssertionError(f"split table missing columns: {sorted(missing)}")
    if splits["subject_id"].duplicated().any():
        raise AssertionError("a subject appears more than once in the split map")
    allowed = {"train", "validation", "test"}
    observed = set(splits["split"].dropna())
    if not observed.issubset(allowed) or splits["split"].isna().any():
        raise AssertionError(f"invalid split labels: {sorted(observed)}")
    sets = {name: set(splits.loc[splits["split"] == name, "subject_id"]) for name in allowed}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left].intersection(sets[right])
        if overlap:
            raise AssertionError(
                f"patient overlap between {left} and {right}: {sorted(overlap)[:10]}"
            )


def attach_patient_splits(episodes: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """Attach a validated subject split map to every episode."""

    validate_patient_splits(splits)
    if "split" in episodes:
        episodes = episodes.drop(columns=["split"])
    out = episodes.merge(splits[["subject_id", "split"]], on="subject_id", how="left")
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), "subject_id"].drop_duplicates().tolist()
        raise ValueError(f"episodes contain subjects absent from split map: {missing[:10]}")
    return out
