#!/usr/bin/env python3
"""Aggregate-only integrity gates for the restricted MIMIC-IV AKI audit.

The pre-training gate may read patient-level Parquet files in private storage,
but it never emits identifiers, timestamps, event values, or prediction values.
The post-training gate privately validates predictions and emits violation
counts only; it never emits patient-level values.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mimic.aki.config import AkiAuditConfig, load_aki_config
from src.mimic.aki.experiment import (
    _aggregate,
    _aggregate_contrasts,
    _contrasts,
    _evaluate,
    _validate_model_dataset,
)
from src.mimic.aki.episodes import _Protocol


SPLITS = ("train", "validation", "test")
COHORT_ARTIFACTS = {
    "aki_admission_audit.parquet",
    "aki_cohort_flow.csv",
    "aki_creatinine_measurements.parquet",
    "aki_creatinine_rejections.parquet",
    "aki_unit_audit.csv",
}
PREPARED_ARTIFACTS = {
    "censor_reasons.csv",
    "episode_audit.parquet",
    "episode_flow.csv",
    "episodes.parquet",
    "labeled_episode_events.parquet",
    "matching_assignments.parquet",
    "matching_balance.csv",
    "matching_balance.parquet",
    "matching_estimability.csv",
    "matching_flow.csv",
    "matching_flow.parquet",
    "patient_splits.parquet",
    "primary_class_distribution.csv",
    "primary_events.parquet",
    "primary_labels.parquet",
    "primary_matched_events.parquet",
    "primary_matched_labels.parquet",
    "primary_matching_features.parquet",
    "protocol_metadata.csv",
    "protocol_metadata.parquet",
    "recovery_timing_sensitivity.csv",
    "recovery_timing_sensitivity.parquet",
    "run_manifest.json",
    "secondary_class_distribution.csv",
    "secondary_events.parquet",
    "secondary_labels.parquet",
}
EXPERIMENT_ARTIFACTS = {
    "aggregate_metrics.csv",
    "aggregate_metrics.parquet",
    "aggregate_ordered_control_contrasts.csv",
    "aggregate_ordered_control_contrasts.parquet",
    "condition_estimability.csv",
    "condition_estimability.parquet",
    "ordered_control_contrasts.csv",
    "ordered_control_contrasts.parquet",
    "predictions.parquet",
    "run_manifest.json",
    "seed_metrics.csv",
    "seed_metrics.parquet",
    "task_estimability.csv",
    "task_estimability.parquet",
}
PREDICTION_COLUMNS = [
    "task",
    "model",
    "condition",
    "feature_set",
    "model_seed",
    "permutation_seed",
    "episode_id",
    "subject_id",
    "hadm_id",
    "true_label",
    "probability_transient",
    "probability_persistent",
    "probability_relapsing",
]
BINARY_METRIC_COLUMNS = ["auroc", "f1", "recall"]
MULTICLASS_METRIC_COLUMNS = [
    "macro_ovr_auc",
    "macro_f1",
    "macro_recall",
    "ovr_auc_transient",
    "f1_transient",
    "recall_transient",
    "ovr_auc_persistent",
    "f1_persistent",
    "recall_persistent",
    "ovr_auc_relapsing",
    "f1_relapsing",
    "recall_relapsing",
]
SEED_METRIC_COLUMNS = [
    "task",
    "model",
    "condition",
    "feature_set",
    "model_seed",
    "permutation_seed",
    *BINARY_METRIC_COLUMNS,
    "n_samples",
    *MULTICLASS_METRIC_COLUMNS,
]
CONTRAST_COLUMNS = [
    "task",
    "model",
    "model_seed",
    "metric",
    "left_condition",
    "right_condition",
    "left_value",
    "right_value",
    "ordered_minus_control",
    "bootstrap_estimate",
    "confidence_lower",
    "confidence_upper",
    "confidence_level",
    "bootstrap_interval_method",
    "bootstrap_resamples_requested",
    "bootstrap_resamples_valid",
    "bootstrap_clustered_by_subject",
]
TASK_ESTIMABILITY_COLUMNS = ["task", "estimable", "non_estimable_reasons"]
CONDITION_ESTIMABILITY_COLUMNS = [
    "task",
    "model",
    "condition",
    "estimable",
    "reason",
]


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _csv_matches_parquet(csv_path: Path, parquet_frame: pd.DataFrame) -> bool:
    expected = io.StringIO()
    parquet_frame.to_csv(expected, index=False)
    return csv_path.read_text(encoding="utf-8") == expected.getvalue()


def _normalized_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_normalized_json(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _records_match_manifest(frame: pd.DataFrame, records: Any) -> bool:
    if not isinstance(records, list):
        return False
    observed = _normalized_json(frame.to_dict(orient="records"))
    expected = _normalized_json(records)
    return observed == expected


def _add_failure(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() == "true"


def _empty_reason(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) == 0
    return str(value).strip() in {"", "[]", "()", "nan"}


def _private_tree(root: Path) -> tuple[bool, int, int]:
    insecure = 0
    symlinks = 0
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            symlinks += 1
        mode = stat.S_IMODE(path.lstat().st_mode)
        if mode & 0o077:
            insecure += 1
    return insecure == 0 and symlinks == 0, insecure, symlinks


def _frames_equivalent(
    observed: pd.DataFrame, expected: pd.DataFrame, sort_columns: Sequence[str]
) -> bool:
    if set(observed.columns) != set(expected.columns):
        return False
    columns = sorted(observed.columns)
    left = observed[columns].sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    right = expected[columns].sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    if len(left) != len(right):
        return False
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                return False
        else:
            left_values = left[column].map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
            )
            right_values = right[column].map(
                lambda value: json.dumps(value, sort_keys=True, default=str)
            )
            if not left_values.equals(right_values):
                return False
    return True


def _smd(frame: pd.DataFrame, feature: str, class_a: str, class_b: str) -> float:
    left = pd.to_numeric(
        frame.loc[frame["label"] == class_a, feature], errors="coerce"
    ).dropna()
    right = pd.to_numeric(
        frame.loc[frame["label"] == class_b, feature], errors="coerce"
    ).dropna()
    if left.empty or right.empty:
        return float("nan")
    pooled = math.sqrt((float(left.var(ddof=0)) + float(right.var(ddof=0))) / 2.0)
    difference = float(left.mean() - right.mean())
    if pooled == 0:
        return 0.0 if difference == 0 else float("inf")
    return difference / pooled


def _class_report(
    name: str,
    labels: pd.DataFrame,
    required_labels: Sequence[str],
    split_map: pd.DataFrame,
    failures: list[str],
) -> dict[str, Any]:
    duplicate_episodes = int(labels["episode_id"].duplicated().sum())
    duplicate_subjects = int(labels["subject_id"].duplicated().sum())
    null_episode_ids = int(labels["episode_id"].isna().sum())
    joined = labels[["subject_id", "split"]].merge(
        split_map, on="subject_id", how="left", suffixes=("_label", "_map")
    )
    split_mismatches = int(
        (
            joined["split_map"].isna()
            | joined["split"].ne(joined["split_map"])
        ).sum()
    )
    counts = (
        labels.groupby(["split", "label"], dropna=False)
        .agg(n_episodes=("episode_id", "size"), n_subjects=("subject_id", "nunique"))
        .reset_index()
        .sort_values(["split", "label"], kind="mergesort")
    )
    all_classes = all(
        set(labels.loc[labels["split"] == split, "label"]) == set(required_labels)
        for split in SPLITS
    )
    split_names_exact = set(labels["split"]) == set(SPLITS)
    _add_failure(failures, duplicate_episodes == 0, f"{name}: duplicate episode rows")
    _add_failure(failures, duplicate_subjects == 0, f"{name}: duplicate subject rows")
    _add_failure(failures, null_episode_ids == 0, f"{name}: null episode identifiers")
    _add_failure(failures, split_mismatches == 0, f"{name}: split-map mismatch")
    _add_failure(failures, split_names_exact, f"{name}: split names are not exact")
    _add_failure(
        failures,
        all_classes,
        f"{name}: a required class is absent from at least one split",
    )
    return {
        "counts": _records(counts),
        "duplicate_episode_rows": duplicate_episodes,
        "duplicate_subject_rows": duplicate_subjects,
        "episode_id_nulls": null_episode_ids,
        "split_map_mismatch_rows": split_mismatches,
        "split_names_exact": split_names_exact,
        "all_required_classes_each_split": all_classes,
    }


def _event_report(
    name: str,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    class_labels: Sequence[str],
    config_hash: str,
    max_length: int,
    failures: list[str],
) -> dict[str, Any]:
    validation_error = ""
    try:
        _validate_model_dataset(events, labels, tuple(class_labels))
    except Exception as exc:
        validation_error = type(exc).__name__
        failures.append(f"{name}: model dataset validation failed")

    required = [
        "episode_id",
        "subject_id",
        "hadm_id",
        "sequence_index",
        "specimen_time",
        "split",
        "config_hash",
    ]
    required_nulls = int(events[required].isna().sum().sum())
    ordered = events.copy()
    ordered["_parsed_time"] = pd.to_datetime(ordered["specimen_time"], errors="coerce")
    invalid_parsed_times = int(ordered["_parsed_time"].isna().sum())
    ordered = ordered.sort_values(
        ["episode_id", "sequence_index"], kind="mergesort"
    ).reset_index(drop=True)
    expected_index = ordered.groupby("episode_id", sort=False).cumcount()
    index_violations = int(
        pd.to_numeric(ordered["sequence_index"], errors="coerce")
        .ne(expected_index)
        .sum()
    )
    time_reversals = int(
        ordered.groupby("episode_id", sort=False)["_parsed_time"]
        .diff()
        .dt.total_seconds()
        .lt(0)
        .sum()
    )
    duplicate_episode_times = int(
        events.duplicated(["episode_id", "specimen_time"]).sum()
    )
    lengths = events.groupby("episode_id", sort=False).size()
    maximum_length = int(lengths.max()) if not lengths.empty else 0
    over_limit = int((lengths > max_length).sum())
    event_hashes = set(events["config_hash"].dropna().astype(str))
    config_match = event_hashes == {config_hash}

    _add_failure(failures, required_nulls == 0, f"{name}: null required event cells")
    _add_failure(failures, invalid_parsed_times == 0, f"{name}: invalid event timestamps")
    _add_failure(
        failures,
        index_violations == 0,
        f"{name}: sequence-index integrity failure",
    )
    _add_failure(failures, time_reversals == 0, f"{name}: event time reversal")
    _add_failure(
        failures,
        duplicate_episode_times == 0,
        f"{name}: duplicate episode timestamp",
    )
    _add_failure(failures, over_limit == 0, f"{name}: sequence exceeds maximum length")
    _add_failure(failures, config_match, f"{name}: configuration hash mismatch")
    return {
        "label_rows": int(len(labels)),
        "event_rows": int(len(events)),
        "event_episode_count": int(events["episode_id"].nunique()),
        "required_null_cells": required_nulls,
        "invalid_parsed_timestamp_rows": invalid_parsed_times,
        "sequence_index_violation_rows": index_violations,
        "time_reversal_rows": time_reversals,
        "duplicate_episode_timestamp_rows": duplicate_episode_times,
        "max_sequence_length": maximum_length,
        "episodes_over_maximum": over_limit,
        "config_hash_match": config_match,
        "dataset_validation_error": validation_error,
    }


def run_pretraining_gate(
    config: AkiAuditConfig, cohort_dir: Path, prepared_dir: Path
) -> tuple[dict[str, Any], int]:
    failures: list[str] = []
    actual_cohort_artifacts = {
        path.name for path in cohort_dir.iterdir() if path.is_file()
    }
    cohort_artifact_set_exact = actual_cohort_artifacts == COHORT_ARTIFACTS
    _add_failure(
        failures, cohort_artifact_set_exact, "cohort artifact set is not exact"
    )
    actual_artifacts = {path.name for path in prepared_dir.iterdir() if path.is_file()}
    artifact_set_exact = actual_artifacts == PREPARED_ARTIFACTS
    _add_failure(failures, artifact_set_exact, "prepared artifact set is not exact")
    cohort_private, cohort_insecure, cohort_symlinks = _private_tree(cohort_dir)
    prepared_private, prepared_insecure, prepared_symlinks = _private_tree(prepared_dir)
    _add_failure(
        failures,
        cohort_private,
        "cohort tree has group/world permissions or symbolic links",
    )
    _add_failure(
        failures,
        prepared_private,
        "prepared tree has group/world permissions or symbolic links",
    )

    protocol = pd.read_csv(prepared_dir / "protocol_metadata.csv")
    protocol_ok = (
        len(protocol) == 1
        and str(protocol.iloc[0]["config_hash"]) == config.config_hash
    )
    _add_failure(failures, protocol_ok, "prepared protocol hash mismatch")
    manifest = _read_json(prepared_dir / "run_manifest.json")
    manifest_ok = (
        manifest.get("stage") == "prepare"
        and manifest.get("config_sha256") == config.config_hash
    )
    _add_failure(failures, manifest_ok, "preparation manifest mismatch")

    cohort_flow = pd.read_csv(cohort_dir / "aki_cohort_flow.csv")
    unit_audit = pd.read_csv(cohort_dir / "aki_unit_audit.csv")
    cohort_hash_ok = (
        set(cohort_flow["config_hash"].dropna().astype(str)) == {config.config_hash}
        and set(unit_audit["config_hash"].dropna().astype(str)) == {config.config_hash}
    )
    _add_failure(failures, cohort_hash_ok, "cohort audit hash mismatch")
    flow_counts = pd.to_numeric(cohort_flow["count"], errors="coerce")
    flow_denominators = pd.to_numeric(cohort_flow["denominator"], errors="coerce")
    flow_fractions = pd.to_numeric(cohort_flow["fraction"], errors="coerce")
    cohort_flow_arithmetic_ok = bool(
        np.isfinite(flow_counts).all()
        and np.isfinite(flow_denominators).all()
        and np.isfinite(flow_fractions).all()
        and (flow_counts >= 0).all()
        and (flow_denominators > 0).all()
        and (flow_counts <= flow_denominators).all()
        and (flow_fractions >= 0).all()
        and (flow_fractions <= 1).all()
        and np.allclose(
            flow_fractions.to_numpy(dtype=float),
            (flow_counts / flow_denominators).to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
        )
    )
    unit_candidate = pd.to_numeric(unit_audit["candidate_rows"], errors="coerce")
    unit_rejected = pd.to_numeric(unit_audit["rejected_rows"], errors="coerce")
    unit_retained = pd.to_numeric(unit_audit["retained_rows"], errors="coerce")
    unit_arithmetic_ok = bool(
        np.isfinite(unit_candidate).all()
        and np.isfinite(unit_rejected).all()
        and np.isfinite(unit_retained).all()
        and (unit_candidate >= 0).all()
        and (unit_rejected >= 0).all()
        and (unit_retained >= 0).all()
        and unit_candidate.eq(unit_rejected + unit_retained).all()
    )
    retained_flow = cohort_flow[
        (cohort_flow["entity"] == "measurement")
        & (cohort_flow["stage"] == "final_cohort")
        & (cohort_flow["reason"] == "included")
    ]
    unit_flow_reconciles = bool(
        len(retained_flow) == 1
        and int(retained_flow.iloc[0]["count"]) == int(unit_retained.sum())
    )
    _add_failure(failures, cohort_flow_arithmetic_ok, "cohort flow arithmetic failure")
    _add_failure(failures, unit_arithmetic_ok, "unit audit arithmetic failure")
    _add_failure(failures, unit_flow_reconciles, "unit and cohort flows disagree")

    episode_flow = pd.read_csv(prepared_dir / "episode_flow.csv")
    censor_reasons = pd.read_csv(prepared_dir / "censor_reasons.csv")
    patient_splits = pd.read_parquet(prepared_dir / "patient_splits.parquet")
    duplicate_split_subjects = int(patient_splits["subject_id"].duplicated().sum())
    null_split_subjects = int(patient_splits["subject_id"].isna().sum())
    split_seed_ok = set(patient_splits["split_seed"]) == {
        int(config.splitting["split_seed"])
    }
    split_names_exact = set(patient_splits["split"]) == set(SPLITS)
    subject_sets = {
        split: set(patient_splits.loc[patient_splits["split"] == split, "subject_id"])
        for split in SPLITS
    }
    overlaps = {
        "train_validation": len(subject_sets["train"] & subject_sets["validation"]),
        "train_test": len(subject_sets["train"] & subject_sets["test"]),
        "validation_test": len(subject_sets["validation"] & subject_sets["test"]),
    }
    _add_failure(
        failures, duplicate_split_subjects == 0, "patient split map has duplicate subjects"
    )
    _add_failure(failures, null_split_subjects == 0, "patient split map has null subjects")
    _add_failure(failures, split_seed_ok, "patient split seed mismatch")
    _add_failure(failures, split_names_exact, "patient split names are not exact")
    _add_failure(
        failures, all(value == 0 for value in overlaps.values()), "patient split overlap"
    )
    split_map = patient_splits[["subject_id", "split"]].rename(
        columns={"split": "split_map"}
    )

    primary = pd.read_parquet(prepared_dir / "primary_labels.parquet")
    primary_matched = pd.read_parquet(prepared_dir / "primary_matched_labels.parquet")
    secondary = pd.read_parquet(prepared_dir / "secondary_labels.parquet")
    binary_labels = tuple(str(value) for value in config.metrics["binary"]["labels"])
    multiclass_labels = tuple(
        str(value) for value in config.metrics["multiclass"]["labels"]
    )
    class_report = {
        "primary_labels": _class_report(
            "primary_labels", primary, binary_labels, split_map, failures
        ),
        "primary_matched_labels": _class_report(
            "primary_matched_labels",
            primary_matched,
            binary_labels,
            split_map,
            failures,
        ),
        "secondary_labels": _class_report(
            "secondary_labels", secondary, multiclass_labels, split_map, failures
        ),
    }

    cross_task = pd.concat(
        [
            primary_matched[["subject_id", "split"]],
            secondary[["subject_id", "split"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    cross_task_mismatches = int(
        (cross_task.groupby("subject_id")["split"].nunique() > 1).sum()
    )
    _add_failure(
        failures,
        cross_task_mismatches == 0,
        "subjects have inconsistent splits across tasks",
    )

    assignments = pd.read_parquet(prepared_dir / "matching_assignments.parquet")
    balance = pd.read_parquet(prepared_dir / "matching_balance.parquet")
    estimability = pd.read_csv(prepared_dir / "matching_estimability.csv")
    estimable = (
        len(estimability) == 1 and _as_bool(estimability.iloc[0]["estimable"])
    )
    _add_failure(failures, estimable, "matching is marked non-estimable")
    pair_sizes = assignments.groupby(["split", "pair_id"], sort=False).size()
    expected_pair_labels = {
        str(config.matching["class_a"]),
        str(config.matching["class_b"]),
    }
    pair_label_sets = assignments.groupby(["split", "pair_id"], sort=False)[
        "label"
    ].agg(lambda values: set(values.astype(str)))
    pair_structure_ok = bool(
        not pair_sizes.empty
        and pair_sizes.eq(2).all()
        and pair_label_sets.map(lambda values: values == expected_pair_labels).all()
    )
    duplicate_assignment_episodes = int(assignments["episode_id"].duplicated().sum())
    duplicate_assignment_subjects = int(assignments["subject_id"].duplicated().sum())
    pair_counts = {
        split: int(
            assignments.loc[assignments["split"] == split, "pair_id"].nunique()
        )
        for split in SPLITS
    }
    minimum_pairs = {
        split: int(config.matching["minimum_pairs_per_split"][split])
        for split in SPLITS
    }
    pair_minimums_ok = all(
        pair_counts[split] >= minimum_pairs[split] for split in SPLITS
    )
    _add_failure(failures, pair_structure_ok, "matched-pair structure failure")
    _add_failure(
        failures,
        duplicate_assignment_episodes == 0,
        "matching assignments duplicate episodes",
    )
    _add_failure(
        failures,
        duplicate_assignment_subjects == 0,
        "matching assignments duplicate subjects",
    )
    _add_failure(failures, pair_minimums_ok, "matched-pair minimum not met")
    assignment_identity_columns = [
        "episode_id",
        "subject_id",
        "split",
        "label",
        "pair_id",
    ]
    assignment_reconciliation = assignments[assignment_identity_columns].merge(
        primary_matched[assignment_identity_columns],
        on=assignment_identity_columns,
        how="outer",
        indicator=True,
    )
    assignment_identity_mismatches = int(
        assignment_reconciliation["_merge"].ne("both").sum()
    )
    _add_failure(
        failures,
        assignment_identity_mismatches == 0,
        "matching assignments and matched labels differ",
    )

    feature_names = list(config.matching["coarsening"])
    thresholds = {
        feature: float(
            config.matching.get("maximum_absolute_smd_overrides", {}).get(
                feature, config.matching["maximum_absolute_smd"]
            )
        )
        for feature in feature_names
    }
    features = pd.read_parquet(
        prepared_dir / "primary_matching_features.parquet",
        columns=["episode_id", "subject_id", "label", "split", *feature_names],
    )
    selected = features.merge(
        assignments[["episode_id", "pair_id"]],
        on="episode_id",
        how="inner",
        validate="one_to_one",
    )
    recomputed_rows: list[dict[str, Any]] = []
    balance_reconciles = True
    all_finite = True
    all_within = True
    for split in SPLITS:
        before_split = features[features["split"] == split]
        after_split = selected[selected["split"] == split]
        for feature in feature_names:
            before_value = _smd(
                before_split,
                feature,
                str(config.matching["class_a"]),
                str(config.matching["class_b"]),
            )
            after_value = _smd(
                after_split,
                feature,
                str(config.matching["class_a"]),
                str(config.matching["class_b"]),
            )
            finite = math.isfinite(after_value)
            within = finite and abs(after_value) <= thresholds[feature]
            all_finite &= finite
            all_within &= within
            persisted = balance[
                (balance["split"] == split) & (balance["feature"] == feature)
            ]
            if len(persisted) != 1:
                balance_reconciles = False
            else:
                row = persisted.iloc[0]
                balance_reconciles &= bool(
                    math.isclose(
                        float(row["smd_before"]),
                        before_value,
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        float(row["smd_after"]),
                        after_value,
                        rel_tol=1e-10,
                        abs_tol=1e-12,
                    )
                    and math.isclose(
                        float(row["maximum_absolute_smd"]),
                        thresholds[feature],
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                    and _as_bool(row["passes_threshold"]) == within
                )
            recomputed_rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "smd_before": before_value,
                    "smd_after": after_value,
                    "maximum_absolute_smd": thresholds[feature],
                    "finite_after": finite,
                    "within_threshold": within,
                }
            )
    expected_balance_rows = len(SPLITS) * len(feature_names)
    _add_failure(
        failures,
        len(balance) == expected_balance_rows,
        "matching balance row count is incomplete",
    )
    _add_failure(failures, balance_reconciles, "matching balance does not reconcile")
    _add_failure(failures, all_finite, "a post-match SMD is non-finite")
    _add_failure(failures, all_within, "a post-match SMD exceeds its threshold")

    primary_events = pd.read_parquet(prepared_dir / "primary_matched_events.parquet")
    secondary_events = pd.read_parquet(prepared_dir / "secondary_events.parquet")
    max_length = int(config.representations["max_length"])
    event_report = {
        "primary_matched": _event_report(
            "primary_matched",
            primary_events,
            primary_matched,
            binary_labels,
            config.config_hash,
            max_length,
            failures,
        ),
        "secondary": _event_report(
            "secondary",
            secondary_events,
            secondary,
            multiclass_labels,
            config.config_hash,
            max_length,
            failures,
        ),
    }

    episodes = pd.read_parquet(
        prepared_dir / "episodes.parquet",
        columns=["episode_id", "status", "phenotype", "include_in_analysis"],
    )
    episode_audit = pd.read_parquet(
        prepared_dir / "episode_audit.parquet",
        columns=[
            "episode_id",
            "status",
            "phenotype",
            "include_in_analysis",
            "protocol_sha256",
            "protocol_json",
        ],
    )
    audit_duplicate_ids = int(episode_audit["episode_id"].duplicated().sum())
    audit_null_ids = int(episode_audit["episode_id"].isna().sum())
    episode_duplicate_ids = int(episodes["episode_id"].duplicated().sum())
    episode_null_ids = int(episodes["episode_id"].isna().sum())
    missing_audit = len(set(episodes["episode_id"]) - set(episode_audit["episode_id"]))
    protocol_pairs = episode_audit[["protocol_json", "protocol_sha256"]].drop_duplicates()
    digest_pairs = int(len(protocol_pairs))
    protocol_null_cells = int(
        episode_audit[["protocol_json", "protocol_sha256"]].isna().sum().sum()
    )
    protocol_hash_mismatches = 0
    protocol_json_parse_failures = 0
    expected_protocol_json = json.dumps(
        _Protocol.from_config(config).serializable(),
        sort_keys=True,
        separators=(",", ":"),
    )
    protocol_config_mismatches = 0
    for row in protocol_pairs.itertuples(index=False):
        try:
            json.loads(str(row.protocol_json))
        except (TypeError, ValueError):
            protocol_json_parse_failures += 1
        observed_digest = hashlib.sha256(str(row.protocol_json).encode("utf-8")).hexdigest()
        protocol_hash_mismatches += int(observed_digest != str(row.protocol_sha256))
        protocol_config_mismatches += int(str(row.protocol_json) != expected_protocol_json)
    expected_included = episode_audit["status"].eq("labeled") & episode_audit[
        "phenotype"
    ].isin({"transient", "persistent", "relapsing"})
    observed_included = episode_audit["include_in_analysis"].map(_as_bool)
    include_logic_mismatches = int(observed_included.ne(expected_included).sum())
    episode_reconciliation = episodes.merge(
        episode_audit[
            ["episode_id", "status", "phenotype", "include_in_analysis"]
        ],
        on="episode_id",
        how="left",
        suffixes=("_episode", "_audit"),
        validate="one_to_one",
    )
    episode_decision_mismatches = int(
        (
            episode_reconciliation["status_audit"].isna()
            | episode_reconciliation["status_episode"].ne(
                episode_reconciliation["status_audit"]
            )
            | episode_reconciliation["phenotype_episode"].ne(
                episode_reconciliation["phenotype_audit"]
            )
            | episode_reconciliation["include_in_analysis_episode"]
            .map(_as_bool)
            .ne(episode_reconciliation["include_in_analysis_audit"].map(_as_bool))
        ).sum()
    )
    _add_failure(failures, audit_duplicate_ids == 0, "episode audit duplicates")
    _add_failure(failures, audit_null_ids == 0, "episode audit null identifiers")
    _add_failure(failures, episode_duplicate_ids == 0, "episode table duplicates")
    _add_failure(failures, episode_null_ids == 0, "episode table null identifiers")
    _add_failure(failures, missing_audit == 0, "episodes are missing audit decisions")
    _add_failure(failures, digest_pairs == 1, "state-machine protocol digest is not unique")
    _add_failure(failures, protocol_null_cells == 0, "state-machine protocol fields are null")
    _add_failure(
        failures,
        protocol_hash_mismatches == 0,
        "state-machine protocol digest mismatch",
    )
    _add_failure(
        failures,
        protocol_json_parse_failures == 0,
        "state-machine protocol JSON is invalid",
    )
    _add_failure(
        failures,
        protocol_config_mismatches == 0,
        "state-machine protocol differs from frozen configuration",
    )
    _add_failure(
        failures, include_logic_mismatches == 0, "episode inclusion logic mismatch"
    )
    _add_failure(
        failures,
        episode_decision_mismatches == 0,
        "episode and audit decision tables disagree",
    )

    summary = manifest.get("summary", {})
    manifest_reconciles = bool(
        isinstance(summary, Mapping)
        and int(summary.get("episodes", -1)) == len(episodes)
        and int(summary.get("audited_decisions", -1)) == len(episode_audit)
        and int(summary.get("primary_matched_episodes", -1))
        == len(primary_matched)
        and int(summary.get("secondary_episodes", -1)) == len(secondary)
        and int(summary.get("subjects_split", -1))
        == patient_splits["subject_id"].nunique()
        and bool(summary.get("matching_estimable")) == estimable
    )
    _add_failure(failures, manifest_reconciles, "preparation manifest counts differ")

    report = {
        "gate": "pretraining",
        "overall_gate": "PASS" if not failures else "FAIL",
        "training_authorized": not failures,
        "config_hash": config.config_hash,
        "cohort_artifact_count": len(actual_cohort_artifacts),
        "expected_cohort_artifact_count": len(COHORT_ARTIFACTS),
        "cohort_artifact_set_exact": cohort_artifact_set_exact,
        "artifact_count": len(actual_artifacts),
        "expected_artifact_count": len(PREPARED_ARTIFACTS),
        "artifact_set_exact": artifact_set_exact,
        "provenance_ok": protocol_ok and manifest_ok and cohort_hash_ok,
        "manifest_reconciles": manifest_reconciles,
        "private_storage": {
            "cohort_private": cohort_private,
            "cohort_insecure_path_count": cohort_insecure,
            "cohort_symlink_count": cohort_symlinks,
            "prepared_private": prepared_private,
            "prepared_insecure_path_count": prepared_insecure,
            "prepared_symlink_count": prepared_symlinks,
        },
        "cohort_audit_integrity": {
            "flow_arithmetic_ok": cohort_flow_arithmetic_ok,
            "unit_arithmetic_ok": unit_arithmetic_ok,
            "unit_flow_reconciles": unit_flow_reconciles,
        },
        "cohort_flow": _records(cohort_flow),
        "unit_audit": _records(unit_audit),
        "episode_flow": _records(episode_flow),
        "censor_reasons": _records(censor_reasons),
        "class_report": class_report,
        "matching_report": {
            "estimable": estimable,
            "non_estimable_reasons_json": (
                estimability.iloc[0].get("non_estimable_reasons_json", "[]")
                if len(estimability) == 1
                else "unavailable"
            ),
            "pair_counts": pair_counts,
            "minimum_pairs": minimum_pairs,
            "pair_structure_ok": pair_structure_ok,
            "pair_minimums_ok": pair_minimums_ok,
            "assignment_duplicate_episode_rows": duplicate_assignment_episodes,
            "assignment_duplicate_subject_rows": duplicate_assignment_subjects,
            "assignment_identity_mismatch_rows": assignment_identity_mismatches,
            "persisted_balance_rows": int(len(balance)),
            "expected_balance_rows": expected_balance_rows,
            "persisted_balance_reconciles": balance_reconciles,
            "all_recomputed_smd_finite": all_finite,
            "all_recomputed_smd_within_threshold": all_within,
            "full_recomputed_balance": recomputed_rows,
        },
        "split_report": {
            "subject_counts": {
                split: int(len(subject_sets[split])) for split in SPLITS
            },
            "duplicate_subject_rows": duplicate_split_subjects,
            "subject_id_nulls": null_split_subjects,
            "split_seed_values_match": split_seed_ok,
            "split_names_exact": split_names_exact,
            "cross_task_split_mismatches": cross_task_mismatches,
            **{f"{name}_overlap": value for name, value in overlaps.items()},
        },
        "event_report": event_report,
        "episode_integrity": {
            "episode_rows": int(len(episodes)),
            "audit_rows": int(len(episode_audit)),
            "episode_id_duplicates": episode_duplicate_ids,
            "episode_id_nulls": episode_null_ids,
            "audit_id_duplicates": audit_duplicate_ids,
            "audit_id_nulls": audit_null_ids,
            "episode_ids_missing_audit": int(missing_audit),
            "state_machine_protocol_unique_digests": digest_pairs,
            "state_machine_protocol_null_cells": protocol_null_cells,
            "state_machine_protocol_hash_mismatches": protocol_hash_mismatches,
            "state_machine_protocol_json_parse_failures": protocol_json_parse_failures,
            "state_machine_protocol_config_mismatches": protocol_config_mismatches,
            "include_logic_mismatch_rows": include_logic_mismatches,
            "episode_decision_mismatch_rows": episode_decision_mismatches,
        },
        "failures": failures,
    }
    return report, 0 if not failures else 2


def run_posttraining_gate(
    config: AkiAuditConfig,
    config_path: Path,
    prepared_dir: Path,
    experiment_dir: Path,
    train_log: Path,
    gpu_telemetry: Path,
) -> tuple[dict[str, Any], int]:
    failures: list[str] = []
    run_root = experiment_dir.parent
    private_permissions, insecure_paths, symlink_paths = _private_tree(run_root)
    paths_share_run_root = bool(
        prepared_dir.parent == run_root
        and config_path.parent == run_root / "protocol"
        and train_log.parent == run_root / "logs"
        and gpu_telemetry.parent == run_root / "logs"
    )
    _add_failure(
        failures,
        private_permissions,
        "run tree has group/world permissions or symbolic links",
    )
    _add_failure(
        failures,
        paths_share_run_root,
        "post-training inputs do not share the private run root",
    )
    actual_artifacts = {path.name for path in experiment_dir.iterdir() if path.is_file()}
    artifact_set_exact = actual_artifacts == EXPERIMENT_ARTIFACTS
    _add_failure(failures, artifact_set_exact, "experiment artifact set is not exact")
    _add_failure(
        failures,
        not (experiment_dir / "predictions.csv").exists(),
        "predictions CSV must not exist",
    )

    prepared_protocol = pd.read_csv(prepared_dir / "protocol_metadata.csv")
    prepared_manifest = _read_json(prepared_dir / "run_manifest.json")
    prepared_provenance_ok = bool(
        len(prepared_protocol) == 1
        and str(prepared_protocol.iloc[0]["config_hash"]) == config.config_hash
        and prepared_manifest.get("stage") == "prepare"
        and prepared_manifest.get("config_sha256") == config.config_hash
    )
    _add_failure(
        failures, prepared_provenance_ok, "prepared protocol provenance mismatch"
    )

    manifest = _read_json(experiment_dir / "run_manifest.json")
    manifest_artifacts = manifest.get("artifacts", {})
    manifest_paths_ok = bool(
        isinstance(manifest_artifacts, Mapping)
        and Path(str(manifest_artifacts.get("prepared_directory", ""))).resolve()
        == prepared_dir.resolve()
        and Path(str(manifest_artifacts.get("experiment_directory", ""))).resolve()
        == experiment_dir.resolve()
    )
    manifest_ok = (
        manifest.get("stage") == "train"
        and manifest.get("config_sha256") == config.config_hash
        and manifest.get("config") == config.as_dict()
        and manifest_paths_ok
        and set(manifest.get("enabled_models", []))
        == {"logistic_regression", "xgboost", "lstm", "transformer"}
    )
    _add_failure(failures, manifest_ok, "training manifest mismatch")

    seed_metrics = pd.read_parquet(experiment_dir / "seed_metrics.parquet")
    aggregate_metrics = pd.read_parquet(experiment_dir / "aggregate_metrics.parquet")
    contrasts = pd.read_parquet(
        experiment_dir / "ordered_control_contrasts.parquet"
    )
    aggregate_contrasts = pd.read_parquet(
        experiment_dir / "aggregate_ordered_control_contrasts.parquet"
    )
    task_estimability = pd.read_parquet(
        experiment_dir / "task_estimability.parquet"
    )
    condition_estimability = pd.read_parquet(
        experiment_dir / "condition_estimability.parquet"
    )
    schema_checks = {
        "seed_metrics": list(seed_metrics.columns) == SEED_METRIC_COLUMNS,
        "predictions": (
            pq.ParquetFile(experiment_dir / "predictions.parquet").schema_arrow.names
            == PREDICTION_COLUMNS
        ),
        "ordered_control_contrasts": list(contrasts.columns) == CONTRAST_COLUMNS,
        "task_estimability": (
            list(task_estimability.columns) == TASK_ESTIMABILITY_COLUMNS
        ),
        "condition_estimability": (
            list(condition_estimability.columns) == CONDITION_ESTIMABILITY_COLUMNS
        ),
    }
    _add_failure(
        failures, all(schema_checks.values()), "an experiment table schema is not exact"
    )
    csv_parquet_checks = {
        "seed_metrics": _csv_matches_parquet(
            experiment_dir / "seed_metrics.csv", seed_metrics
        ),
        "aggregate_metrics": _csv_matches_parquet(
            experiment_dir / "aggregate_metrics.csv", aggregate_metrics
        ),
        "ordered_control_contrasts": _csv_matches_parquet(
            experiment_dir / "ordered_control_contrasts.csv", contrasts
        ),
        "aggregate_ordered_control_contrasts": _csv_matches_parquet(
            experiment_dir / "aggregate_ordered_control_contrasts.csv",
            aggregate_contrasts,
        ),
        "task_estimability": _csv_matches_parquet(
            experiment_dir / "task_estimability.csv", task_estimability
        ),
        "condition_estimability": _csv_matches_parquet(
            experiment_dir / "condition_estimability.csv", condition_estimability
        ),
    }
    _add_failure(
        failures,
        all(csv_parquet_checks.values()),
        "an aggregate CSV differs from its Parquet table",
    )

    n_seeds = len(config.models["seeds"])
    expected_model_rows = {
        "logistic_regression": 7 * n_seeds,
        "xgboost": n_seeds,
        "lstm": 4 * n_seeds,
        "transformer": 4 * n_seeds,
        "state_machine_oracle": 1,
    }
    expected_seed_rows = 2 * sum(expected_model_rows.values())
    expected_aggregate_rows = 2 * 17
    expected_contrast_rows = 2 * 2 * n_seeds * 6
    expected_aggregate_contrast_rows = 2 * 2 * 6
    _add_failure(
        failures, len(seed_metrics) == expected_seed_rows, "seed metric row count"
    )
    _add_failure(
        failures,
        len(aggregate_metrics) == expected_aggregate_rows,
        "aggregate metric row count",
    )
    _add_failure(
        failures, len(contrasts) == expected_contrast_rows, "contrast row count"
    )
    _add_failure(
        failures,
        len(aggregate_contrasts) == expected_aggregate_contrast_rows,
        "aggregate contrast row count",
    )
    run_key_columns = [
        "task",
        "model",
        "condition",
        "feature_set",
        "model_seed",
        "permutation_seed",
    ]
    expected_run_keys: set[tuple[Any, ...]] = set()
    configured_seed_pairs = list(
        zip(
            (int(value) for value in config.models["seeds"]),
            (int(value) for value in config.representations["permutation_seeds"]),
        )
    )
    logistic_features = (
        "invariant_summary",
        "count_only",
        "first_value",
        "last_value",
        "first_to_last_change",
        "linear_slope",
        "position_combined",
    )
    sequence_conditions = (
        ("ordered_train_ordered_test", "ordered"),
        ("ordered_train_shuffled_test", "shuffled"),
        ("ordered_train_reversed_test", "reversed"),
        ("shuffled_train_shuffled_test", "shuffled"),
    )
    for task in ("binary", "multiclass"):
        for model_seed, permutation_seed in configured_seed_pairs:
            for feature_set in logistic_features:
                expected_run_keys.add(
                    (
                        task,
                        "logistic_regression",
                        "trained_and_tested_same_representation",
                        feature_set,
                        model_seed,
                        permutation_seed,
                    )
                )
            expected_run_keys.add(
                (
                    task,
                    "xgboost",
                    "trained_and_tested_same_representation",
                    "invariant_summary",
                    model_seed,
                    permutation_seed,
                )
            )
            for model in ("lstm", "transformer"):
                for condition, feature_set in sequence_conditions:
                    expected_run_keys.add(
                        (
                            task,
                            model,
                            condition,
                            feature_set,
                            model_seed,
                            permutation_seed,
                        )
                    )
        expected_run_keys.add(
            (
                task,
                "state_machine_oracle",
                "oracle_ceiling",
                "configured_trajectory_state_machine",
                -1,
                -1,
            )
        )
    actual_run_keys = set(
        seed_metrics[run_key_columns].itertuples(index=False, name=None)
    )
    duplicate_metric_run_keys = int(seed_metrics.duplicated(run_key_columns).sum())
    model_matrix_ok = bool(
        actual_run_keys == expected_run_keys and duplicate_metric_run_keys == 0
    )
    _add_failure(failures, model_matrix_ok, "model/task matrix is incomplete")

    configured_pairs = set(configured_seed_pairs)
    ordinary = seed_metrics[seed_metrics["model"] != "state_machine_oracle"]
    oracle = seed_metrics[seed_metrics["model"] == "state_machine_oracle"]
    ordinary_pairs = set(
        zip(
            ordinary["model_seed"].astype(int),
            ordinary["permutation_seed"].astype(int),
        )
    )
    oracle_pairs = set(
        zip(oracle["model_seed"].astype(int), oracle["permutation_seed"].astype(int))
    )
    seed_pairs_ok = ordinary_pairs == configured_pairs and oracle_pairs == {(-1, -1)}
    _add_failure(failures, seed_pairs_ok, "seed pairing mismatch")

    recomputed_aggregate_metrics = _aggregate(seed_metrics, config)
    recomputed_aggregate_contrasts = _aggregate_contrasts(contrasts, config)
    aggregate_metrics_reconcile = _frames_equivalent(
        aggregate_metrics,
        recomputed_aggregate_metrics,
        ["task", "model", "condition", "feature_set"],
    )
    aggregate_contrasts_reconcile = _frames_equivalent(
        aggregate_contrasts,
        recomputed_aggregate_contrasts,
        ["task", "model", "metric", "left_condition", "right_condition"],
    )
    _add_failure(
        failures, aggregate_metrics_reconcile, "aggregate metrics do not recompute"
    )
    _add_failure(
        failures,
        aggregate_contrasts_reconcile,
        "aggregate contrasts do not recompute",
    )

    test_counts = {
        "binary": int(
            (
                pd.read_parquet(
                    prepared_dir / "primary_matched_labels.parquet",
                    columns=["split"],
                )["split"]
                == "test"
            ).sum()
        ),
        "multiclass": int(
            (
                pd.read_parquet(
                    prepared_dir / "secondary_labels.parquet", columns=["split"]
                )["split"]
                == "test"
            ).sum()
        ),
    }
    n_samples_ok = all(
        set(seed_metrics.loc[seed_metrics["task"] == task, "n_samples"].astype(int))
        == {count}
        for task, count in test_counts.items()
    )
    _add_failure(failures, n_samples_ok, "metric sample counts mismatch")

    metric_columns_by_task = {
        "binary": BINARY_METRIC_COLUMNS,
        "multiclass": MULTICLASS_METRIC_COLUMNS,
    }
    seed_metrics_finite = True
    seed_metrics_bounded = True
    seed_metrics_irrelevant_null = True
    all_metric_columns = set(BINARY_METRIC_COLUMNS + MULTICLASS_METRIC_COLUMNS)
    for task, relevant_columns in metric_columns_by_task.items():
        task_rows = seed_metrics[seed_metrics["task"] == task]
        relevant = task_rows[relevant_columns].to_numpy(dtype=float)
        seed_metrics_finite &= bool(np.isfinite(relevant).all())
        seed_metrics_bounded &= bool(((relevant >= 0) & (relevant <= 1)).all())
        irrelevant_columns = sorted(all_metric_columns.difference(relevant_columns))
        seed_metrics_irrelevant_null &= bool(
            task_rows[irrelevant_columns].isna().all().all()
        )
    _add_failure(failures, seed_metrics_finite, "seed metrics are non-finite")
    _add_failure(failures, seed_metrics_bounded, "seed metrics are outside [0,1]")
    _add_failure(
        failures,
        seed_metrics_irrelevant_null,
        "task-inapplicable seed metrics are not null",
    )

    task_estimable = bool(
        len(task_estimability) == 2
        and set(task_estimability["task"]) == {"binary", "multiclass"}
        and task_estimability["estimable"].map(_as_bool).all()
        and task_estimability["non_estimable_reasons"].map(_empty_reason).all()
    )
    condition_estimable = bool(
        len(condition_estimability) == 2
        and set(condition_estimability["task"]) == {"binary", "multiclass"}
        and set(condition_estimability["model"]) == {"state_machine_oracle"}
        and set(condition_estimability["condition"]) == {"oracle_ceiling"}
        and condition_estimability["estimable"].map(_as_bool).all()
        and condition_estimability["reason"].fillna("").astype(str).eq("").all()
    )
    _add_failure(failures, task_estimable, "a task is non-estimable")
    _add_failure(failures, condition_estimable, "oracle condition is non-estimable")

    bootstrap = config.metrics["bootstrap"]
    contrast_arithmetic = bool(
        np.allclose(
            contrasts["ordered_minus_control"].to_numpy(dtype=float),
            (
                contrasts["left_value"].to_numpy(dtype=float)
                - contrasts["right_value"].to_numpy(dtype=float)
            ),
            rtol=1e-12,
            atol=1e-12,
        )
        and np.allclose(
            contrasts["ordered_minus_control"].to_numpy(dtype=float),
            contrasts["bootstrap_estimate"].to_numpy(dtype=float),
            rtol=1e-12,
            atol=1e-12,
        )
    )
    bootstrap_ok = bool(
        (
            contrasts["bootstrap_resamples_requested"]
            == int(bootstrap["n_resamples"])
        ).all()
        and (contrasts["bootstrap_resamples_valid"] > 0).all()
        and np.allclose(
            contrasts["confidence_level"].to_numpy(dtype=float),
            float(bootstrap["confidence_level"]),
        )
        and (
            contrasts["bootstrap_interval_method"]
            == str(bootstrap["interval_method"])
        ).all()
        and contrasts["bootstrap_clustered_by_subject"].map(_as_bool).all()
        and (
            contrasts["bootstrap_resamples_valid"]
            <= contrasts["bootstrap_resamples_requested"]
        ).all()
        and (
            contrasts["confidence_lower"] <= contrasts["confidence_upper"]
        ).all()
        and np.isfinite(
            contrasts[
                [
                    "left_value",
                    "right_value",
                    "ordered_minus_control",
                    "bootstrap_estimate",
                    "confidence_lower",
                    "confidence_upper",
                    "confidence_level",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    contrast_key_columns = [
        "task",
        "model",
        "model_seed",
        "metric",
        "left_condition",
        "right_condition",
    ]
    duplicate_contrast_keys = int(contrasts.duplicated(contrast_key_columns).sum())
    expected_core_contrasts = _contrasts(seed_metrics)
    contrast_core_reconciles = bool(
        duplicate_contrast_keys == 0
        and _frames_equivalent(
            contrasts[list(expected_core_contrasts.columns)],
            expected_core_contrasts,
            contrast_key_columns,
        )
    )
    _add_failure(failures, contrast_arithmetic, "contrast arithmetic mismatch")
    _add_failure(failures, bootstrap_ok, "bootstrap audit mismatch")
    _add_failure(
        failures, contrast_core_reconciles, "seed-level contrasts do not recompute"
    )

    predictions_path = experiment_dir / "predictions.parquet"
    prediction_file = pq.ParquetFile(predictions_path)
    prediction_rows = int(prediction_file.metadata.num_rows)
    prediction_schema = prediction_file.schema_arrow.names
    expected_prediction_rows = int(seed_metrics["n_samples"].sum())
    predictions_metadata_ok = (
        prediction_rows == expected_prediction_rows
        and prediction_schema == PREDICTION_COLUMNS
    )
    _add_failure(
        failures, predictions_metadata_ok, "prediction Parquet metadata mismatch"
    )
    predictions = pd.read_parquet(predictions_path)
    required_prediction_columns = [
        "task",
        "model",
        "condition",
        "feature_set",
        "model_seed",
        "permutation_seed",
        "episode_id",
        "subject_id",
        "hadm_id",
        "true_label",
    ]
    prediction_required_null_cells = int(
        predictions[required_prediction_columns].isna().sum().sum()
    )
    duplicate_prediction_keys = int(
        predictions.duplicated([*run_key_columns, "episode_id"]).sum()
    )
    prediction_counts = (
        predictions.groupby(run_key_columns, dropna=False)
        .size()
        .rename("prediction_count")
        .reset_index()
    )
    expected_counts = seed_metrics[[*run_key_columns, "n_samples"]]
    count_reconciliation = expected_counts.merge(
        prediction_counts,
        on=run_key_columns,
        how="outer",
        indicator=True,
    )
    prediction_run_count_mismatches = int(
        (
            count_reconciliation["_merge"].ne("both")
            | count_reconciliation["n_samples"].ne(
                count_reconciliation["prediction_count"]
            )
        ).sum()
    )
    truth_columns = ["episode_id", "subject_id", "hadm_id", "label", "split"]
    binary_truth = pd.read_parquet(
        prepared_dir / "primary_matched_labels.parquet", columns=truth_columns
    )
    multiclass_truth = pd.read_parquet(
        prepared_dir / "secondary_labels.parquet", columns=truth_columns
    )
    binary_truth = binary_truth[binary_truth["split"] == "test"].assign(task="binary")
    multiclass_truth = multiclass_truth[multiclass_truth["split"] == "test"].assign(
        task="multiclass"
    )
    test_truth = pd.concat([binary_truth, multiclass_truth], ignore_index=True)
    aligned = predictions.merge(
        test_truth.drop(columns="split"),
        on=["task", "episode_id"],
        how="left",
        suffixes=("_prediction", "_prepared"),
        validate="many_to_one",
    )
    prediction_rows_missing_truth = int(aligned["label"].isna().sum())
    prediction_identity_ok = (
        aligned["subject_id_prediction"]
        .eq(aligned["subject_id_prepared"])
        .fillna(False)
        & aligned["hadm_id_prediction"]
        .eq(aligned["hadm_id_prepared"])
        .fillna(False)
        & aligned["true_label"].eq(aligned["label"]).fillna(False)
    )
    prediction_identity_mismatches = int((~prediction_identity_ok).sum())
    probability_nonfinite_rows = 0
    probability_out_of_bounds_rows = 0
    probability_sum_mismatch_rows = 0
    label_protocol_mismatch_rows = 0
    irrelevant_probability_nonnull_rows = 0
    task_probability_columns = {
        "binary": ["probability_transient", "probability_persistent"],
        "multiclass": [
            "probability_transient",
            "probability_persistent",
            "probability_relapsing",
        ],
    }
    task_labels = {
        "binary": set(str(value) for value in config.metrics["binary"]["labels"]),
        "multiclass": set(
            str(value) for value in config.metrics["multiclass"]["labels"]
        ),
    }
    metric_reconciliation_mismatched_runs = 0
    metric_rows_by_key = seed_metrics.set_index(run_key_columns)
    oracle_one_hot_violation_rows = 0
    for task, probability_columns in task_probability_columns.items():
        task_predictions = predictions[predictions["task"] == task]
        matrix = task_predictions[probability_columns].to_numpy(dtype=float)
        probability_nonfinite_rows += int((~np.isfinite(matrix)).any(axis=1).sum())
        probability_out_of_bounds_rows += int(
            ((matrix < -1e-6) | (matrix > 1.0 + 1e-6)).any(axis=1).sum()
        )
        probability_sum_mismatch_rows += int(
            (~np.isclose(matrix.sum(axis=1), 1.0, rtol=0, atol=1e-6)).sum()
        )
        label_protocol_mismatch_rows += int(
            (~task_predictions["true_label"].astype(str).isin(task_labels[task])).sum()
        )
        labels_in_order = tuple(
            str(value) for value in config.metrics[task]["labels"]
        )
        for key, run_predictions in task_predictions.groupby(
            run_key_columns, sort=False, dropna=False
        ):
            try:
                persisted_metrics = metric_rows_by_key.loc[key]
                recalculated = _evaluate(
                    task,
                    run_predictions["true_label"].to_numpy(),
                    run_predictions[probability_columns].to_numpy(dtype=float),
                    labels_in_order,
                    config,
                )
            except Exception:
                metric_reconciliation_mismatched_runs += 1
                continue
            relevant_metrics = metric_columns_by_task[task]
            metrics_match = int(persisted_metrics["n_samples"]) == int(
                recalculated["n_samples"]
            ) and all(
                math.isclose(
                    float(persisted_metrics[column]),
                    float(recalculated[column]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for column in relevant_metrics
            )
            metric_reconciliation_mismatched_runs += int(not metrics_match)
        oracle_rows = task_predictions[
            task_predictions["model"] == "state_machine_oracle"
        ]
        if not oracle_rows.empty:
            oracle_matrix = oracle_rows[probability_columns].to_numpy(dtype=float)
            expected_oracle = np.zeros_like(oracle_matrix)
            label_positions = {label: index for index, label in enumerate(labels_in_order)}
            oracle_labels_valid = oracle_rows["true_label"].astype(str).isin(label_positions)
            for row_number, label in enumerate(
                oracle_rows["true_label"].astype(str).tolist()
            ):
                if label in label_positions:
                    expected_oracle[row_number, label_positions[label]] = 1.0
            oracle_one_hot_violation_rows += int((~oracle_labels_valid).sum())
            oracle_one_hot_violation_rows += int(
                (~np.isclose(oracle_matrix, expected_oracle, rtol=0, atol=0))
                .any(axis=1)
                .sum()
            )
    irrelevant_probability_nonnull_rows = int(
        predictions.loc[
            predictions["task"] == "binary", "probability_relapsing"
        ].notna().sum()
    )
    prediction_integrity_ok = all(
        value == 0
        for value in (
            duplicate_prediction_keys,
            prediction_required_null_cells,
            prediction_run_count_mismatches,
            prediction_rows_missing_truth,
            prediction_identity_mismatches,
            probability_nonfinite_rows,
            probability_out_of_bounds_rows,
            probability_sum_mismatch_rows,
            label_protocol_mismatch_rows,
            irrelevant_probability_nonnull_rows,
            metric_reconciliation_mismatched_runs,
            oracle_one_hot_violation_rows,
        )
    )
    _add_failure(failures, prediction_integrity_ok, "private prediction integrity failure")

    restricted_columns = {"episode_id", "subject_id", "hadm_id"}
    safe_aggregate_files = [
        experiment_dir / "seed_metrics.csv",
        experiment_dir / "aggregate_metrics.csv",
        experiment_dir / "ordered_control_contrasts.csv",
        experiment_dir / "aggregate_ordered_control_contrasts.csv",
        experiment_dir / "task_estimability.csv",
        experiment_dir / "condition_estimability.csv",
    ]
    aggregate_csvs_safe = True
    for path in safe_aggregate_files:
        columns = set(pd.read_csv(path, nrows=0).columns)
        aggregate_csvs_safe &= not columns.intersection(restricted_columns)
        aggregate_csvs_safe &= not any(
            column.endswith("_time") or "timestamp" in column for column in columns
        )
    _add_failure(
        failures, aggregate_csvs_safe, "aggregate CSV contains a restricted column"
    )

    marker_lines = train_log.read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [line for line in marker_lines if line.startswith("AKI_SEQUENCE_DEVICE ")]
    completes = [
        line for line in marker_lines if line.startswith("AKI_SEQUENCE_COMPLETE ")
    ]
    expected_neural_fits = 2 * n_seeds * 2 * 2
    start_pattern = re.compile(
        r"^AKI_SEQUENCE_DEVICE model=(lstm|transformer) seed=(-?\d+) "
        r"device=(\S+) accelerator=(.+)$",
        flags=re.IGNORECASE,
    )
    complete_pattern = re.compile(
        r"^AKI_SEQUENCE_COMPLETE model=(lstm|transformer) seed=(-?\d+) "
        r"device=(\S+) accelerator=(.+?) best_epoch=(\d+)$",
        flags=re.IGNORECASE,
    )
    start_counter: Counter[tuple[str, int]] = Counter()
    complete_counter: Counter[tuple[str, int]] = Counter()
    marker_parse_failures = 0
    marker_accelerators: set[str] = set()
    marker_devices: set[str] = set()
    marker_epoch_violations = 0
    for line in starts:
        match = start_pattern.fullmatch(line)
        if match is None:
            marker_parse_failures += 1
        else:
            start_counter[(match.group(1).lower(), int(match.group(2)))] += 1
            marker_devices.add(match.group(3))
            marker_accelerators.add(match.group(4).strip())
    for line in completes:
        match = complete_pattern.fullmatch(line)
        if match is None:
            marker_parse_failures += 1
        else:
            complete_counter[(match.group(1).lower(), int(match.group(2)))] += 1
            marker_devices.add(match.group(3))
            marker_accelerators.add(match.group(4).strip())
            best_epoch = int(match.group(5))
            marker_epoch_violations += int(
                best_epoch < 1
                or best_epoch > int(config.models["sequence_training"]["max_epochs"])
            )
    expected_marker_counter = Counter(
        {
            (model, int(seed)): 4
            for model in ("lstm", "transformer")
            for seed in config.models["seeds"]
        }
    )
    markers_ok = bool(
        len(starts) == expected_neural_fits
        and len(completes) == expected_neural_fits
        and marker_parse_failures == 0
        and marker_epoch_violations == 0
        and start_counter == expected_marker_counter
        and complete_counter == expected_marker_counter
        and marker_devices == {"cuda:0"}
        and len(marker_accelerators) == 1
        and all(re.search(r"H200", name, re.IGNORECASE) for name in marker_accelerators)
    )
    _add_failure(failures, markers_ok, "H200 sequence-device markers are incomplete")

    telemetry = pd.read_csv(gpu_telemetry)
    telemetry_schema_ok = list(telemetry.columns) == [
        "uuid",
        "name",
        "utilization_gpu_percent",
        "memory_used_mib",
    ]
    telemetry_names = (
        set(telemetry["name"].astype(str).str.strip())
        if telemetry_schema_ok
        else set()
    )
    telemetry_uuids = (
        set(telemetry["uuid"].astype(str).str.strip())
        if telemetry_schema_ok
        else set()
    )
    telemetry_names_ok = bool(
        telemetry_schema_ok
        and not telemetry.empty
        and len(telemetry_names) == 1
        and len(telemetry_uuids) == 1
        and telemetry["name"].astype(str).str.contains("H200", case=False).all()
        and telemetry_names == marker_accelerators
    )
    maximum_gpu_utilization = float(
        pd.to_numeric(
            telemetry.get("utilization_gpu_percent", pd.Series(dtype=float)),
            errors="coerce",
        ).max()
    )
    maximum_gpu_memory_mib = float(
        pd.to_numeric(
            telemetry.get("memory_used_mib", pd.Series(dtype=float)), errors="coerce"
        ).max()
    )
    telemetry_activity_ok = bool(
        telemetry_names_ok
        and maximum_gpu_utilization > 0
        and maximum_gpu_memory_mib > 0
    )
    _add_failure(failures, telemetry_activity_ok, "GPU telemetry shows no H200 activity")

    summary = manifest.get("summary", {})
    manifest_reconciles = bool(
        isinstance(summary, Mapping)
        and int(summary.get("seed_metric_rows", -1)) == len(seed_metrics)
        and int(summary.get("aggregate_metric_rows", -1)) == len(aggregate_metrics)
        and int(summary.get("prediction_rows", -1)) == prediction_rows
        and int(summary.get("contrast_rows", -1)) == len(contrasts)
        and int(summary.get("aggregate_contrast_rows", -1))
        == len(aggregate_contrasts)
        and _records_match_manifest(
            task_estimability, summary.get("task_estimability")
        )
        and _records_match_manifest(
            condition_estimability, summary.get("condition_estimability")
        )
    )
    _add_failure(failures, manifest_reconciles, "training manifest counts differ")

    report = {
        "gate": "posttraining",
        "overall_gate": "PASS" if not failures else "FAIL",
        "config_hash": config.config_hash,
        "artifact_count": len(actual_artifacts),
        "expected_artifact_count": len(EXPERIMENT_ARTIFACTS),
        "artifact_set_exact": artifact_set_exact,
        "prepared_provenance_ok": prepared_provenance_ok,
        "manifest_ok": manifest_ok,
        "manifest_paths_ok": manifest_paths_ok,
        "manifest_reconciles": manifest_reconciles,
        "schema_checks": schema_checks,
        "csv_parquet_checks": csv_parquet_checks,
        "seed_metric_rows": int(len(seed_metrics)),
        "aggregate_metric_rows": int(len(aggregate_metrics)),
        "prediction_rows_from_footer": prediction_rows,
        "expected_prediction_rows": expected_prediction_rows,
        "contrast_rows": int(len(contrasts)),
        "aggregate_contrast_rows": int(len(aggregate_contrasts)),
        "model_matrix_ok": model_matrix_ok,
        "duplicate_metric_run_keys": duplicate_metric_run_keys,
        "seed_pairs_ok": seed_pairs_ok,
        "aggregate_metrics_reconcile": aggregate_metrics_reconcile,
        "aggregate_contrasts_reconcile": aggregate_contrasts_reconcile,
        "test_counts": test_counts,
        "n_samples_ok": n_samples_ok,
        "seed_metrics_finite": seed_metrics_finite,
        "seed_metrics_bounded": seed_metrics_bounded,
        "seed_metrics_irrelevant_null": seed_metrics_irrelevant_null,
        "task_estimability_ok": task_estimable,
        "condition_estimability_ok": condition_estimable,
        "contrast_arithmetic_ok": contrast_arithmetic,
        "duplicate_contrast_keys": duplicate_contrast_keys,
        "contrast_core_reconciles": contrast_core_reconciles,
        "bootstrap_ok": bootstrap_ok,
        "prediction_metadata_ok": predictions_metadata_ok,
        "prediction_integrity_checked_privately": True,
        "patient_level_values_emitted": False,
        "prediction_integrity_ok": prediction_integrity_ok,
        "prediction_violation_counts": {
            "duplicate_prediction_keys": duplicate_prediction_keys,
            "required_identity_or_run_null_cells": prediction_required_null_cells,
            "run_count_mismatches": prediction_run_count_mismatches,
            "rows_missing_prepared_truth": prediction_rows_missing_truth,
            "identity_or_label_mismatches": prediction_identity_mismatches,
            "nonfinite_probability_rows": probability_nonfinite_rows,
            "out_of_bounds_probability_rows": probability_out_of_bounds_rows,
            "probability_sum_mismatch_rows": probability_sum_mismatch_rows,
            "protocol_label_mismatch_rows": label_protocol_mismatch_rows,
            "irrelevant_probability_nonnull_rows": (
                irrelevant_probability_nonnull_rows
            ),
            "metric_reconciliation_mismatched_runs": (
                metric_reconciliation_mismatched_runs
            ),
            "oracle_one_hot_violation_rows": oracle_one_hot_violation_rows,
        },
        "aggregate_csvs_safe": aggregate_csvs_safe,
        "sequence_device_start_markers": len(starts),
        "sequence_device_complete_markers": len(completes),
        "sequence_marker_parse_failures": marker_parse_failures,
        "sequence_marker_epoch_violations": marker_epoch_violations,
        "sequence_marker_per_model_seed_counts_ok": bool(
            start_counter == expected_marker_counter
            and complete_counter == expected_marker_counter
        ),
        "sequence_h200_markers_ok": markers_ok,
        "gpu_telemetry_rows": int(len(telemetry)),
        "gpu_telemetry_schema_ok": telemetry_schema_ok,
        "gpu_telemetry_unique_uuid_count": len(telemetry_uuids),
        "maximum_gpu_utilization_percent": maximum_gpu_utilization,
        "maximum_gpu_memory_used_mib": maximum_gpu_memory_mib,
        "gpu_telemetry_activity_ok": telemetry_activity_ok,
        "private_permissions": private_permissions,
        "insecure_path_count": insecure_paths,
        "symlink_path_count": symlink_paths,
        "failures": failures,
    }
    return report, 0 if not failures else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("pretrain")
    pre.add_argument("--config", required=True)
    pre.add_argument("--cohort-dir", required=True)
    pre.add_argument("--prepared-dir", required=True)
    post = subparsers.add_parser("posttrain")
    post.add_argument("--config", required=True)
    post.add_argument("--prepared-dir", required=True)
    post.add_argument("--experiment-dir", required=True)
    post.add_argument("--train-log", required=True)
    post.add_argument("--gpu-telemetry", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config_path = Path(args.config).expanduser().absolute()
    config = load_aki_config(config_path)
    if args.command == "pretrain":
        report, status = run_pretraining_gate(
            config,
            Path(args.cohort_dir).expanduser().absolute(),
            Path(args.prepared_dir).expanduser().absolute(),
        )
    else:
        report, status = run_posttraining_gate(
            config,
            config_path,
            Path(args.prepared_dir).expanduser().absolute(),
            Path(args.experiment_dir).expanduser().absolute(),
            Path(args.train_log).expanduser().absolute(),
            Path(args.gpu_telemetry).expanduser().absolute(),
        )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
