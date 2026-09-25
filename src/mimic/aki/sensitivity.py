"""Secondary peak-to-recovery timing sensitivity for AKI episodes.

This analysis is intentionally separate from the transient/persistent/relapsing
state machine.  It supports a Heung-style fast/intermediate/no-recovery timing
table without redefining the primary phenotype or validating relapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd


SENSITIVITY_COLUMNS = [
    "episode_id",
    "subject_id",
    "hadm_id",
    "primary_phenotype",
    "recovery_timing_category",
    "sensitivity_status",
    "sensitivity_reason_code",
    "reference_time",
    "baseline_value",
    "recovery_threshold",
    "sensitivity_recovery_time",
    "hours_from_reference_to_recovery",
    "intermediate_horizon_end",
    "last_observation_time",
    "observation_count",
    "maximum_gap_hours",
]


@dataclass(frozen=True)
class _SensitivityProtocol:
    enabled: bool
    reference_time_column: str
    baseline_column: str
    threshold_above_baseline_mg_dl: float
    value_comparison: str
    fast_max_hours: float
    intermediate_max_hours: float
    time_boundary: str
    sustained_hours: float
    max_gap_hours: float
    endpoint_tolerance_hours: float
    minimum_measurements: int
    duplicate_policy: str


def _section(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        episodes = config.get("episodes", config)
    else:
        episodes = getattr(config, "episodes", None)
    if not isinstance(episodes, Mapping):
        raise ValueError("configuration section 'episodes' is required")
    value = episodes.get("recovery_timing_sensitivity")
    if not isinstance(value, Mapping):
        raise ValueError("episodes.recovery_timing_sensitivity must be a mapping")
    return value


def _protocol(config: Any) -> _SensitivityProtocol:
    raw = _section(config)
    required = {
        "enabled",
        "reference_time_column",
        "baseline_column",
        "threshold_above_baseline_mg_dl",
        "value_comparison",
        "fast_max_hours",
        "intermediate_max_hours",
        "time_boundary",
        "sustained_hours",
        "max_gap_hours",
        "endpoint_tolerance_hours",
        "minimum_measurements",
        "duplicate_policy",
    }
    missing = required.difference(raw)
    if missing:
        raise ValueError("episodes.recovery_timing_sensitivity is missing: " f"{sorted(missing)}")
    if not isinstance(raw["enabled"], bool):
        raise ValueError("recovery_timing_sensitivity.enabled must be boolean")
    for key in ("reference_time_column", "baseline_column"):
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"recovery_timing_sensitivity.{key} must be a string")
    numeric = {}
    for key in (
        "threshold_above_baseline_mg_dl",
        "fast_max_hours",
        "intermediate_max_hours",
        "sustained_hours",
        "max_gap_hours",
        "endpoint_tolerance_hours",
    ):
        value = raw[key]
        if isinstance(value, bool):
            raise ValueError(f"recovery_timing_sensitivity.{key} must be numeric")
        numeric[key] = float(value)
        if not np.isfinite(numeric[key]) or numeric[key] < 0:
            raise ValueError(f"recovery_timing_sensitivity.{key} must be finite and >= 0")
    if numeric["fast_max_hours"] <= 0:
        raise ValueError("recovery_timing_sensitivity.fast_max_hours must be > 0")
    if numeric["intermediate_max_hours"] <= numeric["fast_max_hours"]:
        raise ValueError(
            "recovery_timing_sensitivity.intermediate_max_hours must exceed fast_max_hours"
        )
    if numeric["max_gap_hours"] <= 0:
        raise ValueError("recovery_timing_sensitivity.max_gap_hours must be > 0")
    minimum = raw["minimum_measurements"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("recovery_timing_sensitivity.minimum_measurements must be >= 1")
    if raw["value_comparison"] not in {"lt", "le"}:
        raise ValueError("recovery_timing_sensitivity.value_comparison must be lt or le")
    if raw["time_boundary"] not in {"inclusive", "exclusive"}:
        raise ValueError("recovery_timing_sensitivity.time_boundary must be inclusive or exclusive")
    if raw["duplicate_policy"] not in {"error", "drop_exact"}:
        raise ValueError("recovery_timing_sensitivity.duplicate_policy must be error or drop_exact")
    return _SensitivityProtocol(
        enabled=raw["enabled"],
        reference_time_column=raw["reference_time_column"],
        baseline_column=raw["baseline_column"],
        threshold_above_baseline_mg_dl=numeric["threshold_above_baseline_mg_dl"],
        value_comparison=raw["value_comparison"],
        fast_max_hours=numeric["fast_max_hours"],
        intermediate_max_hours=numeric["intermediate_max_hours"],
        time_boundary=raw["time_boundary"],
        sustained_hours=numeric["sustained_hours"],
        max_gap_hours=numeric["max_gap_hours"],
        endpoint_tolerance_hours=numeric["endpoint_tolerance_hours"],
        minimum_measurements=minimum,
        duplicate_policy=raw["duplicate_policy"],
    )


def validate_recovery_timing_sensitivity_config(config: Any) -> None:
    """Validate the complete sensitivity protocol without touching data."""

    _protocol(config)


def _within(value: pd.Series | float, threshold: float, comparison: str) -> Any:
    return value < threshold if comparison == "lt" else value <= threshold


def _time_within(hours: float, boundary: float, mode: str) -> bool:
    return hours <= boundary if mode == "inclusive" else hours < boundary


def _coverage(
    group: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    protocol: _SensitivityProtocol,
) -> tuple[bool, int, float, str]:
    tolerance = timedelta(hours=protocol.endpoint_tolerance_hours)
    window = group[(group["specimen_time"] >= start) & (group["specimen_time"] <= end + tolerance)]
    post = window[window["specimen_time"] > start]
    if len(post) < protocol.minimum_measurements:
        return False, len(post), float("nan"), "insufficient_measurements"
    if window["specimen_time"].max() < end - tolerance:
        return False, len(post), float("nan"), "observation_ends_before_intermediate_horizon"
    times = [start, *post["specimen_time"].tolist()]
    clipped = [times[0]]
    for time in times[1:]:
        clipped.append(time)
        if time >= end:
            break
    if clipped[-1] < end:
        clipped.append(end)
    gaps = [(right - left).total_seconds() / 3600 for left, right in zip(clipped[:-1], clipped[1:])]
    maximum = max(gaps, default=0.0)
    if maximum > protocol.max_gap_hours:
        return False, len(post), maximum, "measurement_gap_exceeds_maximum"
    return True, len(post), maximum, "complete_10_day_observation"


def _first_sustained_recovery(
    group: pd.DataFrame,
    reference: pd.Timestamp,
    end: pd.Timestamp,
    threshold: float,
    protocol: _SensitivityProtocol,
) -> pd.Timestamp | None:
    candidates = group[
        (group["specimen_time"] > reference)
        & (group["specimen_time"] <= end)
        & _within(group["creatinine_mg_dl"], threshold, protocol.value_comparison)
    ]
    for candidate in candidates["specimen_time"]:
        confirmation = candidate + timedelta(hours=protocol.sustained_hours)
        if confirmation > end:
            continue
        if protocol.sustained_hours == 0:
            return pd.Timestamp(candidate)
        sustain = group[
            (group["specimen_time"] >= candidate) & (group["specimen_time"] <= confirmation)
        ]
        values_recovered = bool(
            _within(sustain["creatinine_mg_dl"], threshold, protocol.value_comparison).all()
        )
        observed, _, _, _ = _coverage(group, pd.Timestamp(candidate), confirmation, protocol)
        if values_recovered and observed:
            return pd.Timestamp(candidate)
    return None


def build_recovery_timing_sensitivity(
    measurements: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """Classify fast/intermediate/no-recovery timing from each episode peak."""

    protocol = _protocol(config)
    phenotype_column = "phenotype" if "phenotype" in episodes else "label"
    common_episode_columns = {
        "episode_id",
        "subject_id",
        "hadm_id",
        phenotype_column,
    }
    missing_common = common_episode_columns.difference(episodes.columns)
    if missing_common:
        raise ValueError(f"sensitivity episodes missing: {sorted(missing_common)}")
    if not protocol.enabled:
        disabled = episodes[["episode_id", "subject_id", "hadm_id", phenotype_column]].rename(
            columns={phenotype_column: "primary_phenotype"}
        )
        disabled["recovery_timing_category"] = "not_run"
        disabled["sensitivity_status"] = "disabled"
        disabled["sensitivity_reason_code"] = "disabled_by_protocol"
        return disabled.reindex(columns=SENSITIVITY_COLUMNS)

    required_measurements = {
        "subject_id",
        "hadm_id",
        "specimen_time",
        "creatinine_mg_dl",
    }
    missing_measurements = required_measurements.difference(measurements.columns)
    if missing_measurements:
        raise ValueError(f"sensitivity measurements missing: {sorted(missing_measurements)}")
    required_episodes = {
        "episode_id",
        "subject_id",
        "hadm_id",
        phenotype_column,
        protocol.reference_time_column,
        protocol.baseline_column,
    }
    missing_episodes = required_episodes.difference(episodes.columns)
    if missing_episodes:
        raise ValueError(f"sensitivity episodes missing: {sorted(missing_episodes)}")

    labs = measurements.copy()
    labs["specimen_time"] = pd.to_datetime(labs["specimen_time"], errors="coerce")
    labs["creatinine_mg_dl"] = pd.to_numeric(labs["creatinine_mg_dl"], errors="coerce")
    if labs[["specimen_time", "creatinine_mg_dl"]].isna().any().any():
        raise ValueError("sensitivity measurements contain invalid timestamps/values")
    duplicate = labs.duplicated(
        ["subject_id", "hadm_id", "specimen_time", "creatinine_mg_dl"], keep="first"
    )
    if duplicate.any() and protocol.duplicate_policy == "error":
        raise ValueError("sensitivity measurements contain exact duplicates")
    labs = labs.loc[~duplicate].sort_values(
        ["subject_id", "hadm_id", "specimen_time"], kind="mergesort"
    )
    groups = {
        key: group.reset_index(drop=True)
        for key, group in labs.groupby(["subject_id", "hadm_id"], sort=False)
    }

    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        phenotype = getattr(episode, phenotype_column)
        base = {
            "episode_id": episode.episode_id,
            "subject_id": episode.subject_id,
            "hadm_id": episode.hadm_id,
            "primary_phenotype": phenotype,
        }
        reference = pd.to_datetime(
            getattr(episode, protocol.reference_time_column), errors="coerce"
        )
        baseline = pd.to_numeric(
            pd.Series([getattr(episode, protocol.baseline_column)]), errors="coerce"
        ).iloc[0]
        if pd.isna(reference) or pd.isna(baseline):
            rows.append(
                {
                    **base,
                    "recovery_timing_category": "not_evaluable",
                    "sensitivity_status": "not_evaluable",
                    "sensitivity_reason_code": "missing_reference_or_baseline",
                }
            )
            continue
        group = groups.get((episode.subject_id, episode.hadm_id))
        if group is None:
            rows.append(
                {
                    **base,
                    "recovery_timing_category": "not_evaluable",
                    "sensitivity_status": "not_evaluable",
                    "sensitivity_reason_code": "missing_admission_measurements",
                }
            )
            continue
        horizon = reference + timedelta(hours=protocol.intermediate_max_hours)
        threshold = float(baseline) + protocol.threshold_above_baseline_mg_dl
        recovery = _first_sustained_recovery(group, reference, horizon, threshold, protocol)
        complete, count, maximum_gap, coverage_reason = _coverage(
            group, reference, horizon, protocol
        )
        if recovery is not None:
            elapsed = (recovery - reference).total_seconds() / 3600
            if _time_within(elapsed, protocol.fast_max_hours, protocol.time_boundary):
                category = "fast"
            else:
                category = "intermediate"
            status = "classified"
            reason = f"{category}_recovery_from_peak"
        elif complete:
            elapsed = float("nan")
            category = "no_recovery_by_intermediate_horizon"
            status = "classified"
            reason = "no_recovery_by_intermediate_horizon"
        else:
            elapsed = float("nan")
            category = "censored"
            status = "censored"
            reason = coverage_reason
        rows.append(
            {
                **base,
                "recovery_timing_category": category,
                "sensitivity_status": status,
                "sensitivity_reason_code": reason,
                "reference_time": reference,
                "baseline_value": float(baseline),
                "recovery_threshold": threshold,
                "sensitivity_recovery_time": recovery,
                "hours_from_reference_to_recovery": elapsed,
                "intermediate_horizon_end": horizon,
                "last_observation_time": group["specimen_time"].max(),
                "observation_count": count,
                "maximum_gap_hours": maximum_gap,
            }
        )
    return pd.DataFrame(rows).reindex(columns=SENSITIVITY_COLUMNS)


__all__ = [
    "SENSITIVITY_COLUMNS",
    "build_recovery_timing_sensitivity",
    "validate_recovery_timing_sensitivity_config",
]
