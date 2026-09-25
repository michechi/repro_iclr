"""Deterministic, auditable construction of serum-creatinine AKI episodes.

This module deliberately contains no clinical defaults.  Every threshold, time
window, evidence requirement, and duplicate policy used by the state machine is
read from the ``episodes`` section of the supplied configuration.  It is thus
safe to use the same implementation for protocol sensitivity analyses without
silently changing the estimand.

The public entry point, :func:`construct_aki_episodes`, accepts the cleaned
creatinine measurements for exactly one patient.  Measurements may span more
than one admission; each admission is evaluated independently and episodes are
never allowed to cross an admission boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # Avoid a runtime dependency on the configuration module.
    from .config import AkiAuditConfig


PHENOTYPES = ("transient", "persistent", "relapsing")

EPISODE_COLUMNS = [
    "episode_id",
    "subject_id",
    "hadm_id",
    "episode_number",
    "baseline_value",
    "baseline_method",
    "baseline_n_measurements",
    "baseline_start_time",
    "baseline_end_time",
    "onset_time",
    "onset_value",
    "onset_rise",
    "peak_time",
    "peak_value",
    "recovery_time",
    "recovery_value",
    "recovery_confirmed_time",
    "recurrence_time",
    "recurrence_value",
    "follow_up_end_time",
    "required_observation_end_time",
    "last_observation_time",
    "phenotype",
    "status",
    "include_in_analysis",
    "reason_code",
]

AUDIT_EXTRA_COLUMNS = [
    "audit_record_type",
    "reason_detail",
    "onset_threshold_mg_dl",
    "onset_window_hours",
    "recovery_threshold_mg_dl",
    "recovery_window_hours",
    "sustained_recovery_hours",
    "relapse_window_days",
    "peak_window_end",
    "peak_boundary",
    "peak_tie_breaker",
    "post_onset_observation_count",
    "maximum_observed_gap_hours",
    "observation_adequate",
    "exact_duplicates_removed",
    "conflicting_timestamp_groups",
    "protocol_sha256",
    "protocol_json",
]

AUDIT_COLUMNS = EPISODE_COLUMNS + AUDIT_EXTRA_COLUMNS


class EpisodeConfigError(ValueError):
    """Raised when an episode-defining scientific choice is absent or invalid."""


@dataclass(frozen=True)
class EpisodeConstructionResult:
    """The constructed episodes and their one-row-per-decision audit trail.

    The object can also be unpacked as ``episodes, audit`` for convenience.
    Both data frames preserve their schemas when no AKI onset is found.
    """

    episodes: pd.DataFrame
    audit: pd.DataFrame
    resolved_measurements: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.episodes
        yield self.audit


@dataclass(frozen=True)
class _Protocol:
    timestamp_col: str
    value_col: str
    subject_id_col: str
    hadm_id_col: str
    admission_bounded: bool
    baseline_method: str
    baseline_lookback_hours: float
    baseline_minimum_measurements: int
    onset_absolute_rise: float
    onset_window_hours: float
    onset_boundary: str
    onset_tie_breaker: str
    recovery_margin: float
    recovery_window_hours: float
    recovery_time_boundary: str
    recovery_value_comparison: str
    sustained_recovery_hours: float
    follow_up_days: float
    follow_up_boundary: str
    peak_window_end: str
    peak_boundary: str
    peak_tie_breaker: str
    relapse_window_days: float
    relapse_anchor: str
    relapse_comparator: str
    relapse_boundary: str
    max_gap_hours: float
    endpoint_tolerance_hours: float
    minimum_post_onset_measurements: int
    exact_duplicate_policy: str
    conflicting_timestamp_policy: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | "AkiAuditConfig" | Any) -> "_Protocol":
        section = _member(config, "episodes", missing_ok=True)
        if section is _MISSING:
            # A direct episodes mapping is useful for the pure function in tests,
            # while application configs canonically place it under ``episodes``.
            section = config

        def required(*paths: Sequence[str]) -> Any:
            for path in paths:
                value = _nested_member(section, path)
                if value is not _MISSING and value is not None:
                    return value
            alternatives = " or ".join("episodes." + ".".join(p) for p in paths)
            raise EpisodeConfigError(f"Missing mandatory scientific configuration: {alternatives}")

        protocol = cls(
            timestamp_col=str(required(("columns", "timestamp"), ("timestamp_column",))),
            value_col=str(required(("columns", "value"), ("value_column",))),
            subject_id_col=str(required(("columns", "subject_id"), ("subject_id_column",))),
            hadm_id_col=str(required(("columns", "hadm_id"), ("hadm_id_column",))),
            admission_bounded=_as_bool(required(("admission_bounded",)), "admission_bounded"),
            baseline_method=str(
                required(("baseline", "method"), ("baseline", "estimator"))
            ).lower(),
            baseline_lookback_hours=_as_positive_float(
                required(
                    ("baseline", "lookback_hours"),
                    ("baseline", "lookback_duration_hours"),
                ),
                "baseline.lookback_hours",
            ),
            baseline_minimum_measurements=_as_positive_int(
                required(
                    ("baseline", "minimum_measurements"),
                    ("baseline", "minimum_evidence"),
                ),
                "baseline.minimum_measurements",
            ),
            onset_absolute_rise=_as_positive_float(
                required(
                    ("onset", "absolute_rise_mg_dl"),
                    ("onset", "absolute_rise"),
                ),
                "onset.absolute_rise_mg_dl",
            ),
            onset_window_hours=_as_positive_float(
                required(
                    ("onset", "window_hours"),
                ),
                "onset.window_hours",
            ),
            onset_boundary=str(
                required(
                    ("onset", "boundary"),
                )
            ).lower(),
            onset_tie_breaker=str(
                required(
                    ("onset", "tie_breaker"),
                )
            ).lower(),
            recovery_margin=_as_nonnegative_float(
                required(
                    ("recovery", "threshold_above_baseline_mg_dl"),
                    ("recovery", "threshold_mg_dl"),
                ),
                "recovery.threshold_above_baseline_mg_dl",
            ),
            recovery_window_hours=_as_positive_float(
                required(
                    ("recovery", "window_hours"),
                ),
                "recovery.window_hours",
            ),
            recovery_time_boundary=str(
                required(("recovery", "time_boundary"), ("recovery", "boundary"))
            ).lower(),
            recovery_value_comparison=str(
                required(
                    ("recovery", "value_comparison"),
                )
            ).lower(),
            sustained_recovery_hours=_as_nonnegative_float(
                required(
                    ("recovery", "sustained_hours"),
                ),
                "recovery.sustained_hours",
            ),
            follow_up_days=_as_positive_float(
                required(
                    ("follow_up", "duration_days"),
                    ("followup", "duration_days"),
                ),
                "follow_up.duration_days",
            ),
            follow_up_boundary=str(
                required(("follow_up", "boundary"), ("followup", "boundary"))
            ).lower(),
            peak_window_end=str(required(("peak", "window_end"))).lower(),
            peak_boundary=str(required(("peak", "boundary"))).lower(),
            peak_tie_breaker=str(required(("peak", "tie_breaker"))).lower(),
            relapse_window_days=_as_positive_float(
                required(
                    ("relapse", "window_days"),
                ),
                "relapse.window_days",
            ),
            relapse_anchor=str(
                required(
                    ("relapse", "anchor"),
                )
            ).lower(),
            relapse_comparator=str(
                required(
                    ("relapse", "comparator"),
                )
            ).lower(),
            relapse_boundary=str(
                required(
                    ("relapse", "boundary"),
                )
            ).lower(),
            max_gap_hours=_as_positive_float(
                required(
                    ("observation", "max_gap_hours"),
                ),
                "observation.max_gap_hours",
            ),
            endpoint_tolerance_hours=_as_nonnegative_float(
                required(
                    ("observation", "endpoint_tolerance_hours"),
                ),
                "observation.endpoint_tolerance_hours",
            ),
            minimum_post_onset_measurements=_as_positive_int(
                required(
                    ("observation", "minimum_post_onset_measurements"),
                ),
                "observation.minimum_post_onset_measurements",
            ),
            exact_duplicate_policy=str(
                required(
                    ("duplicates", "exact"),
                )
            ).lower(),
            conflicting_timestamp_policy=str(
                required(
                    ("duplicates", "conflicting_timestamp"),
                )
            ).lower(),
        )
        protocol.validate()
        return protocol

    def validate(self) -> None:
        if not self.admission_bounded:
            raise EpisodeConfigError(
                "episodes.admission_bounded must be true for the locked index-admission protocol"
            )
        _choice(
            self.baseline_method,
            "baseline.method",
            {"minimum", "median", "mean", "latest", "earliest"},
        )
        _choice(self.onset_boundary, "onset.boundary", {"inclusive", "exclusive"})
        _choice(self.onset_tie_breaker, "onset.tie_breaker", {"earliest", "largest_rise"})
        _choice(self.recovery_time_boundary, "recovery.time_boundary", {"inclusive", "exclusive"})
        _choice(self.recovery_value_comparison, "recovery.value_comparison", {"lt", "le"})
        _choice(self.follow_up_boundary, "follow_up.boundary", {"inclusive", "exclusive"})
        _choice(
            self.peak_window_end,
            "peak.window_end",
            {"follow_up", "phenotype_horizon"},
        )
        _choice(self.peak_boundary, "peak.boundary", {"inclusive", "exclusive"})
        _choice(self.peak_tie_breaker, "peak.tie_breaker", {"earliest", "latest"})
        _choice(self.relapse_anchor, "relapse.anchor", {"onset", "recovery"})
        _choice(
            self.relapse_comparator,
            "relapse.comparator",
            {"episode_baseline", "recovery_nadir", "rolling_baseline"},
        )
        _choice(self.relapse_boundary, "relapse.boundary", {"inclusive", "exclusive"})
        _choice(self.exact_duplicate_policy, "duplicates.exact", {"drop", "error"})
        _choice(
            self.conflicting_timestamp_policy,
            "duplicates.conflicting_timestamp",
            {"ambiguous", "error", "first", "last", "mean"},
        )

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Onset:
    index: int
    time: pd.Timestamp
    value: float
    baseline: float
    baseline_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Coverage:
    adequate: bool
    count: int
    max_gap_hours: float
    reason: str | None


_MISSING = object()


def construct_aki_episodes(
    measurements: pd.DataFrame | Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | "AkiAuditConfig" | Any,
) -> EpisodeConstructionResult:
    """Construct admission-bounded AKI episodes for one patient.

    Parameters
    ----------
    measurements:
        Timestamped, unit-normalized serum-creatinine rows for exactly one
        patient.  Column names are specified explicitly in ``config``.
    config:
        Either an ``AkiAuditConfig``, a mapping containing an ``episodes``
        section, or the episodes mapping itself.  Missing protocol choices raise
        :class:`EpisodeConfigError` before any trajectory is evaluated.

    Returns
    -------
    EpisodeConstructionResult
        ``episodes`` contains AKI onsets, including explicit censored or
        ambiguous rows. ``audit`` contains the same decisions plus the evidence
        and resolved protocol needed to reproduce every label.
    """

    protocol = _Protocol.from_config(config)
    frame = pd.DataFrame(measurements).copy()
    _validate_input_columns(frame, protocol)

    protocol_json = json.dumps(protocol.serializable(), sort_keys=True, separators=(",", ":"))
    protocol_sha = hashlib.sha256(protocol_json.encode("utf-8")).hexdigest()

    if frame.empty:
        return EpisodeConstructionResult(_empty_episodes(), _empty_audit(), frame.copy())

    frame["__row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["__time"] = pd.to_datetime(frame[protocol.timestamp_col], errors="coerce")
    frame["__value"] = pd.to_numeric(frame[protocol.value_col], errors="coerce")
    invalid = frame["__time"].isna() | frame["__value"].isna() | ~np.isfinite(frame["__value"])
    if invalid.any():
        bad_rows = frame.index[invalid].tolist()
        raise ValueError(
            "Episode construction requires cleaned timestamps and finite numeric creatinine "
            f"values; invalid row indices: {bad_rows[:10]}"
        )
    if (frame["__value"] < 0).any():
        bad_rows = frame.index[frame["__value"] < 0].tolist()
        raise ValueError(
            f"Creatinine values must be non-negative; invalid row indices: {bad_rows[:10]}"
        )

    subjects = frame[protocol.subject_id_col].dropna().unique()
    if len(subjects) != 1 or frame[protocol.subject_id_col].isna().any():
        raise ValueError("construct_aki_episodes accepts rows for exactly one non-null subject_id")
    if frame[protocol.hadm_id_col].isna().any():
        raise ValueError("Admission-bounded construction requires a non-null hadm_id on every row")

    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    resolved_parts: list[pd.DataFrame] = []
    input_columns = list(frame.columns)
    subject_id = subjects[0]

    # sort=False preserves deterministic input admission order; output is sorted
    # chronologically below, independent of the original order of measurements.
    for hadm_id, admission in frame.groupby(protocol.hadm_id_col, sort=False, dropna=False):
        cleaned, exact_removed, conflict_count, conflict_ambiguous = _resolve_duplicates(
            admission, protocol
        )
        cleaned = cleaned.sort_values(["__time", "__row_order"], kind="mergesort").reset_index(
            drop=True
        )
        canonical = cleaned.copy()
        canonical[protocol.timestamp_col] = canonical["__time"]
        canonical[protocol.value_col] = canonical["__value"]
        resolved_parts.append(canonical[input_columns].copy())

        if conflict_ambiguous:
            row = _decision_row(
                subject_id=subject_id,
                hadm_id=hadm_id,
                episode_number=1,
                phenotype="ambiguous",
                status="ambiguous",
                reason_code="conflicting_simultaneous_values",
                reason_detail=(
                    "At least one specimen timestamp has multiple non-identical creatinine "
                    "values and the configured policy is to mark the trajectory ambiguous."
                ),
                protocol=protocol,
                protocol_json=protocol_json,
                protocol_sha=protocol_sha,
                exact_duplicates_removed=exact_removed,
                conflicting_timestamp_groups=conflict_count,
                last_observation_time=cleaned["__time"].max() if not cleaned.empty else pd.NaT,
            )
            episode_rows.append(_episode_projection(row))
            audit_rows.append(row)
            continue

        admission_episode_rows, admission_audit_rows = _construct_for_admission(
            cleaned,
            subject_id=subject_id,
            hadm_id=hadm_id,
            protocol=protocol,
            protocol_json=protocol_json,
            protocol_sha=protocol_sha,
            exact_duplicates_removed=exact_removed,
            conflicting_timestamp_groups=conflict_count,
        )
        episode_rows.extend(admission_episode_rows)
        audit_rows.extend(admission_audit_rows)

    episodes = pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=AUDIT_COLUMNS)
    if not episodes.empty:
        episodes = episodes.sort_values(
            ["onset_time", "hadm_id", "episode_number"], na_position="last", kind="mergesort"
        ).reset_index(drop=True)
    if not audit.empty:
        audit = audit.sort_values(
            ["onset_time", "hadm_id", "episode_number"], na_position="last", kind="mergesort"
        ).reset_index(drop=True)
    resolved = pd.concat(resolved_parts, ignore_index=True)
    resolved = resolved.drop(columns=["__row_order", "__time", "__value"], errors="ignore")
    return EpisodeConstructionResult(
        episodes=episodes,
        audit=audit,
        resolved_measurements=resolved,
    )


def _construct_for_admission(
    frame: pd.DataFrame,
    *,
    subject_id: Any,
    hadm_id: Any,
    protocol: _Protocol,
    protocol_json: str,
    protocol_sha: str,
    exact_duplicates_removed: int,
    conflicting_timestamp_groups: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    cursor = 0
    episode_number = 0

    while cursor < len(frame):
        onset = _find_onset(frame, cursor, protocol)
        if onset is None:
            if episode_number == 0:
                reason = (
                    "insufficient_baseline_evidence"
                    if not _has_any_eligible_baseline(frame, cursor, protocol)
                    else "no_aki_onset"
                )
                detail = (
                    "No measurement had the configured minimum baseline evidence."
                    if reason == "insufficient_baseline_evidence"
                    else "No measurement met the configured absolute-rise onset rule."
                )
                audit_rows.append(
                    _decision_row(
                        subject_id=subject_id,
                        hadm_id=hadm_id,
                        episode_number=0,
                        phenotype="no_aki",
                        status="no_aki",
                        reason_code=reason,
                        reason_detail=detail,
                        protocol=protocol,
                        protocol_json=protocol_json,
                        protocol_sha=protocol_sha,
                        exact_duplicates_removed=exact_duplicates_removed,
                        conflicting_timestamp_groups=conflicting_timestamp_groups,
                        last_observation_time=frame["__time"].max() if not frame.empty else pd.NaT,
                    )
                )
            break

        episode_number += 1
        decision = _classify_episode(
            frame,
            onset,
            subject_id=subject_id,
            hadm_id=hadm_id,
            episode_number=episode_number,
            protocol=protocol,
            protocol_json=protocol_json,
            protocol_sha=protocol_sha,
            exact_duplicates_removed=exact_duplicates_removed,
            conflicting_timestamp_groups=conflicting_timestamp_groups,
        )
        episode_rows.append(_episode_projection(decision))
        audit_rows.append(decision)

        required_end = decision["required_observation_end_time"]
        if pd.isna(required_end):
            break
        later = frame.index[frame["__time"] > required_end].tolist()
        if not later:
            break
        cursor = later[0]

    return episode_rows, audit_rows


def _find_onset(frame: pd.DataFrame, start_index: int, protocol: _Protocol) -> _Onset | None:
    qualifiers: list[_Onset] = []
    for index in range(max(1, start_index), len(frame)):
        # Candidate onsets are restricted by ``start_index`` so episodes cannot
        # overlap, but a new episode may use recovered measurements immediately
        # before that boundary as its configured lookback evidence.
        baseline_indices = _baseline_indices(frame, index, 0, protocol)
        if len(baseline_indices) < protocol.baseline_minimum_measurements:
            continue
        baseline = _estimate_baseline(frame, baseline_indices, protocol.baseline_method)
        rise = float(frame.at[index, "__value"] - baseline)
        if rise >= protocol.onset_absolute_rise:
            qualifiers.append(
                _Onset(
                    index=index,
                    time=frame.at[index, "__time"],
                    value=float(frame.at[index, "__value"]),
                    baseline=baseline,
                    baseline_indices=tuple(baseline_indices),
                )
            )
            # All measurements are uniquely timestamped after duplicate
            # resolution, so the first qualifying time is deterministic.
            if protocol.onset_tie_breaker == "earliest":
                return qualifiers[0]
    if not qualifiers:
        return None
    # ``largest_rise`` is an explicit alternative for sensitivity analyses.
    # Ties resolve by the stable chronological index.
    return max(
        qualifiers, key=lambda candidate: (candidate.value - candidate.baseline, -candidate.index)
    )


def _baseline_indices(
    frame: pd.DataFrame, index: int, floor_index: int, protocol: _Protocol
) -> list[int]:
    current_time = frame.at[index, "__time"]
    eligible: list[int] = []
    for prior in range(floor_index, index):
        delta_hours = (current_time - frame.at[prior, "__time"]).total_seconds() / 3600.0
        if delta_hours <= 0:
            continue
        in_onset = _within(delta_hours, protocol.onset_window_hours, protocol.onset_boundary)
        in_baseline = delta_hours <= protocol.baseline_lookback_hours
        if in_onset and in_baseline:
            eligible.append(prior)
    return eligible


def _has_any_eligible_baseline(frame: pd.DataFrame, start_index: int, protocol: _Protocol) -> bool:
    return any(
        len(_baseline_indices(frame, index, 0, protocol)) >= protocol.baseline_minimum_measurements
        for index in range(max(1, start_index), len(frame))
    )


def _estimate_baseline(frame: pd.DataFrame, indices: Sequence[int], method: str) -> float:
    values = frame.loc[list(indices), "__value"].to_numpy(dtype=float)
    if method == "minimum":
        return float(np.min(values))
    if method == "median":
        return float(np.median(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "latest":
        return float(values[-1])
    if method == "earliest":
        return float(values[0])
    raise AssertionError(f"Validated baseline method was not implemented: {method}")


def _classify_episode(
    frame: pd.DataFrame,
    onset: _Onset,
    *,
    subject_id: Any,
    hadm_id: Any,
    episode_number: int,
    protocol: _Protocol,
    protocol_json: str,
    protocol_sha: str,
    exact_duplicates_removed: int,
    conflicting_timestamp_groups: int,
) -> dict[str, Any]:
    follow_up_end = onset.time + timedelta(days=protocol.follow_up_days)
    recovery_limit = onset.time + timedelta(hours=protocol.recovery_window_hours)
    threshold = onset.baseline + protocol.recovery_margin

    # Search through the longest window that could alter the phenotype.  A
    # recovery-anchored relapse window is extended once recovery is identified.
    provisional_end = follow_up_end
    recovery = _find_sustained_recovery(frame, onset.index, provisional_end, threshold, protocol)
    if recovery is not None and protocol.relapse_anchor == "recovery":
        relapse_end = recovery[1] + timedelta(days=protocol.relapse_window_days)
    else:
        relapse_end = onset.time + timedelta(days=protocol.relapse_window_days)
    required_end = max(follow_up_end, relapse_end)
    if required_end > provisional_end:
        recovery = _find_sustained_recovery(frame, onset.index, required_end, threshold, protocol)
        if recovery is not None and protocol.relapse_anchor == "recovery":
            relapse_end = recovery[1] + timedelta(days=protocol.relapse_window_days)
            required_end = max(follow_up_end, relapse_end)

    coverage = _coverage(frame, onset.time, required_end, protocol)
    recurrence: tuple[int, pd.Timestamp, float] | None = None
    if recovery is not None:
        recurrence = _find_recurrence(
            frame,
            onset,
            recovery,
            relapse_end,
            protocol,
        )

    peak_end = follow_up_end if protocol.peak_window_end == "follow_up" else required_end
    window_rows = frame[
        (frame["__time"] >= onset.time)
        & _time_mask(frame["__time"], peak_end, protocol.peak_boundary)
    ]
    if window_rows.empty:
        peak_time = onset.time
        peak_value = onset.value
    else:
        maximum = window_rows["__value"].max()
        peak_candidates = window_rows.loc[window_rows["__value"].eq(maximum)]
        if protocol.peak_tie_breaker == "earliest":
            peak_index = peak_candidates.index[0]
        else:
            peak_index = peak_candidates.index[-1]
        peak_time = frame.at[peak_index, "__time"]
        peak_value = float(frame.at[peak_index, "__value"])

    recovery_time = recovery[1] if recovery is not None else pd.NaT
    recovery_value = recovery[2] if recovery is not None else np.nan
    recovery_confirmed = recovery[3] if recovery is not None else pd.NaT
    recurrence_time = recurrence[1] if recurrence is not None else pd.NaT
    recurrence_value = recurrence[2] if recurrence is not None else np.nan

    if not coverage.adequate:
        phenotype = "censored"
        status = "censored"
        reason_code = coverage.reason or "inadequate_follow_up"
        reason_detail = (
            "The admission-bounded measurement record does not satisfy the configured "
            "density and endpoint evidence through every phenotype-defining window."
        )
    elif recurrence is not None:
        phenotype = "relapsing"
        status = "labeled"
        reason_code = "sustained_recovery_then_recurrence"
        reason_detail = (
            "A sustained recovery was followed by a new qualifying creatinine rise inside "
            "the configured relapse window; relapse takes phenotype precedence."
        )
    elif recovery is not None and _timestamp_within(
        recovery_time, recovery_limit, protocol.recovery_time_boundary
    ):
        phenotype = "transient"
        status = "labeled"
        reason_code = "recovery_within_window_without_recurrence"
        reason_detail = (
            "Creatinine recovered below the configured baseline-relative threshold within "
            "the recovery window, remained recovered for the required interval, and did "
            "not recur during follow-up."
        )
    else:
        phenotype = "persistent"
        status = "labeled"
        reason_code = "no_sustained_recovery_within_window"
        reason_detail = (
            "No sustained recovery satisfying the configured evidence rule occurred within "
            "the recovery window; any later sustained recovery does not change persistence."
        )

    baseline_times = frame.loc[list(onset.baseline_indices), "__time"]
    return _decision_row(
        subject_id=subject_id,
        hadm_id=hadm_id,
        episode_number=episode_number,
        phenotype=phenotype,
        status=status,
        reason_code=reason_code,
        reason_detail=reason_detail,
        protocol=protocol,
        protocol_json=protocol_json,
        protocol_sha=protocol_sha,
        exact_duplicates_removed=exact_duplicates_removed,
        conflicting_timestamp_groups=conflicting_timestamp_groups,
        baseline_value=onset.baseline,
        baseline_n_measurements=len(onset.baseline_indices),
        baseline_start_time=baseline_times.min(),
        baseline_end_time=baseline_times.max(),
        onset_time=onset.time,
        onset_value=onset.value,
        onset_rise=onset.value - onset.baseline,
        peak_time=peak_time,
        peak_value=peak_value,
        recovery_time=recovery_time,
        recovery_value=recovery_value,
        recovery_confirmed_time=recovery_confirmed,
        recurrence_time=recurrence_time,
        recurrence_value=recurrence_value,
        follow_up_end_time=follow_up_end,
        required_observation_end_time=required_end,
        last_observation_time=frame["__time"].max(),
        post_onset_observation_count=coverage.count,
        maximum_observed_gap_hours=coverage.max_gap_hours,
        observation_adequate=coverage.adequate,
    )


def _find_sustained_recovery(
    frame: pd.DataFrame,
    onset_index: int,
    search_end: pd.Timestamp,
    threshold: float,
    protocol: _Protocol,
) -> tuple[int, pd.Timestamp, float, pd.Timestamp] | None:
    for index in range(onset_index + 1, len(frame)):
        time = frame.at[index, "__time"]
        if time > search_end:
            break
        value = float(frame.at[index, "__value"])
        is_recovered = (
            value < threshold if protocol.recovery_value_comparison == "lt" else value <= threshold
        )
        if not is_recovered:
            continue
        confirmation_end = time + timedelta(hours=protocol.sustained_recovery_hours)
        if confirmation_end > search_end:
            continue
        sustain_rows = frame[(frame["__time"] >= time) & (frame["__time"] <= confirmation_end)]
        if protocol.recovery_value_comparison == "lt":
            all_recovered = bool((sustain_rows["__value"] < threshold).all())
        else:
            all_recovered = bool((sustain_rows["__value"] <= threshold).all())
        sustain_coverage = _coverage(frame, time, confirmation_end, protocol, minimum_count=1)
        if all_recovered and sustain_coverage.adequate:
            return index, time, value, confirmation_end
    return None


def _find_recurrence(
    frame: pd.DataFrame,
    onset: _Onset,
    recovery: tuple[int, pd.Timestamp, float, pd.Timestamp],
    relapse_end: pd.Timestamp,
    protocol: _Protocol,
) -> tuple[int, pd.Timestamp, float] | None:
    recovery_index, recovery_time, _, confirmed_time = recovery
    for index in range(recovery_index + 1, len(frame)):
        time = frame.at[index, "__time"]
        if time <= confirmed_time:
            continue
        delta_to_end = time - relapse_end
        if delta_to_end > pd.Timedelta(0) or (
            delta_to_end == pd.Timedelta(0) and protocol.relapse_boundary == "exclusive"
        ):
            break
        value = float(frame.at[index, "__value"])
        recent = [
            prior
            for prior in range(recovery_index, index)
            if 0
            < (time - frame.at[prior, "__time"]).total_seconds() / 3600.0
            <= protocol.onset_window_hours
        ]
        if not recent:
            continue
        if protocol.relapse_comparator == "episode_baseline":
            comparator = onset.baseline
        elif protocol.relapse_comparator == "recovery_nadir":
            comparator = float(frame.loc[recent, "__value"].min())
        else:  # rolling_baseline
            eligible = [
                prior
                for prior in recent
                if (time - frame.at[prior, "__time"]).total_seconds() / 3600.0
                <= protocol.baseline_lookback_hours
            ]
            if len(eligible) < protocol.baseline_minimum_measurements:
                continue
            comparator = _estimate_baseline(frame, eligible, protocol.baseline_method)
        if value - comparator >= protocol.onset_absolute_rise:
            return index, time, value
    return None


def _coverage(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    protocol: _Protocol,
    *,
    minimum_count: int | None = None,
) -> _Coverage:
    if end == start:
        # A protocol may explicitly define zero hours of sustained recovery.
        # In that case the recovery measurement itself is complete evidence;
        # requiring a later measurement would silently impose a non-zero rule.
        return _Coverage(True, 0, 0.0, None)
    tolerance = timedelta(hours=protocol.endpoint_tolerance_hours)
    observed = frame[(frame["__time"] >= start) & (frame["__time"] <= end + tolerance)]
    post = observed[observed["__time"] > start]
    required_count = (
        protocol.minimum_post_onset_measurements if minimum_count is None else minimum_count
    )
    if len(post) < required_count:
        return _Coverage(False, len(post), np.nan, "insufficient_post_onset_measurements")
    last_time = observed["__time"].max()
    if last_time < end - tolerance:
        return _Coverage(False, len(post), np.nan, "follow_up_ends_before_required_horizon")

    evidence_times = [start] + observed.loc[observed["__time"] > start, "__time"].tolist()
    # Stop at the first measurement that establishes observation through the end;
    # later rows inside the endpoint tolerance cannot create an artificial gap.
    clipped: list[pd.Timestamp] = [evidence_times[0]]
    for time in evidence_times[1:]:
        clipped.append(time)
        if time >= end:
            break
    if clipped[-1] < end:
        clipped.append(end)
    gaps = [
        (right - left).total_seconds() / 3600.0 for left, right in zip(clipped[:-1], clipped[1:])
    ]
    max_gap = max(gaps, default=0.0)
    if max_gap > protocol.max_gap_hours:
        return _Coverage(False, len(post), max_gap, "measurement_gap_exceeds_configured_maximum")
    return _Coverage(True, len(post), max_gap, None)


def _resolve_duplicates(
    frame: pd.DataFrame, protocol: _Protocol
) -> tuple[pd.DataFrame, int, int, bool]:
    working = frame.sort_values("__row_order", kind="mergesort").copy()
    exact_mask = working.duplicated(["__time", "__value"], keep="first")
    exact_count = int(exact_mask.sum())
    if exact_count and protocol.exact_duplicate_policy == "error":
        raise ValueError(f"Found {exact_count} exact duplicate creatinine measurements")
    if exact_count:
        working = working.loc[~exact_mask].copy()

    counts = working.groupby("__time", dropna=False)["__value"].nunique(dropna=False)
    conflict_times = counts[counts > 1].index.tolist()
    conflict_count = len(conflict_times)
    if not conflict_count:
        return working, exact_count, 0, False
    policy = protocol.conflicting_timestamp_policy
    if policy == "error":
        raise ValueError(
            f"Found {conflict_count} timestamps with conflicting creatinine measurements"
        )
    if policy == "ambiguous":
        return working, exact_count, conflict_count, True
    if policy in {"first", "last"}:
        working = working.drop_duplicates("__time", keep=policy)
    elif policy == "mean":
        conflict_set = set(conflict_times)
        retained = working[~working["__time"].isin(conflict_set)].copy()
        averaged = (
            working[working["__time"].isin(conflict_set)]
            .groupby("__time", as_index=False, sort=False)
            .agg({"__value": "mean", "__row_order": "min"})
        )
        template_columns = list(working.columns)
        for column in template_columns:
            if column not in averaged:
                averaged[column] = np.nan
        # Preserve identifiers from the admission rather than averaging them.
        averaged[protocol.subject_id_col] = working[protocol.subject_id_col].iloc[0]
        averaged[protocol.hadm_id_col] = working[protocol.hadm_id_col].iloc[0]
        averaged[protocol.timestamp_col] = averaged["__time"]
        averaged[protocol.value_col] = averaged["__value"]
        working = pd.concat([retained, averaged[template_columns]], ignore_index=True)
    return working, exact_count, conflict_count, False


def _decision_row(
    *,
    subject_id: Any,
    hadm_id: Any,
    episode_number: int,
    phenotype: str,
    status: str,
    reason_code: str,
    reason_detail: str,
    protocol: _Protocol,
    protocol_json: str,
    protocol_sha: str,
    exact_duplicates_removed: int,
    conflicting_timestamp_groups: int,
    baseline_value: float = np.nan,
    baseline_n_measurements: int = 0,
    baseline_start_time: Any = pd.NaT,
    baseline_end_time: Any = pd.NaT,
    onset_time: Any = pd.NaT,
    onset_value: float = np.nan,
    onset_rise: float = np.nan,
    peak_time: Any = pd.NaT,
    peak_value: float = np.nan,
    recovery_time: Any = pd.NaT,
    recovery_value: float = np.nan,
    recovery_confirmed_time: Any = pd.NaT,
    recurrence_time: Any = pd.NaT,
    recurrence_value: float = np.nan,
    follow_up_end_time: Any = pd.NaT,
    required_observation_end_time: Any = pd.NaT,
    last_observation_time: Any = pd.NaT,
    post_onset_observation_count: int = 0,
    maximum_observed_gap_hours: float = np.nan,
    observation_adequate: bool = False,
) -> dict[str, Any]:
    normalized_subject = _id_text(subject_id)
    normalized_hadm = _id_text(hadm_id)
    episode_id = f"{normalized_subject}-{normalized_hadm}-{episode_number:03d}"
    return {
        "episode_id": episode_id,
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "episode_number": episode_number,
        "baseline_value": baseline_value,
        "baseline_method": protocol.baseline_method,
        "baseline_n_measurements": baseline_n_measurements,
        "baseline_start_time": baseline_start_time,
        "baseline_end_time": baseline_end_time,
        "onset_time": onset_time,
        "onset_value": onset_value,
        "onset_rise": onset_rise,
        "peak_time": peak_time,
        "peak_value": peak_value,
        "recovery_time": recovery_time,
        "recovery_value": recovery_value,
        "recovery_confirmed_time": recovery_confirmed_time,
        "recurrence_time": recurrence_time,
        "recurrence_value": recurrence_value,
        "follow_up_end_time": follow_up_end_time,
        "last_observation_time": last_observation_time,
        "phenotype": phenotype,
        "status": status,
        "include_in_analysis": bool(status == "labeled" and phenotype in PHENOTYPES),
        "reason_code": reason_code,
        "audit_record_type": "episode" if episode_number else "admission",
        "reason_detail": reason_detail,
        "onset_threshold_mg_dl": protocol.onset_absolute_rise,
        "onset_window_hours": protocol.onset_window_hours,
        "recovery_threshold_mg_dl": protocol.recovery_margin,
        "recovery_window_hours": protocol.recovery_window_hours,
        "sustained_recovery_hours": protocol.sustained_recovery_hours,
        "relapse_window_days": protocol.relapse_window_days,
        "peak_window_end": protocol.peak_window_end,
        "peak_boundary": protocol.peak_boundary,
        "peak_tie_breaker": protocol.peak_tie_breaker,
        "required_observation_end_time": required_observation_end_time,
        "post_onset_observation_count": post_onset_observation_count,
        "maximum_observed_gap_hours": maximum_observed_gap_hours,
        "observation_adequate": observation_adequate,
        "exact_duplicates_removed": exact_duplicates_removed,
        "conflicting_timestamp_groups": conflicting_timestamp_groups,
        "protocol_sha256": protocol_sha,
        "protocol_json": protocol_json,
    }


def _episode_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in EPISODE_COLUMNS}


def _validate_input_columns(frame: pd.DataFrame, protocol: _Protocol) -> None:
    required = {
        protocol.timestamp_col,
        protocol.value_col,
        protocol.subject_id_col,
        protocol.hadm_id_col,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing episode-construction input columns: {missing}")


def _empty_episodes() -> pd.DataFrame:
    return pd.DataFrame(columns=EPISODE_COLUMNS)


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=AUDIT_COLUMNS)


def _member(value: Any, key: str, *, missing_ok: bool = False) -> Any:
    if isinstance(value, Mapping):
        result = value.get(key, _MISSING)
    else:
        result = getattr(value, key, _MISSING)
    if result is _MISSING and not missing_ok:
        raise KeyError(key)
    return result


def _nested_member(value: Any, path: Sequence[str]) -> Any:
    current = value
    for key in path:
        current = _member(current, key, missing_ok=True)
        if current is _MISSING:
            return _MISSING
    return current


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise EpisodeConfigError(f"episodes.{name} must be a boolean, got {value!r}")


def _as_positive_float(value: Any, name: str) -> float:
    result = _numeric(value, name)
    if result <= 0:
        raise EpisodeConfigError(f"episodes.{name} must be > 0, got {value!r}")
    return result


def _as_nonnegative_float(value: Any, name: str) -> float:
    result = _numeric(value, name)
    if result < 0:
        raise EpisodeConfigError(f"episodes.{name} must be >= 0, got {value!r}")
    return result


def _numeric(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise EpisodeConfigError(f"episodes.{name} must be numeric, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EpisodeConfigError(f"episodes.{name} must be numeric, got {value!r}") from exc
    if not np.isfinite(result):
        raise EpisodeConfigError(f"episodes.{name} must be finite, got {value!r}")
    return result


def _as_positive_int(value: Any, name: str) -> int:
    number = _numeric(value, name)
    if not number.is_integer() or number < 1:
        raise EpisodeConfigError(f"episodes.{name} must be an integer >= 1, got {value!r}")
    return int(number)


def _choice(value: str, name: str, choices: set[str]) -> None:
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise EpisodeConfigError(f"episodes.{name} must be one of {{{allowed}}}, got {value!r}")


def _within(value: float, boundary: float, mode: str) -> bool:
    return value <= boundary if mode == "inclusive" else value < boundary


def _timestamp_within(timestamp: pd.Timestamp, boundary: pd.Timestamp, mode: str) -> bool:
    return timestamp <= boundary if mode == "inclusive" else timestamp < boundary


def _time_mask(series: pd.Series, end: pd.Timestamp, mode: str) -> pd.Series:
    return series <= end if mode == "inclusive" else series < end


def _id_text(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


__all__ = [
    "AUDIT_COLUMNS",
    "EPISODE_COLUMNS",
    "PHENOTYPES",
    "EpisodeConfigError",
    "EpisodeConstructionResult",
    "construct_aki_episodes",
]
