"""Non-model orchestration for the MIMIC-IV AKI trajectory audit.

This module joins the cohort, episode, representation, split, and matching
components without training a classifier.  It deliberately keeps censored and
ambiguous decisions in the population tables while restricting model-facing
tables to episodes explicitly marked ``include_in_analysis``.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd

from .dataset import materialize_episode_events, normalize_episode_labels
from .episodes import construct_aki_episodes
from .matching import (
    MatchingResult,
    coarsened_exact_match,
    select_first_eligible_episode,
)
from .reporting import build_class_distribution, build_episode_flow_report, config_digest
from .representations import build_tabular_feature_sets
from .sensitivity import build_recovery_timing_sensitivity
from .splits import attach_patient_splits, make_patient_splits, validate_patient_splits


ADMISSION_ANNOTATION_COLUMNS = (
    "state_machine_reason_code",
    "admission_censor_annotation_status",
    "admission_censor_reason",
    "admission_censor_endpoint_type",
    "admission_censor_endpoint_time",
)


@dataclass(frozen=True)
class PopulationEpisodeResult:
    """Population-wide episode decisions before analysis-dataset selection."""

    episodes: pd.DataFrame
    audit: pd.DataFrame
    resolved_measurements: pd.DataFrame | None = None


@dataclass(frozen=True)
class AkiPipelineResult:
    """All non-model artifacts generated from one fixed patient split map."""

    episodes: pd.DataFrame
    episode_audit: pd.DataFrame
    labeled_episode_events: pd.DataFrame
    patient_splits: pd.DataFrame
    primary_labels: pd.DataFrame
    primary_events: pd.DataFrame
    primary_matching_features: pd.DataFrame
    primary_matched_labels: pd.DataFrame
    primary_matched_events: pd.DataFrame
    secondary_labels: pd.DataFrame
    secondary_events: pd.DataFrame
    recovery_timing_sensitivity: pd.DataFrame
    matching: MatchingResult
    config_hash: str


@dataclass(frozen=True)
class PersistedPipelineArtifacts:
    """Paths written by :func:`persist_pipeline_result`."""

    parquet: Mapping[str, Path]
    csv_summaries: Mapping[str, Path]


_POPULATION_WORKER_CONFIG: Any | None = None


def _positive_execution_integer(value: Any, name: str) -> int:
    """Validate a non-scientific population-execution setting."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _parallel_worker_config(config: Any) -> Any:
    """Return a pickle-safe protocol without changing its resolved values."""

    as_dict = getattr(config, "as_dict", None)
    worker_config = as_dict() if callable(as_dict) else config
    try:
        pickle.dumps(worker_config)
    except Exception as exc:
        raise TypeError(
            "config must be pickleable for parallel population episode construction"
        ) from exc
    return worker_config


def _initialize_population_worker(config: Any) -> None:
    global _POPULATION_WORKER_CONFIG
    _POPULATION_WORKER_CONFIG = config


def _concat_population_parts(
    results: Sequence[Any],
) -> PopulationEpisodeResult:
    return PopulationEpisodeResult(
        episodes=pd.concat([result.episodes for result in results], ignore_index=True),
        audit=pd.concat([result.audit for result in results], ignore_index=True),
        resolved_measurements=pd.concat(
            [result.resolved_measurements for result in results], ignore_index=True
        ),
    )


def _construct_population_chunk(
    patients: Sequence[pd.DataFrame],
) -> PopulationEpisodeResult:
    if _POPULATION_WORKER_CONFIG is None:
        raise RuntimeError("population worker was not initialized with a configuration")
    results = [
        construct_aki_episodes(patient, _POPULATION_WORKER_CONFIG) for patient in patients
    ]
    return _concat_population_parts(results)


def _patient_chunks(
    ordered: pd.DataFrame,
    subject_column: str,
    patient_chunk_size: int,
) -> Iterator[list[pd.DataFrame]]:
    chunk: list[pd.DataFrame] = []
    for _, patient in ordered.groupby(subject_column, sort=True, dropna=False):
        chunk.append(patient)
        if len(chunk) == patient_chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _construct_population_parallel(
    ordered: pd.DataFrame,
    config: Any,
    *,
    subject_column: str,
    max_workers: int,
    patient_chunk_size: int,
) -> PopulationEpisodeResult:
    worker_config = _parallel_worker_config(config)
    chunks = iter(_patient_chunks(ordered, subject_column, patient_chunk_size))
    results: list[PopulationEpisodeResult] = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_initialize_population_worker,
        initargs=(worker_config,),
    ) as executor:
        pending = deque()
        for _ in range(max_workers * 2):
            try:
                pending.append(executor.submit(_construct_population_chunk, next(chunks)))
            except StopIteration:
                break
        while pending:
            results.append(pending.popleft().result())
            try:
                pending.append(executor.submit(_construct_population_chunk, next(chunks)))
            except StopIteration:
                pass
    return _concat_population_parts(results)


def _section(config: Any, name: str) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        value = config.get(name)
    else:
        value = getattr(config, name, None)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} is required")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"{path}.{key} is required and has no default")
    return mapping[key]


def _admission_censoring_config(config: Any) -> Mapping[str, Any]:
    episodes = _section(config, "episodes")
    value = _required(episodes, "admission_censoring", "episodes")
    if not isinstance(value, Mapping):
        raise ValueError("episodes.admission_censoring must be a mapping")
    required = {
        "hadm_id_column",
        "discharge_time_column",
        "death_time_column",
        "endpoint_precedence",
        "horizon_boundary",
        "unknown_endpoint_reason",
    }
    missing = required.difference(value)
    if missing:
        raise ValueError(
            "episodes.admission_censoring is missing required keys: " f"{sorted(missing)}"
        )
    extra = set(value).difference(required)
    if extra:
        raise ValueError("episodes.admission_censoring contains unknown keys: " f"{sorted(extra)}")
    for key in (
        "hadm_id_column",
        "discharge_time_column",
        "death_time_column",
        "unknown_endpoint_reason",
    ):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"episodes.admission_censoring.{key} must be a non-empty string")
    if value["endpoint_precedence"] not in {
        "death_then_discharge",
        "discharge_then_death",
    }:
        raise ValueError(
            "episodes.admission_censoring.endpoint_precedence must be "
            "'death_then_discharge' or 'discharge_then_death'"
        )
    if value["horizon_boundary"] not in {"inclusive", "exclusive"}:
        raise ValueError(
            "episodes.admission_censoring.horizon_boundary must be " "'inclusive' or 'exclusive'"
        )
    return value


def construct_population_episodes(
    measurements: pd.DataFrame,
    config: Any,
    *,
    max_workers: int = 1,
    patient_chunk_size: int = 128,
) -> PopulationEpisodeResult:
    """Run the deterministic state machine once for every unique subject.

    No-AKI admissions remain in the returned audit table.  Censored and
    ambiguous episodes remain in both tables; they are not filtered here.
    ``max_workers`` and ``patient_chunk_size`` affect execution only.  The
    default single-worker path retains the original serial implementation;
    parallel chunks are collected in the same sorted patient order.
    """

    max_workers = _positive_execution_integer(max_workers, "max_workers")
    patient_chunk_size = _positive_execution_integer(
        patient_chunk_size, "patient_chunk_size"
    )

    episodes_config = _section(config, "episodes")
    columns = _required(episodes_config, "columns", "episodes")
    if not isinstance(columns, Mapping):
        raise ValueError("episodes.columns must be a mapping")
    subject_column = str(_required(columns, "subject_id", "episodes.columns"))
    timestamp_column = str(_required(columns, "timestamp", "episodes.columns"))
    hadm_column = str(_required(columns, "hadm_id", "episodes.columns"))
    required_columns = {subject_column, timestamp_column, hadm_column}
    missing = required_columns.difference(measurements.columns)
    if missing:
        raise ValueError(f"measurement table missing columns: {sorted(missing)}")
    if measurements[subject_column].isna().any():
        raise ValueError("population measurements contain null subject identifiers")

    ordered = measurements.copy()
    ordered["__pipeline_time"] = pd.to_datetime(ordered[timestamp_column], errors="coerce")
    if ordered["__pipeline_time"].isna().any():
        raise ValueError("population measurements contain invalid specimen timestamps")
    ordered = ordered.sort_values(
        [subject_column, hadm_column, "__pipeline_time"], kind="mergesort"
    ).drop(columns="__pipeline_time")

    if ordered.empty:
        empty = construct_aki_episodes(ordered.iloc[0:0], config)
        return PopulationEpisodeResult(
            empty.episodes,
            empty.audit,
            empty.resolved_measurements,
        )
    if max_workers > 1:
        return _construct_population_parallel(
            ordered,
            config,
            subject_column=subject_column,
            max_workers=max_workers,
            patient_chunk_size=patient_chunk_size,
        )

    episode_parts: list[pd.DataFrame] = []
    audit_parts: list[pd.DataFrame] = []
    resolved_parts: list[pd.DataFrame] = []
    for _, patient in ordered.groupby(subject_column, sort=True, dropna=False):
        result = construct_aki_episodes(patient, config)
        episode_parts.append(result.episodes)
        audit_parts.append(result.audit)
        resolved_parts.append(result.resolved_measurements)

    episodes = pd.concat(episode_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    resolved = pd.concat(resolved_parts, ignore_index=True)
    return PopulationEpisodeResult(
        episodes=episodes,
        audit=audit,
        resolved_measurements=resolved,
    )


def annotate_admission_censor_reasons(
    decisions: pd.DataFrame,
    admissions: pd.DataFrame | None,
    config: Any,
) -> pd.DataFrame:
    """Add administrative endpoint provenance to existing censor decisions.

    The episode state-machine reason is never replaced.  Discharge/death can
    only annotate rows already labeled ``status == 'censored'``.  If admission
    metadata was not supplied, the explicit status is ``not_evaluated`` rather
    than an inferred censor cause.

    ``horizon_boundary='inclusive'`` considers an endpoint exactly at the
    required observation horizon attributable; ``'exclusive'`` requires it to
    occur strictly before the horizon.
    """

    cfg = _admission_censoring_config(config)
    required = {"hadm_id", "status", "reason_code", "required_observation_end_time"}
    missing = required.difference(decisions.columns)
    if missing:
        raise ValueError(f"decision table missing censor columns: {sorted(missing)}")
    overlap = set(ADMISSION_ANNOTATION_COLUMNS).intersection(decisions.columns)
    if overlap:
        raise ValueError(
            "decision table is already admission-annotated; refusing to overwrite "
            f"columns {sorted(overlap)}"
        )

    out = decisions.copy()
    out["state_machine_reason_code"] = out["reason_code"]
    out["admission_censor_annotation_status"] = "not_applicable"
    out["admission_censor_reason"] = "not_applicable"
    out["admission_censor_endpoint_type"] = pd.NA
    out["admission_censor_endpoint_time"] = pd.NaT
    censored = out["status"].eq("censored")
    if not censored.any():
        return out

    required_horizon = pd.to_datetime(
        out.loc[censored, "required_observation_end_time"], errors="coerce"
    )
    if required_horizon.isna().any():
        indices = required_horizon.index[required_horizon.isna()].tolist()
        raise ValueError(
            "censored decisions require a valid required_observation_end_time; "
            f"invalid row indices: {indices[:10]}"
        )

    if admissions is None:
        out.loc[censored, "admission_censor_annotation_status"] = "not_evaluated"
        out.loc[censored, "admission_censor_reason"] = "not_evaluated"
        return out

    hadm_column = str(cfg["hadm_id_column"])
    discharge_column = str(cfg["discharge_time_column"])
    death_column = str(cfg["death_time_column"])
    admission_columns = {hadm_column, discharge_column, death_column}
    missing_admission_columns = admission_columns.difference(admissions.columns)
    if missing_admission_columns:
        raise ValueError(
            f"admission table missing configured columns: {sorted(missing_admission_columns)}"
        )
    metadata = admissions[[hadm_column, discharge_column, death_column]].copy()
    if metadata[hadm_column].isna().any():
        raise ValueError("admission metadata contains null configured hadm_id values")
    if metadata[hadm_column].duplicated().any():
        duplicate_ids = (
            metadata.loc[metadata[hadm_column].duplicated(keep=False), hadm_column]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(
            "admission metadata must contain one row per hadm_id; duplicates include "
            f"{duplicate_ids[:10]}"
        )
    for column in (discharge_column, death_column):
        parsed = pd.to_datetime(metadata[column], errors="coerce")
        invalid = metadata[column].notna() & parsed.isna()
        if invalid.any():
            bad = metadata.loc[invalid, hadm_column].tolist()
            raise ValueError(
                f"admission metadata column {column!r} has invalid timestamps for "
                f"hadm_id values {bad[:10]}"
            )
        metadata[column] = parsed
    metadata = metadata.set_index(hadm_column)

    precedence: Sequence[tuple[str, str]]
    if cfg["endpoint_precedence"] == "death_then_discharge":
        precedence = (("death", death_column), ("discharge", discharge_column))
    else:
        precedence = (("discharge", discharge_column), ("death", death_column))
    inclusive = cfg["horizon_boundary"] == "inclusive"
    unknown_reason = str(cfg["unknown_endpoint_reason"])

    for index in out.index[censored]:
        hadm_id = out.at[index, "hadm_id"]
        if hadm_id not in metadata.index:
            out.at[index, "admission_censor_annotation_status"] = "unknown"
            out.at[index, "admission_censor_reason"] = unknown_reason
            continue
        horizon = pd.Timestamp(out.at[index, "required_observation_end_time"])
        row = metadata.loc[hadm_id]
        qualifying: dict[str, pd.Timestamp] = {}
        observed_endpoint = False
        for endpoint_type, column in precedence:
            endpoint = row[column]
            if pd.isna(endpoint):
                continue
            observed_endpoint = True
            endpoint = pd.Timestamp(endpoint)
            inside_horizon = endpoint <= horizon if inclusive else endpoint < horizon
            if inside_horizon:
                qualifying[endpoint_type] = endpoint
        selected = next(
            (
                (endpoint_type, qualifying[endpoint_type])
                for endpoint_type, _ in precedence
                if endpoint_type in qualifying
            ),
            None,
        )
        if selected is not None:
            endpoint_type, endpoint_time = selected
            out.at[index, "admission_censor_annotation_status"] = "attributed"
            out.at[index, "admission_censor_reason"] = f"{endpoint_type}_before_required_horizon"
            out.at[index, "admission_censor_endpoint_type"] = endpoint_type
            out.at[index, "admission_censor_endpoint_time"] = endpoint_time
        elif observed_endpoint:
            out.at[index, "admission_censor_annotation_status"] = "not_attributed"
            out.at[index, "admission_censor_reason"] = (
                "administrative_endpoint_after_required_horizon"
            )
        else:
            out.at[index, "admission_censor_annotation_status"] = "unknown"
            out.at[index, "admission_censor_reason"] = unknown_reason
    return out


def annotate_population_censoring(
    population: PopulationEpisodeResult,
    admissions: pd.DataFrame | None,
    config: Any,
) -> PopulationEpisodeResult:
    """Annotate the episode audit and copy its provenance onto episode rows."""

    audit = annotate_admission_censor_reasons(population.audit, admissions, config)
    if audit["episode_id"].dropna().duplicated().any():
        raise ValueError("episode audit contains duplicate non-null episode_id values")
    annotation = audit[["episode_id", *ADMISSION_ANNOTATION_COLUMNS]].dropna(subset=["episode_id"])
    episodes = population.episodes.copy()
    overlap = set(ADMISSION_ANNOTATION_COLUMNS).intersection(episodes.columns)
    if overlap:
        raise ValueError(
            "episode table is already admission-annotated; refusing to overwrite "
            f"columns {sorted(overlap)}"
        )
    episodes = episodes.merge(annotation, on="episode_id", how="left", validate="one_to_one")
    return PopulationEpisodeResult(
        episodes=episodes,
        audit=audit,
        resolved_measurements=population.resolved_measurements,
    )


def _configured_analysis_labels(config: Any) -> tuple[set[str], set[str]]:
    matching = _section(config, "matching")
    primary = {
        str(_required(matching, "class_a", "matching")),
        str(_required(matching, "class_b", "matching")),
    }
    if primary != {"transient", "persistent"}:
        raise ValueError(
            "the locked primary audit requires matching.class_a/class_b to be "
            "transient and persistent"
        )
    metrics = _section(config, "metrics")
    multiclass = _required(metrics, "multiclass", "metrics")
    if not isinstance(multiclass, Mapping):
        raise ValueError("metrics.multiclass must be a mapping")
    labels = _required(multiclass, "labels", "metrics.multiclass")
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or len(labels) != len(set(labels))
    ):
        raise ValueError("metrics.multiclass.labels must be a unique sequence")
    secondary = {str(label) for label in labels}
    if secondary != {"transient", "persistent", "relapsing"}:
        raise ValueError(
            "the locked secondary audit requires metrics.multiclass.labels to contain "
            "transient, persistent, and relapsing"
        )
    return primary, secondary


def _events_for_labels(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    ids = labels[["episode_id", "split"]].drop_duplicates("episode_id")
    selected = events[events["episode_id"].isin(ids["episode_id"])].copy()
    if "split" in selected:
        selected = selected.drop(columns="split")
    selected = selected.merge(ids, on="episode_id", how="left", validate="many_to_one")
    if selected["split"].isna().any():
        raise AssertionError("internal error: selected events were not assigned a split")
    return selected.sort_values(
        ["episode_id", "specimen_time", "sequence_index"], kind="mergesort"
    ).reset_index(drop=True)


def _matching_input(
    primary_events: pd.DataFrame,
    primary_labels: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    tabular = build_tabular_feature_sets(primary_events, primary_labels, config)
    features = tabular["invariant_summary"].copy()
    if "split" not in features:
        features = features.merge(
            primary_labels[["episode_id", "split"]],
            on="episode_id",
            how="left",
            validate="one_to_one",
        )
    coarsening = _required(_section(config, "matching"), "coarsening", "matching")
    if not isinstance(coarsening, Mapping) or not coarsening:
        raise ValueError("matching.coarsening must be a non-empty mapping")
    required_features = {
        "baseline_creatinine_mg_dl",
        "peak_value",
        "duration_hours",
        "count",
    }
    missing_required = required_features.difference(coarsening)
    if missing_required:
        raise ValueError(
            "matching.coarsening omits locked audit-population features: "
            f"{sorted(missing_required)}"
        )
    representation_config = _section(config, "representations")
    configured_summaries = set(
        _required(representation_config, "summary_features", "representations")
    )
    allowed_features = configured_summaries | {
        "baseline_creatinine_mg_dl",
        "peak_value",
    }
    disallowed = set(coarsening).difference(allowed_features)
    if disallowed:
        raise ValueError(
            "matching features must be order-invariant summaries or locked baseline/peak; "
            f"disallowed: {sorted(disallowed)}"
        )
    missing = [feature for feature in coarsening if feature not in features]
    if missing:
        available_from_episode = [feature for feature in missing if feature in primary_labels]
        if available_from_episode:
            features = features.merge(
                primary_labels[["episode_id", *available_from_episode]],
                on="episode_id",
                how="left",
                validate="one_to_one",
            )
        still_missing = [feature for feature in coarsening if feature not in features]
        if still_missing:
            raise ValueError(
                "configured matching features are absent from both invariant summaries "
                f"and episode columns: {still_missing}"
            )
    if features[list(coarsening)].isna().any().any():
        bad = features.loc[features[list(coarsening)].isna().any(axis=1), "episode_id"].tolist()
        raise ValueError(f"configured matching features are null for episodes {bad[:10]}")
    return features


def prepare_analysis_datasets(
    measurements: pd.DataFrame,
    population: PopulationEpisodeResult,
    config: Any,
) -> AkiPipelineResult:
    """Build primary/secondary datasets and one leakage-safe patient split map."""

    primary_labels_set, secondary_labels_set = _configured_analysis_labels(config)
    resolved_config_hash = config_digest(config)
    if "config_hash" in measurements:
        observed_hashes = set(measurements["config_hash"].dropna().astype(str))
        if observed_hashes != {resolved_config_hash}:
            raise ValueError(
                "measurement config_hash does not match the preparation protocol: "
                f"observed={sorted(observed_hashes)}, expected={resolved_config_hash}"
            )
    canonical_measurements = (
        measurements
        if population.resolved_measurements is None
        else population.resolved_measurements
    )
    recovery_sensitivity = build_recovery_timing_sensitivity(
        canonical_measurements, population.episodes, config
    )
    events, all_labels = materialize_episode_events(
        canonical_measurements, population.episodes, config
    )
    all_labels = normalize_episode_labels(all_labels)

    secondary = select_first_eligible_episode(all_labels, eligible_labels=secondary_labels_set)
    if secondary.empty:
        raise ValueError("no eligible transient/persistent/relapsing episodes for splitting")
    patient_splits = make_patient_splits(secondary, config)
    validate_patient_splits(patient_splits)

    all_labels = attach_patient_splits(all_labels, patient_splits)
    events = _events_for_labels(events, all_labels)
    secondary = select_first_eligible_episode(all_labels, eligible_labels=secondary_labels_set)
    primary = select_first_eligible_episode(all_labels, eligible_labels=primary_labels_set)
    if primary.empty:
        raise ValueError("no eligible transient/persistent episodes for the primary analysis")
    secondary_events = _events_for_labels(events, secondary)
    primary_events = _events_for_labels(events, primary)

    matching_features = _matching_input(primary_events, primary, config)
    matching = coarsened_exact_match(matching_features, config)
    assignment_ids = matching.assignments[["episode_id", "pair_id"]].copy()
    if assignment_ids.empty:
        matched_labels = primary.iloc[0:0].copy()
        matched_labels["pair_id"] = pd.Series(dtype="string")
        matched_events = primary_events.iloc[0:0].copy()
        matched_events["pair_id"] = pd.Series(dtype="string")
    else:
        matched_labels = primary.merge(
            assignment_ids,
            on="episode_id",
            how="inner",
            validate="one_to_one",
        ).sort_values(["pair_id", "label"], kind="mergesort")
        matched_events = primary_events.merge(
            assignment_ids,
            on="episode_id",
            how="inner",
            validate="many_to_one",
        ).sort_values(
            ["pair_id", "episode_id", "specimen_time", "sequence_index"],
            kind="mergesort",
        )
        matched_labels = matched_labels.reset_index(drop=True)
        matched_events = matched_events.reset_index(drop=True)

    return AkiPipelineResult(
        episodes=population.episodes,
        episode_audit=population.audit,
        labeled_episode_events=events,
        patient_splits=patient_splits,
        primary_labels=primary,
        primary_events=primary_events,
        primary_matching_features=matching_features,
        primary_matched_labels=matched_labels,
        primary_matched_events=matched_events,
        secondary_labels=secondary,
        secondary_events=secondary_events,
        recovery_timing_sensitivity=recovery_sensitivity,
        matching=matching,
        config_hash=resolved_config_hash,
    )


def build_aki_audit_datasets(
    measurements: pd.DataFrame,
    config: Any,
    *,
    admissions: pd.DataFrame | None,
    max_workers: int = 1,
    patient_chunk_size: int = 128,
) -> AkiPipelineResult:
    """Construct, annotate, split, select, and match all non-model artifacts."""

    if admissions is not None and "config_hash" in admissions:
        expected_hash = config_digest(config)
        observed_hashes = set(admissions["config_hash"].dropna().astype(str))
        if observed_hashes != {expected_hash}:
            raise ValueError(
                "admission config_hash does not match the preparation protocol: "
                f"observed={sorted(observed_hashes)}, expected={expected_hash}"
            )
    population = construct_population_episodes(
        measurements,
        config,
        max_workers=max_workers,
        patient_chunk_size=patient_chunk_size,
    )
    population = annotate_population_censoring(population, admissions, config)
    return prepare_analysis_datasets(measurements, population, config)


def _censor_summary(audit: pd.DataFrame) -> pd.DataFrame:
    required = {
        "status",
        "admission_censor_annotation_status",
        "admission_censor_reason",
    }
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"annotated audit missing censor columns: {sorted(missing)}")
    censored = audit[audit["status"].eq("censored")]
    total = len(censored)
    result = (
        censored.groupby(
            ["admission_censor_annotation_status", "admission_censor_reason"],
            dropna=False,
        )
        .size()
        .rename("n_episodes")
        .reset_index()
    )
    result["fraction_of_censored_episodes"] = (
        result["n_episodes"] / total if total else float("nan")
    )
    return result


def persist_pipeline_result(
    result: AkiPipelineResult,
    output_dir: str | Path,
) -> PersistedPipelineArtifacts:
    """Persist full tables as Parquet and compact audit summaries as CSV.

    Existing artifacts are never overwritten.  This makes repeated protocol
    runs fail explicitly instead of mixing files from different configurations.
    """

    destination = Path(output_dir).expanduser().resolve()
    parquet_tables: dict[str, pd.DataFrame] = {
        "episodes": result.episodes,
        "episode_audit": result.episode_audit,
        "labeled_episode_events": result.labeled_episode_events,
        "patient_splits": result.patient_splits,
        "primary_labels": result.primary_labels,
        "primary_events": result.primary_events,
        "primary_matching_features": result.primary_matching_features,
        "primary_matched_labels": result.primary_matched_labels,
        "primary_matched_events": result.primary_matched_events,
        "secondary_labels": result.secondary_labels,
        "secondary_events": result.secondary_events,
        "recovery_timing_sensitivity": result.recovery_timing_sensitivity,
        "matching_assignments": result.matching.assignments,
        "matching_flow": result.matching.flow,
        "matching_balance": result.matching.balance,
        "protocol_metadata": pd.DataFrame(
            [{"config_hash": result.config_hash, "schema": "mimic_aki_pipeline_v1"}]
        ),
    }
    estimability = pd.DataFrame(
        [
            {
                "estimable": result.matching.estimable,
                "non_estimable_reasons_json": json.dumps(
                    list(result.matching.non_estimable_reasons)
                ),
            }
        ]
    )
    csv_tables: dict[str, pd.DataFrame] = {
        "episode_flow": build_episode_flow_report(result.episode_audit),
        "censor_reasons": _censor_summary(result.episode_audit),
        "primary_class_distribution": build_class_distribution(result.primary_labels),
        "secondary_class_distribution": build_class_distribution(result.secondary_labels),
        "recovery_timing_sensitivity": (
            result.recovery_timing_sensitivity.groupby(
                ["sensitivity_status", "recovery_timing_category", "sensitivity_reason_code"],
                dropna=False,
            )
            .size()
            .rename("n_episodes")
            .reset_index()
        ),
        "matching_flow": result.matching.flow,
        "matching_balance": result.matching.balance,
        "matching_estimability": estimability,
        "protocol_metadata": pd.DataFrame(
            [{"config_hash": result.config_hash, "schema": "mimic_aki_pipeline_v1"}]
        ),
    }
    parquet_paths = {name: destination / f"{name}.parquet" for name in parquet_tables}
    csv_paths = {name: destination / f"{name}.csv" for name in csv_tables}
    existing = [path for path in [*parquet_paths.values(), *csv_paths.values()] if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing pipeline artifacts: "
            f"{[str(path) for path in existing[:10]]}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in parquet_tables.items():
        table.to_parquet(parquet_paths[name], index=False)
    for name, table in csv_tables.items():
        table.to_csv(csv_paths[name], index=False)
    return PersistedPipelineArtifacts(parquet=parquet_paths, csv_summaries=csv_paths)
