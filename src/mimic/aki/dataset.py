"""Episode-level dataset materialization for the AKI audit."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def _section(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        value = config.get("representations", config)
    else:
        value = getattr(config, "representations", None)
    if not isinstance(value, Mapping):
        raise ValueError("configuration section 'representations' is required")
    return value


def _required(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"representations.{key} is required and has no default")
    return cfg[key]


def normalize_episode_labels(episodes: pd.DataFrame) -> pd.DataFrame:
    """Expose consistent model-facing names without discarding audit columns."""

    out = episodes.copy()
    if "phenotype" in out and "label" not in out:
        out = out.rename(columns={"phenotype": "label"})
    if "baseline_value" in out and "baseline_creatinine_mg_dl" not in out:
        out = out.rename(columns={"baseline_value": "baseline_creatinine_mg_dl"})
    required = {
        "episode_id",
        "subject_id",
        "hadm_id",
        "label",
        "onset_time",
        "baseline_creatinine_mg_dl",
    }
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"episode table missing model-facing columns: {sorted(missing)}")
    return out


def materialize_episode_events(
    measurements: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice cleaned measurements into each configured episode input window.

    Returns ``(events, labels)``.  Only explicitly included, non-null phenotype
    rows are materialized; censored and ambiguous rows remain available in the
    episode audit but cannot enter a model silently.
    """

    cfg = _section(config)
    start_column = str(_required(cfg, "window_start_column"))
    end_column = str(_required(cfg, "window_end_column"))
    start_boundary = str(_required(cfg, "window_start_boundary"))
    end_boundary = str(_required(cfg, "window_end_boundary"))
    if start_boundary not in {"inclusive", "exclusive"}:
        raise ValueError("representations.window_start_boundary must be inclusive or exclusive")
    if end_boundary not in {"inclusive", "exclusive"}:
        raise ValueError("representations.window_end_boundary must be inclusive or exclusive")
    required_measurement_columns = {
        "subject_id",
        "hadm_id",
        "specimen_time",
        "creatinine_mg_dl",
    }
    missing_measurements = required_measurement_columns.difference(measurements.columns)
    if missing_measurements:
        raise ValueError(f"measurement table missing columns: {sorted(missing_measurements)}")
    labels = normalize_episode_labels(episodes)
    if start_column not in labels or end_column not in labels:
        raise ValueError(
            f"episode table must contain configured window columns {start_column!r} and "
            f"{end_column!r}"
        )
    if "include_in_analysis" not in labels:
        raise ValueError("episode table must contain include_in_analysis")
    labels = labels[labels["include_in_analysis"].astype(bool) & labels["label"].notna()].copy()
    for column in ("onset_time", start_column, end_column):
        labels[column] = pd.to_datetime(labels[column], errors="coerce")
    if labels[[start_column, end_column]].isna().any().any():
        raise ValueError("included episodes have null/invalid representation window bounds")
    if (labels[end_column] < labels[start_column]).any():
        raise ValueError("an included episode has a negative representation window")

    labs = measurements.copy()
    labs["specimen_time"] = pd.to_datetime(labs["specimen_time"], errors="coerce")
    labs["creatinine_mg_dl"] = pd.to_numeric(labs["creatinine_mg_dl"], errors="coerce")
    if labs[["specimen_time", "creatinine_mg_dl"]].isna().any().any():
        raise ValueError("clean measurement table contains invalid timestamp/value")

    rows: list[pd.DataFrame] = []
    grouped = {
        (subject_id, hadm_id): group.sort_values("specimen_time", kind="mergesort")
        for (subject_id, hadm_id), group in labs.groupby(["subject_id", "hadm_id"], sort=False)
    }
    for episode in labels.itertuples(index=False):
        key = (episode.subject_id, episode.hadm_id)
        admission_labs = grouped.get(key)
        if admission_labs is None:
            raise ValueError(
                f"no cleaned measurements found for included episode {episode.episode_id}"
            )
        start = getattr(episode, start_column)
        end = getattr(episode, end_column)
        if start_boundary == "inclusive":
            after_start = admission_labs["specimen_time"] >= start
        else:
            after_start = admission_labs["specimen_time"] > start
        if end_boundary == "inclusive":
            before_end = admission_labs["specimen_time"] <= end
        else:
            before_end = admission_labs["specimen_time"] < end
        selected = admission_labs[after_start & before_end].copy()
        if selected.empty:
            raise ValueError(f"configured input window is empty for episode {episode.episode_id}")
        selected["episode_id"] = episode.episode_id
        selected["sequence_index"] = range(len(selected))
        rows.append(selected)
    event_columns = [
        "episode_id",
        "subject_id",
        "hadm_id",
        "sequence_index",
        "specimen_time",
        "creatinine_mg_dl",
    ]
    if rows:
        events = pd.concat(rows, ignore_index=True)
        extra = [column for column in events.columns if column not in event_columns]
        events = events[event_columns + extra]
    else:
        events = pd.DataFrame(columns=event_columns)
    return events, labels.reset_index(drop=True)
