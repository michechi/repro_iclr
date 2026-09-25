"""Representations and perturbation controls for the AKI trajectory audit.

The functions in this module deliberately do not select scientific defaults.
Every feature, transform, time encoding, and truncation rule is supplied by the
protocol configuration.  Sequence perturbations move creatinine values over a
fixed timestamp grid; timestamps and multiplicities are never changed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_EVENT_COLUMNS = {
    "episode_id",
    "subject_id",
    "hadm_id",
    "specimen_time",
    "creatinine_mg_dl",
}


@dataclass(frozen=True)
class RepresentationBundle:
    """All model inputs produced from one episode-event table."""

    ordered: list[dict[str, Any]]
    shuffled: list[dict[str, Any]]
    reversed: list[dict[str, Any]]
    tabular: dict[str, pd.DataFrame]


def _section(config: Any, name: str) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        section = config.get(name, config if name == "representations" else None)
    else:
        section = getattr(config, name, None)
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration section '{name}' is required")
    return section


def _required(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"representations.{key} is required and has no default")
    return cfg[key]


def _stable_group_seed(seed: int, group_value: Any) -> int:
    digest = hashlib.blake2b(str(group_value).encode("utf-8"), digest_size=8).digest()
    group_seed = int.from_bytes(digest, "little", signed=False)
    return int((int(seed) + group_seed) % np.iinfo(np.uint32).max)


def _validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"episode event table is missing columns: {sorted(missing)}")
    out = events.copy()
    out["specimen_time"] = pd.to_datetime(out["specimen_time"], errors="coerce")
    out["creatinine_mg_dl"] = pd.to_numeric(out["creatinine_mg_dl"], errors="coerce")
    if out[["specimen_time", "creatinine_mg_dl"]].isna().any().any():
        raise ValueError("episode events contain null/non-numeric timestamps or values")
    return out.sort_values(["episode_id", "specimen_time"], kind="mergesort").reset_index(drop=True)


def shuffle_values_preserve_timestamps(
    events: pd.DataFrame,
    seed: int,
    *,
    group_col: str = "episode_id",
    value_col: str = "creatinine_mg_dl",
) -> pd.DataFrame:
    """Permute values independently within groups while retaining every row/time.

    A stable hash makes a given episode's permutation independent of DataFrame
    group iteration order.  Duplicate values remain duplicate values.
    """

    if group_col not in events or value_col not in events:
        raise ValueError(f"missing {group_col!r} or {value_col!r}")
    out = events.copy()
    for group_value, idx in out.groupby(group_col, sort=False).groups.items():
        positions = list(idx)
        values = out.loc[positions, value_col].to_numpy(copy=True)
        rng = np.random.default_rng(_stable_group_seed(seed, group_value))
        out.loc[positions, value_col] = values[rng.permutation(len(values))]
    return out


def reverse_values_preserve_timestamps(
    events: pd.DataFrame,
    *,
    group_col: str = "episode_id",
    value_col: str = "creatinine_mg_dl",
) -> pd.DataFrame:
    """Reverse values within each episode without reversing timestamp rows."""

    if group_col not in events or value_col not in events:
        raise ValueError(f"missing {group_col!r} or {value_col!r}")
    out = events.copy()
    for _, idx in out.groupby(group_col, sort=False).groups.items():
        positions = list(idx)
        out.loc[positions, value_col] = out.loc[positions, value_col].to_numpy()[::-1]
    return out


def assert_value_multisets_preserved(
    original: pd.DataFrame,
    perturbed: pd.DataFrame,
    *,
    group_col: str = "episode_id",
    value_col: str = "creatinine_mg_dl",
    time_col: str = "specimen_time",
) -> None:
    """Raise if a perturbation changes rows, timestamp grids, or multiplicities."""

    if len(original) != len(perturbed):
        raise AssertionError("perturbation changed observation count")
    original_groups = set(original[group_col])
    if original_groups != set(perturbed[group_col]):
        raise AssertionError("perturbation changed episode identifiers")
    for episode_id in original_groups:
        left = original.loc[original[group_col] == episode_id].sort_values(time_col)
        right = perturbed.loc[perturbed[group_col] == episode_id].sort_values(time_col)
        if left[time_col].tolist() != right[time_col].tolist():
            raise AssertionError(f"timestamp grid changed for episode {episode_id}")
        if Counter(left[value_col].tolist()) != Counter(right[value_col].tolist()):
            raise AssertionError(f"value multiset changed for episode {episode_id}")


def _truncate(group: pd.DataFrame, max_length: int, direction: str) -> pd.DataFrame:
    if max_length <= 0:
        raise ValueError("representations.max_length must be positive")
    if len(group) <= max_length:
        return group
    if direction == "head":
        return group.head(max_length)
    if direction == "tail":
        return group.tail(max_length)
    if direction == "error":
        raise ValueError(
            f"episode {group['episode_id'].iloc[0]} has {len(group)} observations, "
            f"exceeding max_length={max_length}"
        )
    raise ValueError("representations.truncation must be 'head', 'tail', or 'error'")


def _truncate_events_for_all_controls(events: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Select one common sequence window before any value perturbation.

    This is deliberately limited to sequence conditions.  The invariant,
    count-only, and simple positional controls summarize the complete
    configured episode window, so their observation count is not capped by a
    neural batching constraint.
    """

    max_length = int(_required(cfg, "max_length"))
    direction = str(_required(cfg, "truncation"))
    parts = [
        _truncate(group, max_length, direction)
        for _, group in events.groupby("episode_id", sort=False)
    ]
    if not parts:
        return events.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def _transform_values(
    values: np.ndarray,
    label_row: Mapping[str, Any],
    transform: str,
) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    if transform == "raw":
        return values
    if transform == "log":
        if np.any(values <= 0):
            raise ValueError("log transform requires positive creatinine values")
        return np.log(values)
    if "baseline_creatinine_mg_dl" not in label_row:
        raise ValueError(f"value transform {transform!r} requires baseline_creatinine_mg_dl")
    baseline = float(label_row["baseline_creatinine_mg_dl"])
    if transform == "delta_from_baseline":
        return values - baseline
    if transform == "ratio_to_baseline":
        if baseline <= 0:
            raise ValueError("ratio_to_baseline requires a positive baseline")
        return values / baseline
    raise ValueError(
        "representations.value_transform must be raw, delta_from_baseline, "
        "ratio_to_baseline, or log"
    )


def _sequence_record(
    group: pd.DataFrame,
    label_row: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    max_length = int(_required(cfg, "max_length"))
    truncation = str(_required(cfg, "truncation"))
    group = _truncate(group.sort_values("specimen_time", kind="mergesort"), max_length, truncation)

    times = pd.to_datetime(group["specimen_time"])
    first_time = times.iloc[0]
    elapsed_hours = (times - first_time).dt.total_seconds().to_numpy() / 3600.0
    delta_hours = np.diff(elapsed_hours, prepend=elapsed_hours[0])
    values = _transform_values(
        group["creatinine_mg_dl"].to_numpy(dtype=np.float32),
        label_row,
        str(_required(cfg, "value_transform")),
    )

    channel_names: list[str] = ["creatinine"]
    channels: list[np.ndarray] = [values]
    requested_time_features = list(_required(cfg, "time_features"))
    for feature in requested_time_features:
        if feature == "elapsed_hours":
            channel_names.append(feature)
            channels.append(elapsed_hours.astype(np.float32))
        elif feature == "delta_hours":
            channel_names.append(feature)
            channels.append(delta_hours.astype(np.float32))
        elif feature == "time_since_onset_hours":
            if "onset_time" not in label_row:
                raise ValueError("time_since_onset_hours requires onset_time")
            onset = pd.Timestamp(label_row["onset_time"])
            since = (times - onset).dt.total_seconds().to_numpy() / 3600.0
            channel_names.append(feature)
            channels.append(since.astype(np.float32))
        else:
            raise ValueError(f"unsupported configured time feature: {feature}")

    features = np.column_stack(channels).astype(np.float32)
    return {
        "episode_id": group["episode_id"].iloc[0],
        "subject_id": int(group["subject_id"].iloc[0]),
        "hadm_id": int(group["hadm_id"].iloc[0]),
        "label": label_row["label"],
        "timestamps": times.tolist(),
        "values_mg_dl": group["creatinine_mg_dl"].astype(float).tolist(),
        "features": features,
        "channel_names": channel_names,
        "length": int(len(group)),
    }


def build_sequence_records(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    config: Any,
) -> list[dict[str, Any]]:
    """Convert a long event table into variable-length numeric sequences."""

    events = _validate_events(events)
    cfg = _section(config, "representations")
    if "episode_id" not in labels or "label" not in labels:
        raise ValueError("labels must contain episode_id and label")
    if labels["episode_id"].duplicated().any():
        raise ValueError("labels must have one row per episode_id")
    label_map = labels.set_index("episode_id").to_dict(orient="index")
    records: list[dict[str, Any]] = []
    for episode_id, group in events.groupby("episode_id", sort=False):
        if episode_id not in label_map:
            continue
        records.append(_sequence_record(group, label_map[episode_id], cfg))
    return records


def _quantile_name(q: float) -> str:
    return f"q{int(round(100 * q)):02d}"


def _summary_row(group: pd.DataFrame, requested: Sequence[str]) -> dict[str, float]:
    values = group["creatinine_mg_dl"].to_numpy(dtype=float)
    times = pd.to_datetime(group["specimen_time"])
    elapsed = (times - times.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    gaps = np.diff(elapsed)
    supported: dict[str, float] = {
        "count": float(len(values)),
        "distinct_count": float(len(np.unique(values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "median": float(np.median(values)),
        "range": float(np.max(values) - np.min(values)),
        "repeated_min_count": float(np.sum(values == np.min(values))),
        "repeated_max_count": float(np.sum(values == np.max(values))),
        "duration_hours": float(elapsed[-1]) if len(elapsed) else 0.0,
        "gap_mean_hours": float(np.mean(gaps)) if len(gaps) else 0.0,
        "gap_std_hours": float(np.std(gaps, ddof=0)) if len(gaps) else 0.0,
        "gap_min_hours": float(np.min(gaps)) if len(gaps) else 0.0,
        "gap_max_hours": float(np.max(gaps)) if len(gaps) else 0.0,
    }
    for q in (0.05, 0.10, 0.25, 0.75, 0.90, 0.95):
        supported[_quantile_name(q)] = float(np.quantile(values, q))
    unsupported = set(requested).difference(supported)
    if unsupported:
        raise ValueError(f"unsupported invariant summary features: {sorted(unsupported)}")
    return {name: supported[name] for name in requested}


def _linear_slope(group: pd.DataFrame, hours_per_unit: float) -> float:
    if hours_per_unit <= 0:
        raise ValueError("representations.slope_hours_per_unit must be positive")
    values = group["creatinine_mg_dl"].to_numpy(dtype=float)
    times = pd.to_datetime(group["specimen_time"])
    x = (times - times.iloc[0]).dt.total_seconds().to_numpy() / 3600.0 / hours_per_unit
    centered = x - x.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return float("nan")
    return float(np.dot(centered, values - values.mean()) / denominator)


def build_tabular_feature_sets(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    config: Any,
) -> dict[str, pd.DataFrame]:
    """Build invariant and simple positional control tables.

    Every table retains identifiers and the label.  Feature columns are kept
    separate so callers cannot accidentally mix invariant and positional data.
    """

    events = _validate_events(events)
    cfg = _section(config, "representations")
    requested = list(_required(cfg, "summary_features"))
    slope_hours_per_unit = float(_required(cfg, "slope_hours_per_unit"))
    label_columns = [c for c in ("episode_id", "subject_id", "hadm_id", "label") if c in labels]
    if "episode_id" not in label_columns or "label" not in label_columns:
        raise ValueError("labels must contain episode_id and label")
    label_base = labels[label_columns].drop_duplicates("episode_id")
    if labels["episode_id"].duplicated().any():
        raise ValueError("labels must have one row per episode_id")
    label_map = labels.set_index("episode_id").to_dict(orient="index")
    transform = str(_required(cfg, "value_transform"))

    invariant_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for episode_id, group in events.groupby("episode_id", sort=False):
        group = group.sort_values("specimen_time", kind="mergesort")
        if episode_id not in label_map:
            continue
        transformed = group.copy()
        transformed["creatinine_mg_dl"] = _transform_values(
            group["creatinine_mg_dl"].to_numpy(dtype=np.float32),
            label_map[episode_id],
            transform,
        )
        inv = {"episode_id": episode_id}
        inv.update(_summary_row(transformed, requested))
        invariant_rows.append(inv)
        first = float(transformed["creatinine_mg_dl"].iloc[0])
        last = float(transformed["creatinine_mg_dl"].iloc[-1])
        position_rows.append(
            {
                "episode_id": episode_id,
                "first_value": first,
                "last_value": last,
                "first_to_last_change": last - first,
                "linear_slope": _linear_slope(transformed, slope_hours_per_unit),
            }
        )

    invariant = label_base.merge(pd.DataFrame(invariant_rows), on="episode_id", how="inner")
    position = label_base.merge(pd.DataFrame(position_rows), on="episode_id", how="inner")
    identifiers = label_columns

    result: dict[str, pd.DataFrame] = {"invariant_summary": invariant}
    if "count" not in invariant.columns:
        count_rows = (
            events.groupby("episode_id", as_index=False).size().rename(columns={"size": "count"})
        )
        result["count_only"] = label_base.merge(count_rows, on="episode_id", how="inner")
    else:
        result["count_only"] = invariant[identifiers + ["count"]].copy()
    for feature in ("first_value", "last_value", "first_to_last_change", "linear_slope"):
        result[feature] = position[identifiers + [feature]].copy()
    result["position_combined"] = position.copy()
    return result


def build_representations(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    config: Any,
    *,
    shuffle_seed: int,
) -> RepresentationBundle:
    """Create the complete ordered/shuffled/reversed control bundle."""

    cfg = _section(config, "representations")
    full_window_events = _validate_events(events)
    ordered_events = _truncate_events_for_all_controls(full_window_events, cfg)
    shuffled_events = shuffle_values_preserve_timestamps(ordered_events, shuffle_seed)
    reversed_events = reverse_values_preserve_timestamps(ordered_events)
    assert_value_multisets_preserved(ordered_events, shuffled_events)
    assert_value_multisets_preserved(ordered_events, reversed_events)
    return RepresentationBundle(
        ordered=build_sequence_records(ordered_events, labels, config),
        shuffled=build_sequence_records(shuffled_events, labels, config),
        reversed=build_sequence_records(reversed_events, labels, config),
        tabular=build_tabular_feature_sets(full_window_events, labels, config),
    )
