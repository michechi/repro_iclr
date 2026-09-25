"""Coarsened-exact matching for the order-identifiable AKI estimand."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ASSIGNMENT_COLUMNS = [
    "pair_id",
    "split",
    "stratum",
    "episode_id",
    "subject_id",
    "label",
]
FLOW_COLUMNS = [
    "split",
    "label",
    "n_input",
    "n_matched",
    "n_unmatched",
    "retained_fraction",
]
BALANCE_COLUMNS = [
    "split",
    "feature",
    "smd_before",
    "smd_after",
    "maximum_absolute_smd",
    "passes_threshold",
]


@dataclass(frozen=True)
class MatchingResult:
    matched: pd.DataFrame
    assignments: pd.DataFrame
    flow: pd.DataFrame
    balance: pd.DataFrame
    estimable: bool
    non_estimable_reasons: tuple[str, ...]


def _section(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        value = config.get("matching", config)
    else:
        value = getattr(config, "matching", None)
    if not isinstance(value, Mapping):
        raise ValueError("configuration section 'matching' is required")
    return value


def _required(cfg: Mapping[str, Any], key: str) -> Any:
    if key not in cfg or cfg[key] is None:
        raise ValueError(f"matching.{key} is required and has no default")
    return cfg[key]


def _nonnegative_finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if result < 0:
        raise ValueError(f"{path} must be >= 0")
    return result


def _maximum_smd_thresholds(
    cfg: Mapping[str, Any], features: Sequence[str]
) -> dict[str, float]:
    default = _nonnegative_finite_number(
        _required(cfg, "maximum_absolute_smd"), "matching.maximum_absolute_smd"
    )
    raw_overrides = cfg.get("maximum_absolute_smd_overrides", {})
    if not isinstance(raw_overrides, Mapping):
        raise ValueError("matching.maximum_absolute_smd_overrides must be a mapping")
    unknown = set(raw_overrides).difference(features)
    if unknown:
        raise ValueError(
            "matching.maximum_absolute_smd_overrides keys must be configured "
            f"matching.coarsening features; unknown: {sorted(unknown)}"
        )
    overrides = {
        feature: _nonnegative_finite_number(
            value, f"matching.maximum_absolute_smd_overrides.{feature}"
        )
        for feature, value in raw_overrides.items()
    }
    return {feature: overrides.get(feature, default) for feature in features}


def select_first_eligible_episode(
    episodes: pd.DataFrame,
    *,
    eligible_labels: set[str],
) -> pd.DataFrame:
    """Return each patient's chronologically first non-censored target episode."""

    required = {"subject_id", "episode_id", "onset_time", "label"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"episodes missing columns: {sorted(missing)}")
    eligible = episodes[episodes["label"].isin(eligible_labels)].copy()
    eligible["onset_time"] = pd.to_datetime(eligible["onset_time"], errors="coerce")
    if eligible["onset_time"].isna().any():
        raise ValueError("eligible episodes contain invalid onset_time")
    return (
        eligible.sort_values(["subject_id", "onset_time", "episode_id"], kind="mergesort")
        .drop_duplicates("subject_id", keep="first")
        .reset_index(drop=True)
    )


def _stable_score(seed: int, episode_id: Any) -> int:
    payload = f"{seed}:{episode_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _coarsen(series: pd.Series, spec: Mapping[str, Any], feature: str) -> pd.Series:
    method = spec.get("method")
    if method == "exact":
        return series.astype("string")
    if method == "edges":
        edges = spec.get("edges")
        right = spec.get("right")
        if (
            not isinstance(edges, Sequence)
            or isinstance(edges, (str, bytes))
            or len(edges) < 2
            or right is None
        ):
            raise ValueError(f"matching.coarsening.{feature} edges method requires edges and right")
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"matching feature {feature} contains non-numeric/null values")
        cut = pd.cut(
            numeric,
            bins=np.asarray(edges, dtype=float),
            right=bool(right),
            include_lowest=True,
        )
        if cut.isna().any():
            bad = numeric[cut.isna()].head().tolist()
            raise ValueError(
                f"matching.coarsening.{feature} edges do not cover values such as {bad}"
            )
        return cut.astype("string")
    if method == "width":
        width = spec.get("width")
        origin = spec.get("origin")
        if width is None or origin is None or float(width) <= 0:
            raise ValueError(
                f"matching.coarsening.{feature} width method requires positive width and origin"
            )
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"matching feature {feature} contains non-numeric/null values")
        return np.floor((numeric - float(origin)) / float(width)).astype("int64").astype("string")
    raise ValueError(f"matching.coarsening.{feature}.method must be exact, edges, or width")


def _standardized_difference(frame: pd.DataFrame, feature: str, a: str, b: str) -> float:
    left = pd.to_numeric(frame.loc[frame["label"] == a, feature], errors="coerce").dropna()
    right = pd.to_numeric(frame.loc[frame["label"] == b, feature], errors="coerce").dropna()
    if left.empty or right.empty:
        return float("nan")
    pooled = np.sqrt((left.var(ddof=0) + right.var(ddof=0)) / 2.0)
    difference = float(left.mean() - right.mean())
    if pooled == 0:
        return 0.0 if difference == 0 else float("inf")
    return difference / float(pooled)


def _balance_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    thresholds: Mapping[str, float],
    class_a: str,
    class_b: str,
) -> pd.DataFrame:
    rows = []
    for split in sorted(before["split"].unique()):
        pre = before[before["split"] == split]
        post = after[after["split"] == split]
        for feature, threshold in thresholds.items():
            smd_after = _standardized_difference(post, feature, class_a, class_b)
            rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "smd_before": _standardized_difference(pre, feature, class_a, class_b),
                    "smd_after": smd_after,
                    "maximum_absolute_smd": threshold,
                    "passes_threshold": bool(
                        np.isfinite(smd_after) and abs(smd_after) <= threshold
                    ),
                }
            )
    return pd.DataFrame(rows, columns=BALANCE_COLUMNS)


def coarsened_exact_match(features: pd.DataFrame, config: Any) -> MatchingResult:
    """Create deterministic within-split 1:1 matches without replacement."""

    cfg = _section(config)
    class_a = str(_required(cfg, "class_a"))
    class_b = str(_required(cfg, "class_b"))
    seed = int(_required(cfg, "selection_seed"))
    coarsening = _required(cfg, "coarsening")
    min_pairs = _required(cfg, "minimum_pairs_per_split")
    if not isinstance(coarsening, Mapping) or not coarsening:
        raise ValueError(
            "matching.coarsening must map each invariant feature to a bin specification"
        )
    if not isinstance(min_pairs, Mapping):
        raise ValueError("matching.minimum_pairs_per_split must map split names to counts")
    smd_thresholds = _maximum_smd_thresholds(cfg, list(coarsening))

    required_columns = {"episode_id", "subject_id", "label", "split", *coarsening.keys()}
    missing = required_columns.difference(features.columns)
    if missing:
        raise ValueError(f"matching input missing columns: {sorted(missing)}")
    before = features[features["label"].isin({class_a, class_b})].copy().reset_index(drop=True)
    if before["subject_id"].duplicated().any():
        raise ValueError(
            "primary matching expects one eligible episode per subject; call "
            "select_first_eligible_episode first"
        )
    if before.empty:
        return MatchingResult(
            matched=before,
            assignments=pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
            flow=pd.DataFrame(columns=FLOW_COLUMNS),
            balance=pd.DataFrame(columns=BALANCE_COLUMNS),
            estimable=False,
            non_estimable_reasons=("no eligible primary episodes",),
        )

    bin_columns: list[str] = []
    for feature, spec in coarsening.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"matching.coarsening.{feature} must be a mapping")
        column = f"__cem_{feature}"
        before[column] = _coarsen(before[feature], spec, feature)
        bin_columns.append(column)
    before["__stratum"] = before[bin_columns].astype(str).agg("||".join, axis=1)
    before["__stable_score"] = before["episode_id"].map(lambda value: _stable_score(seed, value))

    assignment_rows: list[dict[str, Any]] = []
    matched_indices: list[int] = []
    for (split, stratum), group in before.groupby(["split", "__stratum"], sort=True):
        left = group[group["label"] == class_a].sort_values("__stable_score")
        right = group[group["label"] == class_b].sort_values("__stable_score")
        n_pairs = min(len(left), len(right))
        for pair_number in range(n_pairs):
            pair_id = f"{split}:{stratum}:{pair_number}"
            for member in (left.iloc[pair_number], right.iloc[pair_number]):
                matched_indices.append(int(member.name))
                assignment_rows.append(
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "stratum": stratum,
                        "episode_id": member["episode_id"],
                        "subject_id": member["subject_id"],
                        "label": member["label"],
                    }
                )

    assignments = pd.DataFrame(assignment_rows, columns=ASSIGNMENT_COLUMNS)
    matched = before.loc[matched_indices].copy() if matched_indices else before.iloc[0:0].copy()
    matched = matched.drop(columns=bin_columns + ["__stratum", "__stable_score"])

    flow_rows = []
    for split in sorted(before["split"].unique()):
        for label in (class_a, class_b):
            n_input = int(((before["split"] == split) & (before["label"] == label)).sum())
            n_matched = int(((matched["split"] == split) & (matched["label"] == label)).sum())
            flow_rows.append(
                {
                    "split": split,
                    "label": label,
                    "n_input": n_input,
                    "n_matched": n_matched,
                    "n_unmatched": n_input - n_matched,
                    "retained_fraction": n_matched / n_input if n_input else float("nan"),
                }
            )
    flow = pd.DataFrame(flow_rows, columns=FLOW_COLUMNS)
    balance = _balance_table(before, matched, smd_thresholds, class_a, class_b)
    if balance.empty:
        balance = pd.DataFrame(columns=BALANCE_COLUMNS)

    reasons: list[str] = []
    for split, minimum in min_pairs.items():
        observed = (
            0
            if assignments.empty
            else assignments.loc[assignments["split"] == split, "pair_id"].nunique()
        )
        if observed < int(minimum):
            reasons.append(
                f"{split} has {observed} matched pairs; configured minimum is {int(minimum)}"
            )
    if not balance.empty:
        failed = balance[~balance["passes_threshold"]]
        if not failed.empty:
            details = ", ".join(
                f"{row.split}/{row.feature}={row.smd_after:.3g} "
                f"(limit={row.maximum_absolute_smd:.3g})"
                for row in failed.itertuples()
            )
            reasons.append(
                "post-match balance exceeds maximum_absolute_smd or is non-finite: "
                f"{details}"
            )

    return MatchingResult(
        matched=matched.reset_index(drop=True),
        assignments=assignments.reset_index(drop=True),
        flow=flow,
        balance=balance,
        estimable=not reasons,
        non_estimable_reasons=tuple(reasons),
    )
