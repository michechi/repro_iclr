"""Metrics and paired uncertainty estimates for the AKI trajectory audit.

The functions in this module deliberately require the protocol to state how
missing classes are handled.  In particular, an AUROC is never silently
reported as zero (or silently dropped) when a validation/test split contains
only one side of a one-vs-rest comparison.

Accepted configuration
----------------------
Callers may pass either the full ``AkiAuditConfig``, its ``metrics`` mapping,
or the relevant nested mapping.  The complete metrics mapping has this shape::

    binary:
      labels: [0, 1]
      positive_label: 1
      threshold: 0.5
      missing_class_policy: raise  # raise | nan | skip
      zero_division: 0             # 0 | 1 | warn
    multiclass:
      labels: [0, 1, 2]
      decision_rule: argmax
      missing_class_policy: raise  # raise | nan | skip
      zero_division: 0             # 0 | 1 | warn
    aggregation:
      ddof: 1
    bootstrap:
      n_resamples: 2000
      confidence_level: 0.95
      seed: 123
      interval_method: percentile
      missing_class_policy: skip   # raise | skip

The values above illustrate the schema only; this module supplies none of them
as production defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


MissingClassPolicy = str


def _metrics_section(config: Any) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        section = config.get("metrics", config)
    else:
        section = getattr(config, "metrics", None)
    if not isinstance(section, Mapping):
        raise ValueError("configuration section 'metrics' is required")
    return section


def _subsection(config: Any, name: str) -> Mapping[str, Any]:
    section = _metrics_section(config)
    if name in section:
        value = section[name]
    else:
        # Permit a nested section to be passed directly while still rejecting
        # unrelated mappings with an actionable error below.
        value = section
    if not isinstance(value, Mapping):
        raise ValueError(f"metrics.{name} must be a mapping")
    return value


def _required(config: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in config or config[key] is None:
        raise ValueError(f"{path}.{key} is required and has no default")
    return config[key]


def _validate_policy(value: Any, *, bootstrap: bool = False) -> MissingClassPolicy:
    policy = str(value)
    allowed = {"raise", "skip"} if bootstrap else {"raise", "nan", "skip"}
    if policy not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"missing_class_policy must be one of: {choices}")
    return policy


def _validate_labels(values: Any, *, expected_size: int | None = None) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("configured labels must be a sequence")
    labels = tuple(values)
    if expected_size is not None and len(labels) != expected_size:
        raise ValueError(f"configured labels must contain exactly {expected_size} values")
    if len(labels) < 2:
        raise ValueError("configured labels must contain at least two values")
    if len(set(labels)) != len(labels):
        raise ValueError("configured labels must be unique")
    return labels


def _as_1d(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _validate_probability_matrix(probabilities: Any, n_rows: int) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2 or array.shape[0] != n_rows:
        raise ValueError(
            "probabilities must have shape [n_samples, n_classes], "
            f"got {array.shape} for {n_rows} samples"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities contain non-finite values")
    tolerance = 1e-6
    if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(array.sum(axis=1), 1.0, atol=tolerance, rtol=tolerance):
        raise ValueError("each probability row must sum to one")
    return array


def _probability_columns(
    probabilities: np.ndarray,
    configured_labels: tuple[Any, ...],
    probability_labels: Sequence[Any] | None,
) -> np.ndarray:
    if probability_labels is None:
        columns = configured_labels
    else:
        columns = tuple(probability_labels)
    if len(columns) != probabilities.shape[1] or len(set(columns)) != len(columns):
        raise ValueError("probability_labels must uniquely name every probability column")
    missing = set(configured_labels).difference(columns)
    extra = set(columns).difference(configured_labels)
    if missing or extra:
        raise ValueError(
            "probability labels do not match configured labels; "
            f"missing={sorted(missing, key=str)}, extra={sorted(extra, key=str)}"
        )
    positions = [columns.index(label) for label in configured_labels]
    return probabilities[:, positions]


def _sklearn_metrics() -> tuple[Callable[..., float], Callable[..., float], Callable[..., float]]:
    try:
        from sklearn.metrics import f1_score, recall_score, roc_auc_score
    except ImportError as exc:  # pragma: no cover - dependency failure path
        raise ImportError(
            "AKI metric reporting requires scikit-learn; install the project dependencies"
        ) from exc
    return roc_auc_score, f1_score, recall_score


def _missing_value(policy: MissingClassPolicy, message: str) -> float | None:
    if policy == "raise":
        raise ValueError(message)
    if policy == "nan":
        return float("nan")
    if policy == "skip":
        return None
    raise AssertionError(f"unvalidated missing-class policy: {policy}")


def evaluate_binary(
    y_true: Any,
    probabilities: Any,
    config: Any,
    *,
    probability_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Report binary AUROC, F1, and recall under an explicit protocol.

    ``probabilities`` may be a one-dimensional score for the configured
    positive label, or a two-column probability matrix.  For a matrix,
    ``probability_labels`` states its column order; when omitted, columns must
    already follow ``metrics.binary.labels``.
    """

    cfg = _subsection(config, "binary")
    labels = _validate_labels(_required(cfg, "labels", "metrics.binary"), expected_size=2)
    positive_label = _required(cfg, "positive_label", "metrics.binary")
    if positive_label not in labels:
        raise ValueError("metrics.binary.positive_label must occur in metrics.binary.labels")
    negative_label = labels[0] if labels[1] == positive_label else labels[1]
    threshold = float(_required(cfg, "threshold", "metrics.binary"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("metrics.binary.threshold must lie in [0, 1]")
    policy = _validate_policy(_required(cfg, "missing_class_policy", "metrics.binary"))
    zero_division = _required(cfg, "zero_division", "metrics.binary")
    if zero_division not in (0, 1, "warn"):
        raise ValueError("metrics.binary.zero_division must be 0, 1, or 'warn'")

    truth = _as_1d(y_true, "y_true")
    unknown = set(np.unique(truth)).difference(labels)
    if unknown:
        raise ValueError(f"y_true contains labels outside the protocol: {sorted(unknown, key=str)}")
    raw_scores = np.asarray(probabilities)
    if raw_scores.ndim == 1:
        scores = np.asarray(raw_scores, dtype=float)
        if len(scores) != len(truth):
            raise ValueError("y_true and probabilities have different lengths")
        if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError("binary probabilities must be finite and lie in [0, 1]")
    else:
        matrix = _validate_probability_matrix(raw_scores, len(truth))
        matrix = _probability_columns(matrix, labels, probability_labels)
        scores = matrix[:, labels.index(positive_label)]

    roc_auc_score, f1_score, recall_score = _sklearn_metrics()
    binary_truth = truth == positive_label
    predictions = np.where(scores >= threshold, positive_label, negative_label)
    support = {label: int(np.sum(truth == label)) for label in labels}
    result: dict[str, Any] = {
        "n_samples": int(len(truth)),
        "labels": list(labels),
        "positive_label": positive_label,
        "threshold": threshold,
        "support": support,
    }
    if not binary_truth.any() or binary_truth.all():
        auc = _missing_value(
            policy,
            "binary AUROC is undefined because y_true does not contain both configured classes",
        )
        if auc is not None:
            result["auroc"] = auc
            result["auc"] = auc
    else:
        result["auroc"] = float(roc_auc_score(binary_truth.astype(int), scores))
        result["auc"] = result["auroc"]

    if any(count == 0 for count in support.values()):
        missing_result = _missing_value(
            policy,
            "binary F1/recall are undefined under the configured missing-class policy "
            "because y_true does not contain both classes",
        )
        if missing_result is not None:
            result["f1"] = missing_result
            result["recall"] = missing_result
    else:
        result["f1"] = float(
            f1_score(
                truth,
                predictions,
                labels=list(labels),
                pos_label=positive_label,
                average="binary",
                zero_division=zero_division,
            )
        )
        result["recall"] = float(
            recall_score(
                truth,
                predictions,
                labels=list(labels),
                pos_label=positive_label,
                average="binary",
                zero_division=zero_division,
            )
        )
    return result


def evaluate_multiclass(
    y_true: Any,
    probabilities: Any,
    config: Any,
    *,
    probability_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Report macro OVR-AUC/F1 and per-class OVR-AUC/recall.

    ``missing_class_policy`` applies to every configured class that has no true
    examples.  ``raise`` aborts reporting; ``nan`` retains the class with NaN
    class metrics and consequently a NaN macro metric; ``skip`` excludes it
    from per-class results and macro denominators.  A class containing all
    observations has an undefined OVR-AUC even though its recall/F1 are defined;
    the same policy is applied to that AUC.
    """

    cfg = _subsection(config, "multiclass")
    labels = _validate_labels(_required(cfg, "labels", "metrics.multiclass"))
    decision_rule = str(_required(cfg, "decision_rule", "metrics.multiclass"))
    if decision_rule != "argmax":
        raise ValueError("metrics.multiclass.decision_rule must be 'argmax'")
    policy = _validate_policy(_required(cfg, "missing_class_policy", "metrics.multiclass"))
    zero_division = _required(cfg, "zero_division", "metrics.multiclass")
    if zero_division not in (0, 1, "warn"):
        raise ValueError("metrics.multiclass.zero_division must be 0, 1, or 'warn'")

    truth = _as_1d(y_true, "y_true")
    unknown = set(np.unique(truth)).difference(labels)
    if unknown:
        raise ValueError(f"y_true contains labels outside the protocol: {sorted(unknown, key=str)}")
    matrix = _validate_probability_matrix(probabilities, len(truth))
    matrix = _probability_columns(matrix, labels, probability_labels)
    predictions = np.asarray(labels, dtype=object)[np.argmax(matrix, axis=1)]
    roc_auc_score, f1_score, recall_score = _sklearn_metrics()

    support = {label: int(np.sum(truth == label)) for label in labels}
    per_auc: dict[Any, float] = {}
    per_f1: dict[Any, float] = {}
    per_recall: dict[Any, float] = {}
    auc_values: list[float] = []
    f1_values: list[float] = []
    recall_values: list[float] = []

    for column, label in enumerate(labels):
        target = truth == label
        missing_positive = not target.any()
        if missing_positive:
            missing = _missing_value(
                policy,
                f"class {label!r} is absent from y_true",
            )
            if missing is None:
                continue
            per_auc[label] = missing
            per_f1[label] = missing
            per_recall[label] = missing
            auc_values.append(missing)
            f1_values.append(missing)
            recall_values.append(missing)
            continue

        class_f1 = float(
            f1_score(
                target.astype(int),
                (predictions == label).astype(int),
                average="binary",
                zero_division=zero_division,
            )
        )
        class_recall = float(
            recall_score(
                target.astype(int),
                (predictions == label).astype(int),
                average="binary",
                zero_division=zero_division,
            )
        )
        per_f1[label] = class_f1
        per_recall[label] = class_recall
        f1_values.append(class_f1)
        recall_values.append(class_recall)

        if target.all():
            missing_auc = _missing_value(
                policy,
                f"one-vs-rest AUROC for class {label!r} is undefined because no rest class exists",
            )
            if missing_auc is not None:
                per_auc[label] = missing_auc
                auc_values.append(missing_auc)
        else:
            class_auc = float(roc_auc_score(target.astype(int), matrix[:, column]))
            per_auc[label] = class_auc
            auc_values.append(class_auc)

    def macro(values: list[float]) -> float:
        if not values:
            return float("nan")
        return float(np.mean(np.asarray(values, dtype=float)))

    macro_auc = macro(auc_values)
    result = {
        "n_samples": int(len(truth)),
        "labels": list(labels),
        "decision_rule": decision_rule,
        "support": support,
        "macro_ovr_auc": macro_auc,
        "macro_auc": macro_auc,
        "macro_f1": macro(f1_values),
        "macro_recall": macro(recall_values),
        "per_class_ovr_auc": per_auc,
        "per_class_auc": per_auc,
        "per_class_f1": per_f1,
        "per_class_recall": per_recall,
    }
    return result


def evaluate_classification(
    y_true: Any,
    probabilities: Any,
    config: Any,
    *,
    task: str,
    probability_labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Dispatch binary or multiclass reporting using an explicit task name."""

    if task == "binary":
        return evaluate_binary(
            y_true,
            probabilities,
            config,
            probability_labels=probability_labels,
        )
    if task == "multiclass":
        return evaluate_multiclass(
            y_true,
            probabilities,
            config,
            probability_labels=probability_labels,
        )
    raise ValueError("task must be 'binary' or 'multiclass'")


def aggregate_seed_metrics(
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    group_by: Sequence[str],
    config: Any,
) -> list[dict[str, Any]]:
    """Aggregate scalar metrics across seeds without flattening nested reports."""

    cfg = _subsection(config, "aggregation")
    ddof = int(_required(cfg, "ddof", "metrics.aggregation"))
    if ddof < 0:
        raise ValueError("metrics.aggregation.ddof must be non-negative")
    if not records:
        return []
    if not metric_names:
        raise ValueError("metric_names must not be empty")

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        missing = set(group_by).difference(record)
        missing.update(set(metric_names).difference(record))
        if missing:
            raise ValueError(f"metric record is missing fields: {sorted(missing)}")
        key = tuple(record[name] for name in group_by)
        groups.setdefault(key, []).append(record)

    output: list[dict[str, Any]] = []
    for key, rows in groups.items():
        aggregate = dict(zip(group_by, key))
        aggregate["n_seeds"] = len(rows)
        for name in metric_names:
            values = np.asarray([row[name] for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            aggregate[f"{name}_n"] = int(len(finite))
            aggregate[f"{name}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            aggregate[f"{name}_std"] = (
                float(np.std(finite, ddof=ddof)) if len(finite) > ddof else float("nan")
            )
        output.append(aggregate)
    return output


def paired_record_differences(
    records: Sequence[Mapping[str, Any]],
    metric_names: Sequence[str],
    pair_by: Sequence[str],
    *,
    condition_key: str,
    left_condition: Any,
    right_condition: Any,
) -> list[dict[str, Any]]:
    """Pair seed-level records and calculate left-minus-right contrasts."""

    pairs: dict[tuple[Any, ...], dict[Any, Mapping[str, Any]]] = {}
    needed = set(pair_by) | set(metric_names) | {condition_key}
    for record in records:
        missing = needed.difference(record)
        if missing:
            raise ValueError(f"metric record is missing fields: {sorted(missing)}")
        condition = record[condition_key]
        if condition not in (left_condition, right_condition):
            continue
        key = tuple(record[name] for name in pair_by)
        bucket = pairs.setdefault(key, {})
        if condition in bucket:
            raise ValueError(f"duplicate record for pair {key} and condition {condition!r}")
        bucket[condition] = record

    output: list[dict[str, Any]] = []
    for key, pair in pairs.items():
        if set(pair) != {left_condition, right_condition}:
            raise ValueError(f"incomplete condition pair for {key}: {sorted(pair, key=str)}")
        row = dict(zip(pair_by, key))
        row["left_condition"] = left_condition
        row["right_condition"] = right_condition
        for name in metric_names:
            left = float(pair[left_condition][name])
            right = float(pair[right_condition][name])
            row[f"{name}_left"] = left
            row[f"{name}_right"] = right
            row[f"{name}_difference"] = left - right
        output.append(row)
    return output


@dataclass(frozen=True)
class BootstrapDifference:
    """Paired bootstrap estimate for a left-minus-right metric contrast."""

    estimate: float
    confidence_lower: float
    confidence_upper: float
    confidence_level: float
    interval_method: str
    n_resamples_requested: int
    n_resamples_valid: int
    clustered: bool


def paired_bootstrap_difference(
    y_true: Any,
    left_predictions: Any,
    right_predictions: Any,
    metric: Callable[[np.ndarray, np.ndarray], float],
    config: Any,
    *,
    cluster_ids: Any | None = None,
) -> BootstrapDifference:
    """Bootstrap a paired model contrast, optionally at patient/cluster level.

    ``metric`` receives ``(resampled_y_true, resampled_predictions)``.  The
    prediction objects may be one- or two-dimensional, but their first
    dimension must match ``y_true``.  With ``cluster_ids``, unique clusters are
    sampled with replacement and all rows from each selected cluster are kept;
    repeated clusters therefore repeat their complete row blocks.
    """

    cfg = _subsection(config, "bootstrap")
    n_resamples = int(_required(cfg, "n_resamples", "metrics.bootstrap"))
    confidence_level = float(_required(cfg, "confidence_level", "metrics.bootstrap"))
    seed = int(_required(cfg, "seed", "metrics.bootstrap"))
    interval_method = str(_required(cfg, "interval_method", "metrics.bootstrap"))
    if interval_method != "percentile":
        raise ValueError("metrics.bootstrap.interval_method must be 'percentile'")
    policy = _validate_policy(
        _required(cfg, "missing_class_policy", "metrics.bootstrap"), bootstrap=True
    )
    if n_resamples <= 0:
        raise ValueError("metrics.bootstrap.n_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("metrics.bootstrap.confidence_level must lie in (0, 1)")

    truth = _as_1d(y_true, "y_true")
    left = np.asarray(left_predictions)
    right = np.asarray(right_predictions)
    if left.ndim not in (1, 2) or right.ndim not in (1, 2):
        raise ValueError("prediction arrays must be one- or two-dimensional")
    if left.shape != right.shape or left.shape[0] != len(truth):
        raise ValueError("paired predictions must have identical shape and match y_true")

    clusters: np.ndarray | None = None
    cluster_rows: list[np.ndarray] | None = None
    if cluster_ids is not None:
        clusters = _as_1d(cluster_ids, "cluster_ids")
        if len(clusters) != len(truth):
            raise ValueError("cluster_ids must have the same length as y_true")
        unique_clusters = np.unique(clusters)
        cluster_rows = [np.flatnonzero(clusters == value) for value in unique_clusters]
        if not cluster_rows:
            raise ValueError("cluster_ids must contain at least one cluster")

    try:
        estimate = float(metric(truth, left) - metric(truth, right))
    except (ValueError, FloatingPointError) as exc:
        raise ValueError("the paired metric is not estimable on the original sample") from exc
    if not np.isfinite(estimate):
        raise ValueError("the paired metric is not finite on the original sample")

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(n_resamples):
        if cluster_rows is None:
            indices = rng.integers(0, len(truth), size=len(truth))
        else:
            selected = rng.integers(0, len(cluster_rows), size=len(cluster_rows))
            indices = np.concatenate([cluster_rows[index] for index in selected])
        try:
            difference = float(
                metric(truth[indices], left[indices]) - metric(truth[indices], right[indices])
            )
        except (ValueError, FloatingPointError):
            if policy == "raise":
                raise
            continue
        if not np.isfinite(difference):
            if policy == "raise":
                raise ValueError("a bootstrap replicate produced a non-finite metric")
            continue
        differences.append(difference)

    if not differences:
        raise ValueError("no valid paired bootstrap replicates were produced")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(differences), [alpha, 1.0 - alpha])
    return BootstrapDifference(
        estimate=estimate,
        confidence_lower=float(lower),
        confidence_upper=float(upper),
        confidence_level=confidence_level,
        interval_method=interval_method,
        n_resamples_requested=n_resamples,
        n_resamples_valid=len(differences),
        clustered=cluster_ids is not None,
    )


__all__ = [
    "BootstrapDifference",
    "aggregate_seed_metrics",
    "evaluate_binary",
    "evaluate_classification",
    "evaluate_multiclass",
    "paired_bootstrap_difference",
    "paired_record_differences",
]
