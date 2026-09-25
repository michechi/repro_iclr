"""Run the model and perturbation matrix for the AKI trajectory audit.

The split map and matched cohort are inputs to this module: model seeds never
change patient assignment or matching.  Each configured model seed is paired
with one configured value-permutation seed, making ordered-versus-perturbed
contrasts reproducible at the episode level.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .episodes import construct_aki_episodes
from .metrics import (
    aggregate_seed_metrics,
    evaluate_binary,
    evaluate_multiclass,
    paired_bootstrap_difference,
)
from .models import (
    sequence_records_to_inputs,
    train_sequence_classifier,
    train_tabular_classifier,
)
from .representations import RepresentationBundle, build_representations


TASKS = ("binary", "multiclass")
SEQUENCE_MODELS = ("lstm", "transformer")
TABULAR_RUNS = (
    ("logistic_regression", "invariant_summary"),
    ("xgboost", "invariant_summary"),
    ("logistic_regression", "count_only"),
    ("logistic_regression", "first_value"),
    ("logistic_regression", "last_value"),
    ("logistic_regression", "first_to_last_change"),
    ("logistic_regression", "linear_slope"),
    ("logistic_regression", "position_combined"),
)


class OracleNotEstimable(RuntimeError):
    """The configured model window lacks evidence needed to rerun the oracle."""


@dataclass(frozen=True)
class ExperimentResult:
    """Machine-readable outputs from all requested model conditions."""

    seed_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    predictions: pd.DataFrame
    contrasts: pd.DataFrame
    aggregate_contrasts: pd.DataFrame
    task_estimability: pd.DataFrame
    condition_estimability: pd.DataFrame


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


def _configured_seed_pairs(config: Any) -> tuple[tuple[int, int], ...]:
    model_values = _required(_section(config, "models"), "seeds", "models")
    permutation_values = _required(
        _section(config, "representations"), "permutation_seeds", "representations"
    )
    if (
        not isinstance(model_values, Sequence)
        or isinstance(model_values, (str, bytes))
        or not isinstance(permutation_values, Sequence)
        or isinstance(permutation_values, (str, bytes))
    ):
        raise ValueError("models.seeds and representations.permutation_seeds must be sequences")
    if len(model_values) != 3 or len(permutation_values) != 3:
        raise ValueError(
            "the locked audit requires exactly three model seeds and three paired "
            "permutation seeds"
        )
    model_seeds = tuple(_integer_seed(value, "models.seeds") for value in model_values)
    permutation_seeds = tuple(
        _integer_seed(value, "representations.permutation_seeds") for value in permutation_values
    )
    if len(set(model_seeds)) != 3 or len(set(permutation_seeds)) != 3:
        raise ValueError("model and permutation seed lists must each contain unique values")
    return tuple(zip(model_seeds, permutation_seeds))


def _integer_seed(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{path} values must be integers")
    return int(value)


def _class_labels(config: Any, task: str) -> tuple[Any, ...]:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    metrics = _section(config, "metrics")
    subsection = metrics.get(task)
    if not isinstance(subsection, Mapping):
        raise ValueError(f"metrics.{task} must be a mapping")
    labels = _required(subsection, "labels", f"metrics.{task}")
    expected = 2 if task == "binary" else 3
    if (
        not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or len(labels) != expected
        or len(set(labels)) != expected
    ):
        raise ValueError(f"metrics.{task}.labels must have exactly {expected} unique values")
    return tuple(labels)


def _validate_model_dataset(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    class_labels: tuple[Any, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered_events, ordered_labels = _validate_dataset_integrity(events, labels, class_labels)
    allowed_splits = {"train", "validation", "test"}
    for split in sorted(allowed_splits):
        present = set(ordered_labels.loc[ordered_labels["split"] == split, "label"])
        missing_classes = set(class_labels).difference(present)
        if missing_classes:
            raise ValueError(
                f"{split} lacks configured classes {sorted(missing_classes, key=str)}; "
                "the requested task is not estimable on this split"
            )
    return ordered_events, ordered_labels


def _validate_dataset_integrity(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    class_labels: tuple[Any, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate identity and leakage invariants without requiring every class."""

    required_labels = {"episode_id", "subject_id", "hadm_id", "label", "split"}
    missing = required_labels.difference(labels.columns)
    if missing:
        raise ValueError(f"model labels are missing columns: {sorted(missing)}")
    if labels["episode_id"].duplicated().any():
        raise ValueError("model labels must contain one row per episode")
    if set(labels["label"]) - set(class_labels):
        raise ValueError("model labels contain phenotypes outside the configured class order")
    allowed_splits = {"train", "validation", "test"}
    if set(labels["split"]) != allowed_splits:
        raise ValueError(
            "model labels must contain non-empty train, validation, and test partitions"
        )
    subject_splits = labels[["subject_id", "split"]].drop_duplicates()
    if subject_splits["subject_id"].duplicated().any():
        raise AssertionError("a subject appears in more than one model split")
    required_events = {"episode_id", "subject_id", "hadm_id"}
    missing_events = required_events.difference(events.columns)
    if missing_events:
        raise ValueError(f"model event table is missing columns: {sorted(missing_events)}")
    event_ids = set(events["episode_id"])
    label_ids = set(labels["episode_id"])
    if event_ids != label_ids:
        raise ValueError(
            "event/label episode identifiers differ; "
            f"events_only={len(event_ids - label_ids)}, labels_only={len(label_ids - event_ids)}"
        )
    event_identity = events[["episode_id", "subject_id", "hadm_id"]].drop_duplicates()
    if event_identity["episode_id"].duplicated().any():
        raise ValueError("event rows disagree on subject_id/hadm_id within an episode")
    label_identity = labels[["episode_id", "subject_id", "hadm_id"]]
    identity = label_identity.merge(
        event_identity,
        on="episode_id",
        how="left",
        suffixes=("_label", "_event"),
        validate="one_to_one",
    )
    if (
        identity["subject_id_label"].ne(identity["subject_id_event"])
        | identity["hadm_id_label"].ne(identity["hadm_id_event"])
    ).any():
        raise ValueError("event and label subject_id/hadm_id identities disagree")
    ordered_labels = labels.sort_values("episode_id", kind="mergesort").reset_index(drop=True)
    ordered_events = events[events["episode_id"].isin(label_ids)].copy()
    return ordered_events, ordered_labels


def _task_non_estimable_reasons(
    labels: pd.DataFrame, class_labels: tuple[Any, ...]
) -> tuple[str, ...]:
    required = {"episode_id", "subject_id", "hadm_id", "label", "split"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"model labels are missing columns: {sorted(missing)}")
    reasons: list[str] = []
    for split in ("train", "validation", "test"):
        present = set(labels.loc[labels["split"] == split, "label"])
        missing_classes = set(class_labels).difference(present)
        if missing_classes:
            reasons.append(f"{split} lacks classes {sorted(missing_classes, key=str)}")
    return tuple(reasons)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"episode_id", "subject_id", "hadm_id", "label", "split"}
    columns = [column for column in frame.columns if column not in excluded]
    if not columns:
        raise ValueError("tabular representation contains no feature columns")
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(f"tabular features must be numeric: {non_numeric}")
    return columns


def _with_splits(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "split" in frame:
        frame = frame.drop(columns="split")
    split_map = labels[["episode_id", "split"]].drop_duplicates("episode_id")
    result = frame.merge(split_map, on="episode_id", how="left", validate="one_to_one")
    if result["split"].isna().any():
        raise AssertionError("a representation row has no patient split")
    return result


def _split_table(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: frame[frame["split"] == split]
        .sort_values("episode_id", kind="mergesort")
        .reset_index(drop=True)
        for split in ("train", "validation", "test")
    }


def _split_records(
    records: Sequence[Mapping[str, Any]], labels: pd.DataFrame
) -> dict[str, list[Mapping[str, Any]]]:
    split_by_id = labels.set_index("episode_id")["split"].to_dict()
    output: dict[str, list[Mapping[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[Any] = set()
    for record in records:
        episode_id = record["episode_id"]
        if episode_id in seen:
            raise ValueError(f"duplicate sequence record for episode {episode_id}")
        seen.add(episode_id)
        try:
            split = split_by_id[episode_id]
        except KeyError as exc:
            raise ValueError(f"sequence record {episode_id} has no label/split") from exc
        output[str(split)].append(record)
    if seen != set(split_by_id):
        raise ValueError("not every labeled episode produced a sequence record")
    for split in output:
        output[split].sort(key=lambda record: str(record["episode_id"]))
    return output


def _metric_column_label(label: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(label)).strip("_").lower()
    return text or "empty_label"


def _evaluate(
    task: str,
    truth: np.ndarray,
    probabilities: np.ndarray,
    class_labels: tuple[Any, ...],
    config: Any,
) -> dict[str, Any]:
    if task == "binary":
        report = evaluate_binary(truth, probabilities, config, probability_labels=class_labels)
        return {
            "auroc": report.get("auroc", float("nan")),
            "f1": report.get("f1", float("nan")),
            "recall": report.get("recall", float("nan")),
            "n_samples": report["n_samples"],
        }
    report = evaluate_multiclass(truth, probabilities, config, probability_labels=class_labels)
    flat: dict[str, Any] = {
        "macro_ovr_auc": report["macro_ovr_auc"],
        "macro_f1": report["macro_f1"],
        "macro_recall": report["macro_recall"],
        "n_samples": report["n_samples"],
    }
    for label in class_labels:
        suffix = _metric_column_label(label)
        flat[f"ovr_auc_{suffix}"] = report["per_class_ovr_auc"].get(label, float("nan"))
        flat[f"f1_{suffix}"] = report["per_class_f1"].get(label, float("nan"))
        flat[f"recall_{suffix}"] = report["per_class_recall"].get(label, float("nan"))
    return flat


def _prediction_rows(
    *,
    task: str,
    model: str,
    condition: str,
    feature_set: str,
    model_seed: int,
    permutation_seed: int,
    test_labels: pd.DataFrame,
    probabilities: np.ndarray,
    class_labels: tuple[Any, ...],
) -> list[dict[str, Any]]:
    if probabilities.shape != (len(test_labels), len(class_labels)):
        raise ValueError("test probabilities do not align with labels/classes")
    rows: list[dict[str, Any]] = []
    for index, episode in test_labels.reset_index(drop=True).iterrows():
        row: dict[str, Any] = {
            "task": task,
            "model": model,
            "condition": condition,
            "feature_set": feature_set,
            "model_seed": model_seed,
            "permutation_seed": permutation_seed,
            "episode_id": episode["episode_id"],
            "subject_id": episode["subject_id"],
            "hadm_id": episode["hadm_id"],
            "true_label": episode["label"],
        }
        for column, label in enumerate(class_labels):
            row[f"probability_{_metric_column_label(label)}"] = float(probabilities[index, column])
        rows.append(row)
    return rows


def _record_run(
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    task: str,
    model: str,
    condition: str,
    feature_set: str,
    model_seed: int,
    permutation_seed: int,
    test_labels: pd.DataFrame,
    probabilities: np.ndarray,
    class_labels: tuple[Any, ...],
    config: Any,
) -> None:
    truth = test_labels["label"].to_numpy()
    metrics = _evaluate(task, truth, probabilities, class_labels, config)
    metric_rows.append(
        {
            "task": task,
            "model": model,
            "condition": condition,
            "feature_set": feature_set,
            "model_seed": model_seed,
            "permutation_seed": permutation_seed,
            **metrics,
        }
    )
    prediction_rows.extend(
        _prediction_rows(
            task=task,
            model=model,
            condition=condition,
            feature_set=feature_set,
            model_seed=model_seed,
            permutation_seed=permutation_seed,
            test_labels=test_labels,
            probabilities=probabilities,
            class_labels=class_labels,
        )
    )


def _run_tabular_models(
    bundle: RepresentationBundle,
    labels: pd.DataFrame,
    task: str,
    class_labels: tuple[Any, ...],
    model_seed: int,
    permutation_seed: int,
    config: Any,
    enabled_models: set[str],
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    for model_name, feature_set in TABULAR_RUNS:
        if model_name not in enabled_models:
            continue
        table = _with_splits(bundle.tabular[feature_set], labels)
        parts = _split_table(table)
        columns = _feature_columns(table)
        fitted = train_tabular_classifier(
            model_name,
            parts["train"][columns].to_numpy(),
            parts["train"]["label"].to_numpy(),
            config,
            seed=model_seed,
            class_labels=class_labels,
            validation_data=(
                parts["validation"][columns].to_numpy(),
                parts["validation"]["label"].to_numpy(),
            ),
        )
        probabilities = fitted.predict_proba(parts["test"][columns].to_numpy())
        _record_run(
            metric_rows,
            prediction_rows,
            task=task,
            model=model_name,
            condition="trained_and_tested_same_representation",
            feature_set=feature_set,
            model_seed=model_seed,
            permutation_seed=permutation_seed,
            test_labels=parts["test"],
            probabilities=probabilities,
            class_labels=class_labels,
            config=config,
        )


def _records_as_inputs(
    parts: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, tuple[list[np.ndarray], np.ndarray, list[Any]]]:
    return {split: sequence_records_to_inputs(records) for split, records in parts.items()}


def _labels_for_episode_order(labels: pd.DataFrame, episode_ids: Sequence[Any]) -> pd.DataFrame:
    indexed = labels.set_index("episode_id", drop=False)
    try:
        return indexed.loc[list(episode_ids)].reset_index(drop=True)
    except KeyError as exc:
        raise AssertionError("prediction episode order could not be resolved") from exc


def _run_sequence_models(
    bundle: RepresentationBundle,
    labels: pd.DataFrame,
    task: str,
    class_labels: tuple[Any, ...],
    model_seed: int,
    permutation_seed: int,
    config: Any,
    enabled_models: set[str],
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    record_sets = {
        "ordered": _records_as_inputs(_split_records(bundle.ordered, labels)),
        "shuffled": _records_as_inputs(_split_records(bundle.shuffled, labels)),
        "reversed": _records_as_inputs(_split_records(bundle.reversed, labels)),
    }
    ordered = record_sets["ordered"]
    shuffled = record_sets["shuffled"]
    reversed_values = record_sets["reversed"]
    test_labels = _labels_for_episode_order(labels, ordered["test"][2])
    for model_name in SEQUENCE_MODELS:
        if model_name not in enabled_models:
            continue
        fitted_ordered = train_sequence_classifier(
            model_name,
            ordered["train"][0],
            ordered["train"][1],
            ordered["validation"][0],
            ordered["validation"][1],
            config,
            seed=model_seed,
            class_labels=class_labels,
        )
        for condition, feature_set, inputs in (
            ("ordered_train_ordered_test", "ordered", ordered["test"]),
            ("ordered_train_shuffled_test", "shuffled", shuffled["test"]),
            ("ordered_train_reversed_test", "reversed", reversed_values["test"]),
        ):
            if inputs[2] != ordered["test"][2]:
                raise AssertionError("sequence perturbations changed test episode order")
            _record_run(
                metric_rows,
                prediction_rows,
                task=task,
                model=model_name,
                condition=condition,
                feature_set=feature_set,
                model_seed=model_seed,
                permutation_seed=permutation_seed,
                test_labels=test_labels,
                probabilities=fitted_ordered.predict_proba(inputs[0]),
                class_labels=class_labels,
                config=config,
            )

        fitted_shuffled = train_sequence_classifier(
            model_name,
            shuffled["train"][0],
            shuffled["train"][1],
            shuffled["validation"][0],
            shuffled["validation"][1],
            config,
            seed=model_seed,
            class_labels=class_labels,
        )
        _record_run(
            metric_rows,
            prediction_rows,
            task=task,
            model=model_name,
            condition="shuffled_train_shuffled_test",
            feature_set="shuffled",
            model_seed=model_seed,
            permutation_seed=permutation_seed,
            test_labels=test_labels,
            probabilities=fitted_shuffled.predict_proba(shuffled["test"][0]),
            class_labels=class_labels,
            config=config,
        )


def state_machine_oracle_probabilities(
    events: pd.DataFrame,
    test_labels: pd.DataFrame,
    config: Any,
    class_labels: tuple[Any, ...],
) -> np.ndarray:
    """Re-run episode construction on each test representation as an oracle."""

    positions = {label: index for index, label in enumerate(class_labels)}
    probabilities = np.zeros((len(test_labels), len(class_labels)), dtype=float)
    for output_row, label_row in test_labels.reset_index(drop=True).iterrows():
        episode_events = events[events["episode_id"] == label_row["episode_id"]].copy()
        if episode_events.empty:
            raise AssertionError(f"oracle has no events for episode {label_row['episode_id']}")
        reconstructed = construct_aki_episodes(episode_events, config).episodes
        expected_onset = pd.Timestamp(label_row["onset_time"])
        matching = reconstructed[
            pd.to_datetime(reconstructed["onset_time"], errors="coerce") == expected_onset
        ]
        if matching.empty:
            raise OracleNotEstimable(
                "the configured trajectory window could not reconstruct test episode "
                f"{label_row['episode_id']} at onset {expected_onset}"
            )
        if len(matching) > 1:
            raise AssertionError(
                "the state-machine oracle reconstructed duplicate decisions for "
                f"episode {label_row['episode_id']} at onset {expected_onset}"
            )
        phenotype = matching.iloc[0]["phenotype"]
        if phenotype not in positions:
            if phenotype == "censored":
                raise OracleNotEstimable(
                    "the configured trajectory window censored the state-machine oracle "
                    f"for episode {label_row['episode_id']}"
                )
            raise AssertionError(
                f"state-machine oracle returned non-target phenotype {phenotype!r} "
                f"for episode {label_row['episode_id']}"
            )
        if phenotype != label_row["label"]:
            raise AssertionError(
                "state-machine oracle disagrees with the persisted audit label for "
                f"episode {label_row['episode_id']}: {phenotype!r} != {label_row['label']!r}"
            )
        probabilities[output_row, positions[phenotype]] = 1.0
    return probabilities


def run_task_experiment(
    events: pd.DataFrame,
    labels: pd.DataFrame,
    config: Any,
    *,
    task: str,
    enabled_models: Sequence[str] = (
        "logistic_regression",
        "xgboost",
        "lstm",
        "transformer",
    ),
    include_oracle: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one binary or multiclass task across the three paired seeds."""

    class_labels = _class_labels(config, task)
    events, labels = _validate_model_dataset(events, labels, class_labels)
    enabled = set(enabled_models)
    allowed = {"logistic_regression", "xgboost", "lstm", "transformer"}
    unknown = enabled.difference(allowed)
    if unknown:
        raise ValueError(f"unsupported enabled models: {sorted(unknown)}")

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    condition_estimability: list[dict[str, Any]] = []
    for model_seed, permutation_seed in _configured_seed_pairs(config):
        bundle = build_representations(events, labels, config, shuffle_seed=permutation_seed)
        _run_tabular_models(
            bundle,
            labels,
            task,
            class_labels,
            model_seed,
            permutation_seed,
            config,
            enabled,
            metric_rows,
            prediction_rows,
        )
        _run_sequence_models(
            bundle,
            labels,
            task,
            class_labels,
            model_seed,
            permutation_seed,
            config,
            enabled,
            metric_rows,
            prediction_rows,
        )

    test_labels = labels[labels["split"] == "test"].sort_values("episode_id", kind="mergesort")
    if include_oracle:
        try:
            oracle = state_machine_oracle_probabilities(events, test_labels, config, class_labels)
        except OracleNotEstimable as exc:
            condition_estimability.append(
                {
                    "task": task,
                    "model": "state_machine_oracle",
                    "condition": "oracle_ceiling",
                    "estimable": False,
                    "reason": str(exc),
                }
            )
        else:
            condition_estimability.append(
                {
                    "task": task,
                    "model": "state_machine_oracle",
                    "condition": "oracle_ceiling",
                    "estimable": True,
                    "reason": "",
                }
            )
            _record_run(
                metric_rows,
                prediction_rows,
                task=task,
                model="state_machine_oracle",
                condition="oracle_ceiling",
                feature_set="configured_trajectory_state_machine",
                model_seed=-1,
                permutation_seed=-1,
                test_labels=test_labels,
                probabilities=oracle,
                class_labels=class_labels,
                config=config,
            )
    return metric_rows, prediction_rows, condition_estimability


def _aggregate(seed_metrics: pd.DataFrame, config: Any) -> pd.DataFrame:
    if seed_metrics.empty:
        return pd.DataFrame()
    identifier_columns = [
        "task",
        "model",
        "condition",
        "feature_set",
        "model_seed",
        "permutation_seed",
        "n_samples",
    ]
    metric_columns = [
        column
        for column in seed_metrics.columns
        if column not in identifier_columns and pd.api.types.is_numeric_dtype(seed_metrics[column])
    ]
    records: list[dict[str, Any]] = []
    group_by = ["task", "model", "condition", "feature_set"]
    for task, task_frame in seed_metrics.groupby("task", sort=False):
        task_metrics = [column for column in metric_columns if task_frame[column].notna().any()]
        if not task_metrics:
            continue
        records.extend(
            aggregate_seed_metrics(
                task_frame.to_dict(orient="records"), task_metrics, group_by, config
            )
        )
    return pd.DataFrame(records)


def _contrasts(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return paired seed-level deltas central to the order audit."""

    if seed_metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for task, task_frame in seed_metrics.groupby("task", sort=False):
        metric = "auroc" if task == "binary" else "macro_ovr_auc"
        for model in SEQUENCE_MODELS:
            model_frame = task_frame[
                (task_frame["model"] == model) & (task_frame["model_seed"] >= 0)
            ]
            for seed, seed_frame in model_frame.groupby("model_seed"):
                indexed = seed_frame.set_index("condition")
                left_name = "ordered_train_ordered_test"
                if left_name not in indexed.index:
                    continue
                left = float(indexed.loc[left_name, metric])
                for right_name in (
                    "ordered_train_shuffled_test",
                    "ordered_train_reversed_test",
                    "shuffled_train_shuffled_test",
                ):
                    if right_name not in indexed.index:
                        continue
                    right = float(indexed.loc[right_name, metric])
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "model_seed": int(seed),
                            "metric": metric,
                            "left_condition": left_name,
                            "right_condition": right_name,
                            "left_value": left,
                            "right_value": right,
                            "ordered_minus_control": left - right,
                        }
                    )
                for control_model, control_feature in (
                    ("logistic_regression", "invariant_summary"),
                    ("xgboost", "invariant_summary"),
                    ("logistic_regression", "count_only"),
                ):
                    control = task_frame[
                        (task_frame["model"] == control_model)
                        & (task_frame["feature_set"] == control_feature)
                        & (task_frame["model_seed"] == seed)
                    ]
                    if len(control) != 1:
                        continue
                    right = float(control.iloc[0][metric])
                    rows.append(
                        {
                            "task": task,
                            "model": model,
                            "model_seed": int(seed),
                            "metric": metric,
                            "left_condition": left_name,
                            "right_condition": f"{control_model}:{control_feature}",
                            "left_value": left,
                            "right_value": right,
                            "ordered_minus_control": left - right,
                        }
                    )
    return pd.DataFrame(rows)


def _prediction_block(
    predictions: pd.DataFrame,
    *,
    task: str,
    model: str,
    condition: str,
    model_seed: int,
    feature_set: str | None = None,
) -> pd.DataFrame:
    mask = (
        (predictions["task"] == task)
        & (predictions["model"] == model)
        & (predictions["condition"] == condition)
        & (predictions["model_seed"] == model_seed)
    )
    if feature_set is not None:
        mask &= predictions["feature_set"] == feature_set
    return predictions.loc[mask].sort_values("episode_id", kind="mergesort").reset_index(drop=True)


def _bootstrap_metric(task: str, class_labels: tuple[Any, ...], config: Any) -> Any:
    from sklearn.metrics import roc_auc_score

    if task == "binary":
        positive = _section(config, "metrics")["binary"]["positive_label"]
        positive_column = class_labels.index(positive)

        def binary_auc(truth: np.ndarray, probabilities: np.ndarray) -> float:
            return float(
                roc_auc_score((truth == positive).astype(int), probabilities[:, positive_column])
            )

        return binary_auc

    def multiclass_auc(truth: np.ndarray, probabilities: np.ndarray) -> float:
        class_aucs = []
        for column, label in enumerate(class_labels):
            target = truth == label
            if not target.any() or target.all():
                raise ValueError(f"one-vs-rest AUROC is undefined for class {label!r}")
            class_aucs.append(float(roc_auc_score(target.astype(int), probabilities[:, column])))
        return float(np.mean(class_aucs))

    return multiclass_auc


def _add_bootstrap_intervals(
    contrasts: pd.DataFrame,
    predictions: pd.DataFrame,
    config: Any,
) -> pd.DataFrame:
    """Attach patient-clustered paired percentile intervals to every contrast."""

    if contrasts.empty:
        return contrasts
    rows: list[dict[str, Any]] = []
    for contrast in contrasts.to_dict(orient="records"):
        task = contrast["task"]
        model = contrast["model"]
        seed = int(contrast["model_seed"])
        left = _prediction_block(
            predictions,
            task=task,
            model=model,
            condition=contrast["left_condition"],
            model_seed=seed,
        )
        right_name = str(contrast["right_condition"])
        if ":" in right_name:
            right_model, right_feature = right_name.split(":", 1)
            right_condition = "trained_and_tested_same_representation"
        else:
            right_model = model
            right_condition = right_name
            right_feature = None
        right = _prediction_block(
            predictions,
            task=task,
            model=right_model,
            condition=right_condition,
            model_seed=seed,
            feature_set=right_feature,
        )
        identity = ["episode_id", "subject_id", "true_label"]
        if left[identity].to_dict(orient="records") != right[identity].to_dict(orient="records"):
            raise AssertionError("paired contrast prediction rows are not episode-aligned")
        class_labels = _class_labels(config, task)
        probability_columns = [
            f"probability_{_metric_column_label(label)}" for label in class_labels
        ]
        bootstrap = paired_bootstrap_difference(
            left["true_label"].to_numpy(),
            left[probability_columns].to_numpy(),
            right[probability_columns].to_numpy(),
            _bootstrap_metric(task, class_labels, config),
            config,
            cluster_ids=left["subject_id"].to_numpy(),
        )
        rows.append(
            {
                **contrast,
                "bootstrap_estimate": bootstrap.estimate,
                "confidence_lower": bootstrap.confidence_lower,
                "confidence_upper": bootstrap.confidence_upper,
                "confidence_level": bootstrap.confidence_level,
                "bootstrap_interval_method": bootstrap.interval_method,
                "bootstrap_resamples_requested": bootstrap.n_resamples_requested,
                "bootstrap_resamples_valid": bootstrap.n_resamples_valid,
                "bootstrap_clustered_by_subject": bootstrap.clustered,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_contrasts(contrasts: pd.DataFrame, config: Any) -> pd.DataFrame:
    if contrasts.empty:
        return pd.DataFrame()
    group_by = [
        "task",
        "model",
        "metric",
        "left_condition",
        "right_condition",
    ]
    return pd.DataFrame(
        aggregate_seed_metrics(
            contrasts.to_dict(orient="records"),
            ["ordered_minus_control"],
            group_by,
            config,
        )
    )


def run_audit_experiments(
    datasets: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: Any,
    *,
    output_dir: str | Path | None = None,
    enabled_models: Sequence[str] = (
        "logistic_regression",
        "xgboost",
        "lstm",
        "transformer",
    ),
    include_oracle: bool = True,
) -> ExperimentResult:
    """Run the primary/secondary experiment and optionally persist all outputs.

    ``datasets`` maps ``binary`` and/or ``multiclass`` to ``(events, labels)``.
    The production CLI supplies both; accepting one task keeps synthetic and
    sensitivity analyses small without changing model behavior.
    """

    unknown_tasks = set(datasets).difference(TASKS)
    if unknown_tasks or not datasets:
        raise ValueError(f"datasets must contain binary/multiclass tasks, got {sorted(datasets)}")
    validated_datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    subject_assignments: list[pd.DataFrame] = []
    for task, (task_events, task_labels) in datasets.items():
        checked = _validate_dataset_integrity(
            task_events,
            task_labels,
            _class_labels(config, task),
        )
        validated_datasets[task] = checked
        subject_assignments.append(
            checked[1][["subject_id", "split"]].drop_duplicates().assign(task=task)
        )
    combined_assignments = pd.concat(subject_assignments, ignore_index=True)
    split_counts = combined_assignments.groupby("subject_id")["split"].nunique()
    cross_task_overlap = split_counts[split_counts > 1]
    if not cross_task_overlap.empty:
        raise AssertionError(
            "subjects have inconsistent splits across tasks: "
            f"{cross_task_overlap.index.tolist()[:10]}"
        )

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    estimability_rows: list[dict[str, Any]] = []
    condition_estimability_rows: list[dict[str, Any]] = []
    for task in TASKS:
        if task not in validated_datasets:
            continue
        events, labels = validated_datasets[task]
        class_labels = _class_labels(config, task)
        reasons = _task_non_estimable_reasons(labels, class_labels)
        estimability_rows.append(
            {
                "task": task,
                "estimable": not reasons,
                "non_estimable_reasons": list(reasons),
            }
        )
        if reasons:
            if include_oracle:
                condition_estimability_rows.append(
                    {
                        "task": task,
                        "model": "state_machine_oracle",
                        "condition": "oracle_ceiling",
                        "estimable": False,
                        "reason": "task is not estimable: " + "; ".join(reasons),
                    }
                )
            if task == "binary":
                raise ValueError("primary binary task is not estimable: " + "; ".join(reasons))
            continue
        task_metrics, task_predictions, task_conditions = run_task_experiment(
            events,
            labels,
            config,
            task=task,
            enabled_models=enabled_models,
            include_oracle=include_oracle,
        )
        metric_rows.extend(task_metrics)
        prediction_rows.extend(task_predictions)
        condition_estimability_rows.extend(task_conditions)

    seed_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate_metrics = _aggregate(seed_metrics, config)
    contrasts = _add_bootstrap_intervals(_contrasts(seed_metrics), predictions, config)
    aggregate_contrasts = _aggregate_contrasts(contrasts, config)
    result = ExperimentResult(
        seed_metrics=seed_metrics,
        aggregate_metrics=aggregate_metrics,
        predictions=predictions,
        contrasts=contrasts,
        aggregate_contrasts=aggregate_contrasts,
        task_estimability=pd.DataFrame(estimability_rows),
        condition_estimability=pd.DataFrame(condition_estimability_rows),
    )
    if output_dir is not None:
        _persist_experiment(result, Path(output_dir))
    return result


def _persist_experiment(result: ExperimentResult, output_dir: Path) -> None:
    tables = (
        ("seed_metrics", result.seed_metrics),
        ("aggregate_metrics", result.aggregate_metrics),
        ("predictions", result.predictions),
        ("ordered_control_contrasts", result.contrasts),
        ("aggregate_ordered_control_contrasts", result.aggregate_contrasts),
        ("task_estimability", result.task_estimability),
        ("condition_estimability", result.condition_estimability),
    )
    paths = [output_dir / f"{name}.parquet" for name, _ in tables]
    paths.extend(output_dir / f"{name}.csv" for name, _ in tables if name != "predictions")
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing experiment artifacts: "
            f"{[str(path) for path in existing[:10]]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables:
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        if name != "predictions":
            frame.to_csv(output_dir / f"{name}.csv", index=False)


__all__ = [
    "ExperimentResult",
    "SEQUENCE_MODELS",
    "TABULAR_RUNS",
    "run_audit_experiments",
    "run_task_experiment",
    "state_machine_oracle_probabilities",
]
