import pandas as pd
import pytest

from src.mimic.aki.experiment import (
    OracleNotEstimable,
    _add_bootstrap_intervals,
    _bootstrap_metric,
    _contrasts,
    run_audit_experiments,
)


def _config() -> dict:
    return {
        "representations": {
            "permutation_seeds": [101, 202, 303],
            "max_length": 16,
            "truncation": "error",
            "value_transform": "raw",
            "time_features": ["elapsed_hours", "delta_hours"],
            "summary_features": [
                "count",
                "distinct_count",
                "min",
                "max",
                "mean",
                "duration_hours",
            ],
            "slope_hours_per_unit": 24,
        },
        "models": {
            "seeds": [11, 22, 33],
            "tabular_preprocessing": {
                "scaling": "standard",
                "missing_values": "error",
            },
            "logistic_regression": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "liblinear",
                "max_iter": 200,
                "tol": 0.0001,
                "fit_intercept": True,
                "class_weight": None,
                "n_jobs": None,
            },
        },
        "metrics": {
            "binary": {
                "labels": ["transient", "persistent"],
                "positive_label": "persistent",
                "threshold": 0.5,
                "missing_class_policy": "raise",
                "zero_division": 0,
            },
            "aggregation": {"ddof": 1},
        },
    }


def _dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = []
    events = []
    episode_number = 0
    for split in ("train", "validation", "test"):
        for phenotype in ("transient", "persistent"):
            for replicate in range(3):
                episode_number += 1
                episode_id = f"e{episode_number:02d}"
                subject_id = episode_number
                labels.append(
                    {
                        "episode_id": episode_id,
                        "subject_id": subject_id,
                        "hadm_id": 1000 + subject_id,
                        "label": phenotype,
                        "split": split,
                        "baseline_creatinine_mg_dl": 1.0,
                        "onset_time": pd.Timestamp("2025-01-01 12:00"),
                    }
                )
                values = (
                    [1.0, 1.4 + replicate * 0.01, 1.05]
                    if phenotype == "transient"
                    else [1.0, 1.4 + replicate * 0.01, 1.45]
                )
                for offset, value in enumerate(values):
                    events.append(
                        {
                            "episode_id": episode_id,
                            "subject_id": subject_id,
                            "hadm_id": 1000 + subject_id,
                            "specimen_time": pd.Timestamp("2025-01-01")
                            + pd.Timedelta(hours=12 * offset),
                            "creatinine_mg_dl": value,
                        }
                    )
    return pd.DataFrame(events), pd.DataFrame(labels)


def test_logistic_end_to_end_runs_three_fixed_seed_pairs(tmp_path) -> None:
    events, labels = _dataset()

    result = run_audit_experiments(
        {"binary": (events, labels)},
        _config(),
        enabled_models=["logistic_regression"],
        include_oracle=False,
        output_dir=tmp_path / "experiment",
    )

    fitted = result.seed_metrics[result.seed_metrics["model_seed"] >= 0]
    assert set(fitted["model_seed"]) == {11, 22, 33}
    assert set(fitted["permutation_seed"]) == {101, 202, 303}
    assert {
        "invariant_summary",
        "count_only",
        "first_value",
        "last_value",
        "first_to_last_change",
        "linear_slope",
    }.issubset(set(fitted["feature_set"]))
    assert not result.aggregate_metrics.empty
    assert (tmp_path / "experiment" / "seed_metrics.parquet").is_file()
    assert (tmp_path / "experiment" / "aggregate_ordered_control_contrasts.csv").is_file()


def test_order_contrast_gets_patient_clustered_paired_interval() -> None:
    rows = []
    for condition in ("ordered_train_ordered_test", "ordered_train_shuffled_test"):
        for subject_id in range(12):
            truth = "persistent" if subject_id % 2 else "transient"
            ordered_score = 0.9 if truth == "persistent" else 0.1
            score = ordered_score if "ordered_test" in condition else 0.5
            rows.append(
                {
                    "task": "binary",
                    "model": "lstm",
                    "condition": condition,
                    "feature_set": "ordered" if "ordered_test" in condition else "shuffled",
                    "model_seed": 11,
                    "episode_id": f"e{subject_id}",
                    "subject_id": subject_id,
                    "true_label": truth,
                    "probability_transient": 1 - score,
                    "probability_persistent": score,
                }
            )
    contrast = pd.DataFrame(
        [
            {
                "task": "binary",
                "model": "lstm",
                "model_seed": 11,
                "metric": "auroc",
                "left_condition": "ordered_train_ordered_test",
                "right_condition": "ordered_train_shuffled_test",
                "left_value": 1.0,
                "right_value": 0.5,
                "ordered_minus_control": 0.5,
            }
        ]
    )
    config = {
        "metrics": {
            "binary": {
                "labels": ["transient", "persistent"],
                "positive_label": "persistent",
            },
            "bootstrap": {
                "n_resamples": 50,
                "confidence_level": 0.95,
                "seed": 7,
                "missing_class_policy": "skip",
                "interval_method": "percentile",
            },
        }
    }

    result = _add_bootstrap_intervals(contrast, pd.DataFrame(rows), config)

    assert bool(result.iloc[0]["bootstrap_clustered_by_subject"])
    assert result.iloc[0]["bootstrap_resamples_valid"] > 0


def test_non_estimable_secondary_does_not_discard_primary_results() -> None:
    events, labels = _dataset()
    config = _config()
    config["metrics"]["multiclass"] = {"labels": ["transient", "persistent", "relapsing"]}

    result = run_audit_experiments(
        {
            "binary": (events, labels),
            "multiclass": (events, labels),
        },
        config,
        enabled_models=["logistic_regression"],
        include_oracle=False,
    )

    assert set(result.seed_metrics["task"]) == {"binary"}
    status = result.task_estimability.set_index("task")
    assert bool(status.loc["binary", "estimable"])
    assert not bool(status.loc["multiclass", "estimable"])


def test_multiclass_bootstrap_auc_accepts_documented_phenotype_order() -> None:
    import numpy as np

    labels = ("transient", "persistent", "relapsing")
    truth = np.asarray(["transient", "persistent", "relapsing"] * 2)
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
        * 2
    )
    metric = _bootstrap_metric("multiclass", labels, {"metrics": {}})

    assert metric(truth, probabilities) == 1.0


def test_unreconstructible_oracle_is_explicitly_skipped(monkeypatch) -> None:
    events, labels = _dataset()

    def fail_oracle(*args, **kwargs):
        raise OracleNotEstimable("configured representation omits baseline evidence")

    monkeypatch.setattr(
        "src.mimic.aki.experiment.state_machine_oracle_probabilities",
        fail_oracle,
    )
    result = run_audit_experiments(
        {"binary": (events, labels)},
        _config(),
        enabled_models=[],
        include_oracle=True,
    )

    status = result.condition_estimability.iloc[0]
    assert not bool(status["estimable"])
    assert "omits baseline evidence" in status["reason"]
    assert result.seed_metrics.empty


def test_rare_secondary_skip_does_not_hide_patient_split_leakage() -> None:
    events, labels = _dataset()
    validation_index = labels.index[labels["split"] == "validation"][0]
    episode_id = labels.at[validation_index, "episode_id"]
    leaked_subject = labels.loc[labels["split"] == "train", "subject_id"].iloc[0]
    labels.loc[validation_index, "subject_id"] = leaked_subject
    events.loc[events["episode_id"] == episode_id, "subject_id"] = leaked_subject
    config = _config()
    config["metrics"]["multiclass"] = {"labels": ["transient", "persistent", "relapsing"]}

    with pytest.raises(AssertionError, match="more than one model split"):
        run_audit_experiments(
            {"multiclass": (events, labels)},
            config,
            enabled_models=[],
            include_oracle=False,
        )


def test_order_contrasts_include_nonlinear_invariant_control() -> None:
    metrics = pd.DataFrame(
        [
            {
                "task": "binary",
                "model": "lstm",
                "condition": "ordered_train_ordered_test",
                "feature_set": "ordered",
                "model_seed": 11,
                "auroc": 0.9,
            },
            {
                "task": "binary",
                "model": "xgboost",
                "condition": "trained_and_tested_same_representation",
                "feature_set": "invariant_summary",
                "model_seed": 11,
                "auroc": 0.7,
            },
        ]
    )

    contrasts = _contrasts(metrics)

    row = contrasts[contrasts["right_condition"] == "xgboost:invariant_summary"].iloc[0]
    assert row["ordered_minus_control"] == pytest.approx(0.2)


def test_tasks_must_reuse_one_subject_split_map() -> None:
    events, labels = _dataset()
    secondary_labels = labels.copy()
    train_index = secondary_labels.index[secondary_labels["split"] == "train"][0]
    secondary_labels.loc[train_index, "split"] = "validation"
    config = _config()
    config["metrics"]["multiclass"] = {"labels": ["transient", "persistent", "relapsing"]}

    with pytest.raises(AssertionError, match="inconsistent splits across tasks"):
        run_audit_experiments(
            {
                "binary": (events, labels),
                "multiclass": (events, secondary_labels),
            },
            config,
            enabled_models=[],
            include_oracle=False,
        )
