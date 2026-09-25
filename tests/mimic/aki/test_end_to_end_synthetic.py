import pandas as pd

from src.mimic.aki.experiment import state_machine_oracle_probabilities
from src.mimic.aki.pipeline import build_aki_audit_datasets


def _config() -> dict:
    return {
        "episodes": {
            "columns": {
                "timestamp": "specimen_time",
                "value": "creatinine_mg_dl",
                "subject_id": "subject_id",
                "hadm_id": "hadm_id",
            },
            "admission_bounded": True,
            "baseline": {"method": "minimum", "lookback_hours": 48, "minimum_measurements": 1},
            "onset": {
                "absolute_rise_mg_dl": 0.3,
                "window_hours": 48,
                "boundary": "inclusive",
                "tie_breaker": "earliest",
            },
            "recovery": {
                "threshold_above_baseline_mg_dl": 0.3,
                "window_hours": 48,
                "time_boundary": "inclusive",
                "value_comparison": "lt",
                "sustained_hours": 48,
            },
            "follow_up": {"duration_days": 7, "boundary": "inclusive"},
            "peak": {
                "window_end": "follow_up",
                "boundary": "inclusive",
                "tie_breaker": "earliest",
            },
            "relapse": {
                "window_days": 7,
                "anchor": "onset",
                "comparator": "recovery_nadir",
                "boundary": "inclusive",
            },
            "observation": {
                "max_gap_hours": 24,
                "endpoint_tolerance_hours": 0,
                "minimum_post_onset_measurements": 1,
            },
            "duplicates": {"exact": "drop", "conflicting_timestamp": "ambiguous"},
            "admission_censoring": {
                "hadm_id_column": "hadm_id",
                "discharge_time_column": "dischtime",
                "death_time_column": "deathtime",
                "endpoint_precedence": "death_then_discharge",
                "horizon_boundary": "inclusive",
                "unknown_endpoint_reason": "missing_admission_endpoint",
            },
            "recovery_timing_sensitivity": {
                "enabled": True,
                "reference_time_column": "peak_time",
                "baseline_column": "baseline_value",
                "threshold_above_baseline_mg_dl": 0.3,
                "value_comparison": "lt",
                "fast_max_hours": 48,
                "intermediate_max_hours": 240,
                "time_boundary": "inclusive",
                "sustained_hours": 0,
                "max_gap_hours": 24,
                "endpoint_tolerance_hours": 0,
                "minimum_measurements": 1,
                "duplicate_policy": "error",
            },
        },
        "representations": {
            "window_start_column": "baseline_start_time",
            "window_end_column": "follow_up_end_time",
            "window_start_boundary": "inclusive",
            "window_end_boundary": "inclusive",
            "max_length": 32,
            "truncation": "error",
            "value_transform": "raw",
            "time_features": ["elapsed_hours"],
            "summary_features": ["count", "min", "max", "duration_hours"],
            "slope_hours_per_unit": 24,
        },
        "splitting": {
            "train_fraction": 0.5,
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "stratification": "single_label",
            "split_seed": 73,
            "rare_stratum_policy": "error",
        },
        "matching": {
            "class_a": "transient",
            "class_b": "persistent",
            "selection_seed": 91,
            "coarsening": {
                "count": {"method": "exact"},
                "min": {"method": "exact"},
                "max": {"method": "exact"},
                "duration_hours": {"method": "exact"},
                "baseline_creatinine_mg_dl": {"method": "exact"},
                "peak_value": {"method": "exact"},
            },
            "minimum_pairs_per_split": {"train": 1, "validation": 1, "test": 1},
            "maximum_absolute_smd": 0,
        },
        "metrics": {"multiclass": {"labels": ["transient", "persistent", "relapsing"]}},
    }


def _patients() -> tuple[pd.DataFrame, pd.DataFrame]:
    measurement_rows = []
    admission_rows = []
    phenotypes = ("transient", "persistent", "relapsing")
    for offset in range(18):
        subject_id = offset + 1
        hadm_id = 10_000 + subject_id
        phenotype = phenotypes[offset % 3]
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=20 * offset)
        for hour in range(0, 193, 12):
            if hour < 24:
                value = 1.0
            elif phenotype == "persistent":
                value = 1.4
            elif phenotype == "transient":
                value = 1.4 if hour < 36 else 1.0
            else:
                value = 1.4 if hour < 36 or hour >= 96 else 1.0
            measurement_rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "specimen_time": start + pd.Timedelta(hours=hour),
                    "creatinine_mg_dl": value,
                }
            )
        admission_rows.append(
            {
                "hadm_id": hadm_id,
                "dischtime": start + pd.Timedelta(hours=200),
                "deathtime": pd.NaT,
            }
        )
    return pd.DataFrame(measurement_rows), pd.DataFrame(admission_rows)


def test_synthetic_patients_flow_from_measurements_to_matched_model_tables() -> None:
    measurements, admissions = _patients()

    result = build_aki_audit_datasets(measurements, _config(), admissions=admissions)

    assert result.episodes["phenotype"].value_counts().to_dict() == {
        "transient": 6,
        "persistent": 6,
        "relapsing": 6,
    }
    assert result.episode_audit["status"].eq("labeled").all()
    assert len(result.secondary_labels) == 18
    assert result.patient_splits["subject_id"].is_unique
    assert result.matching.estimable
    assert not result.primary_matched_events.empty
    assert set(result.primary_matched_labels["label"]) == {"transient", "persistent"}
    assert set(result.recovery_timing_sensitivity["sensitivity_status"]) == {
        "classified",
        "censored",
    }
    test_labels = result.secondary_labels[result.secondary_labels["split"] == "test"].sort_values(
        "episode_id"
    )
    oracle = state_machine_oracle_probabilities(
        result.secondary_events,
        test_labels,
        _config(),
        ("transient", "persistent", "relapsing"),
    )
    assert (oracle.sum(axis=1) == 1).all()


def test_pipeline_models_the_same_duplicate_resolved_events_used_for_labels() -> None:
    measurements, admissions = _patients()
    duplicate_rows = measurements.groupby("subject_id", sort=False).head(1)
    with_duplicates = pd.concat([measurements, duplicate_rows], ignore_index=True)

    result = build_aki_audit_datasets(with_duplicates, _config(), admissions=admissions)

    assert set(result.episode_audit["exact_duplicates_removed"]) == {1}
    modeled_counts = result.labeled_episode_events.groupby("episode_id").size()
    assert modeled_counts.eq(17).all()
