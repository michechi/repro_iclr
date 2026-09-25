from __future__ import annotations

import pandas as pd
import pytest

from src.mimic.aki.pipeline import (
    PopulationEpisodeResult,
    annotate_admission_censor_reasons,
    construct_population_episodes,
    persist_pipeline_result,
    prepare_analysis_datasets,
)


def _analysis_config() -> dict:
    return {
        "episodes": {
            "recovery_timing_sensitivity": {
                "enabled": False,
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
            }
        },
        "representations": {
            "window_start_column": "baseline_start_time",
            "window_end_column": "follow_up_end_time",
            "window_start_boundary": "inclusive",
            "window_end_boundary": "inclusive",
            "max_length": 16,
            "truncation": "error",
            "value_transform": "raw",
            "time_features": ["elapsed_hours"],
            "summary_features": ["count", "min", "max", "mean", "duration_hours"],
            "slope_hours_per_unit": 1.0,
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
            "minimum_pairs_per_split": {
                "train": 1,
                "validation": 1,
                "test": 1,
            },
            "maximum_absolute_smd": 0.0,
        },
        "metrics": {
            "multiclass": {
                "labels": ["transient", "persistent", "relapsing"],
            }
        },
    }


def _synthetic_population() -> tuple[pd.DataFrame, PopulationEpisodeResult]:
    classes = ["transient", "persistent", "relapsing"]
    measurement_rows = []
    episode_rows = []
    for subject_id in range(18):
        label = classes[subject_id % len(classes)]
        hadm_id = 10_000 + subject_id
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=subject_id * 10)
        episode_id = f"episode-{subject_id}"
        for offset, value in enumerate((1.0, 1.4, 1.0)):
            measurement_rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "specimen_time": start + pd.Timedelta(hours=offset),
                    "creatinine_mg_dl": value,
                }
            )
        episode_rows.append(
            {
                "episode_id": episode_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "phenotype": label,
                "status": "labeled",
                "include_in_analysis": True,
                "reason_code": "synthetic",
                "baseline_value": 1.0,
                "peak_value": 1.4,
                "baseline_start_time": start,
                "onset_time": start + pd.Timedelta(hours=1),
                "follow_up_end_time": start + pd.Timedelta(hours=2),
            }
        )
    episodes = pd.DataFrame(episode_rows)
    audit = episodes.assign(
        admission_censor_annotation_status="not_applicable",
        admission_censor_reason="not_applicable",
    )
    return (
        pd.DataFrame(measurement_rows),
        PopulationEpisodeResult(episodes=episodes, audit=audit),
    )


def _censor_config() -> dict:
    return {
        "episodes": {
            "admission_censoring": {
                "hadm_id_column": "hadm_id",
                "discharge_time_column": "dischtime",
                "death_time_column": "deathtime",
                "endpoint_precedence": "death_then_discharge",
                "horizon_boundary": "inclusive",
                "unknown_endpoint_reason": "missing_admission_endpoint",
            }
        }
    }


def _population_episode_config() -> dict:
    return {
        "episodes": {
            "columns": {
                "timestamp": "specimen_time",
                "value": "creatinine_mg_dl",
                "subject_id": "subject_id",
                "hadm_id": "hadm_id",
            },
            "admission_bounded": True,
            "baseline": {
                "method": "minimum",
                "lookback_hours": 48,
                "minimum_measurements": 1,
            },
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
            "duplicates": {
                "exact": "drop",
                "conflicting_timestamp": "ambiguous",
            },
        }
    }


def _population_measurements() -> pd.DataFrame:
    rows = []
    for subject_id, phenotype in enumerate(
        ("transient", "persistent", "relapsing", "no_aki"), start=1
    ):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=subject_id * 20)
        for hour in range(0, 193, 12):
            if phenotype == "no_aki" or hour < 24:
                value = 1.0
            elif phenotype == "persistent":
                value = 1.4
            elif phenotype == "transient":
                value = 1.4 if hour < 36 else 1.0
            else:
                value = 1.4 if hour < 36 or hour >= 96 else 1.0
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": 10_000 + subject_id,
                    "specimen_time": start + pd.Timedelta(hours=hour),
                    "creatinine_mg_dl": value,
                }
            )
    measurements = pd.DataFrame(rows)
    duplicate_time = measurements.loc[
        measurements["subject_id"] == 1, "specimen_time"
    ].iloc[2]
    duplicate = measurements.loc[
        (measurements["subject_id"] == 1)
        & measurements["specimen_time"].eq(duplicate_time)
    ]
    return pd.concat([measurements, duplicate], ignore_index=True).sample(
        frac=1.0, random_state=811
    )


def test_prepare_analysis_datasets_reuses_one_split_and_matches_within_split() -> None:
    measurements, population = _synthetic_population()
    result = prepare_analysis_datasets(measurements, population, _analysis_config())

    assert result.patient_splits["subject_id"].is_unique
    split_lookup = result.patient_splits.set_index("subject_id")["split"]
    for labels in (result.primary_labels, result.secondary_labels):
        observed = labels.set_index("subject_id")["split"]
        assert observed.to_dict() == split_lookup.loc[observed.index].to_dict()
    assert len(result.secondary_labels) == 18
    assert len(result.primary_labels) == 12
    assert result.matching.estimable

    assignments = result.matching.assignments
    for (_, split), pair in assignments.groupby(["pair_id", "split"]):
        assert set(pair["label"]) == {"transient", "persistent"}
        assert pair["subject_id"].nunique() == 2
        assert set(
            result.primary_matched_labels.loc[
                result.primary_matched_labels["pair_id"] == pair["pair_id"].iloc[0], "split"
            ]
        ) == {split}


def test_admission_censor_annotation_preserves_state_machine_reason() -> None:
    decisions = pd.DataFrame(
        {
            "hadm_id": [1, 2, 3],
            "status": ["censored", "censored", "labeled"],
            "reason_code": ["endpoint_not_observed", "measurement_gap", "classified"],
            "required_observation_end_time": pd.to_datetime(
                ["2024-01-08", "2024-01-08", "2024-01-08"]
            ),
        }
    )
    admissions = pd.DataFrame(
        {
            "hadm_id": [1, 2, 3],
            "dischtime": pd.to_datetime(["2024-01-05", "2024-01-09", "2024-01-03"]),
            "deathtime": pd.to_datetime(["2024-01-04", None, None]),
        }
    )

    annotated = annotate_admission_censor_reasons(decisions, admissions, _censor_config())
    assert annotated["state_machine_reason_code"].tolist() == decisions["reason_code"].tolist()
    assert annotated.loc[0, "admission_censor_reason"] == "death_before_required_horizon"
    assert annotated.loc[0, "admission_censor_endpoint_type"] == "death"
    assert annotated.loc[1, "admission_censor_annotation_status"] == "not_attributed"
    assert annotated.loc[2, "admission_censor_annotation_status"] == "not_applicable"

    unevaluated = annotate_admission_censor_reasons(decisions, None, _censor_config())
    assert set(
        unevaluated.loc[unevaluated["status"] == "censored", "admission_censor_annotation_status"]
    ) == {"not_evaluated"}


def test_pipeline_persistence_writes_parquet_tables_and_csv_summaries(tmp_path) -> None:
    measurements, population = _synthetic_population()
    result = prepare_analysis_datasets(measurements, population, _analysis_config())

    artifacts = persist_pipeline_result(result, tmp_path / "audit")

    assert artifacts.parquet["episodes"].is_file()
    assert artifacts.csv_summaries["episode_flow"].is_file()
    assert len(pd.read_parquet(artifacts.parquet["patient_splits"])) == 18
    assert not pd.read_csv(artifacts.csv_summaries["matching_flow"]).empty


def test_non_estimable_matching_still_returns_auditable_empty_model_tables() -> None:
    measurements, population = _synthetic_population()
    episodes = population.episodes.copy()
    episodes.loc[episodes["phenotype"] == "persistent", "peak_value"] = 9.0
    altered = PopulationEpisodeResult(episodes=episodes, audit=population.audit)

    result = prepare_analysis_datasets(measurements, altered, _analysis_config())

    assert not result.matching.estimable
    assert result.primary_matched_labels.empty
    assert result.primary_matched_events.empty
    assert {"episode_id", "pair_id"}.issubset(result.matching.assignments.columns)


def test_population_episode_parallel_output_exactly_matches_serial_output() -> None:
    measurements = _population_measurements()
    config = _population_episode_config()

    serial = construct_population_episodes(measurements, config)
    parallel = construct_population_episodes(
        measurements,
        config,
        max_workers=2,
        patient_chunk_size=2,
    )

    for attribute in ("episodes", "audit", "resolved_measurements"):
        pd.testing.assert_frame_equal(
            getattr(parallel, attribute),
            getattr(serial, attribute),
            check_exact=True,
        )


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"max_workers": 0}, ValueError, "max_workers must be at least 1"),
        ({"max_workers": True}, TypeError, "max_workers must be an integer"),
        (
            {"patient_chunk_size": 0},
            ValueError,
            "patient_chunk_size must be at least 1",
        ),
    ],
)
def test_population_episode_execution_parameters_fail_fast(
    kwargs: dict, exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        construct_population_episodes(
            _population_measurements(),
            _population_episode_config(),
            **kwargs,
        )


def test_population_episode_parallel_worker_errors_are_propagated() -> None:
    measurements = _population_measurements()
    measurements["creatinine_mg_dl"] = measurements["creatinine_mg_dl"].astype(object)
    measurements.loc[measurements["subject_id"] == 2, "creatinine_mg_dl"] = "invalid"

    with pytest.raises(ValueError, match="finite numeric creatinine values"):
        construct_population_episodes(
            measurements,
            _population_episode_config(),
            max_workers=2,
            patient_chunk_size=1,
        )
