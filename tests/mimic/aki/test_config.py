from copy import deepcopy

import pytest

from src.mimic.aki.config import AkiConfigError, aki_config_from_mapping
from src.mimic.aki.reporting import config_digest


def _valid_mapping() -> dict:
    """Complete synthetic protocol values; none are production defaults."""

    return {
        "schema_version": 1,
        "cohort": {
            "creatinine_itemid": 50912,
            "adult": {
                "minimum_age_years": 18,
                "minimum_age_inclusive": True,
                "age_calculation": "anchor_year_delta",
            },
            "admission": {
                "eligible_types": "all",
                "type_matching": "exact",
                "minimum_stay_hours": 0,
                "minimum_stay_inclusive": True,
                "charttime_boundary": "both",
                "null_hadm_policy": "reject",
                "administrative_end_policy": "earliest_discharge_or_death",
                "invalid_death_time_policy": "exclude_admission",
            },
            "minimum_measurements": 3,
            "measurement_count_basis": "distinct_timestamps",
            "missing_specimen_id_policy": "exclude_from_count",
        },
        "cleaning": {
            "numeric_value_source": "valuenum_only",
            "canonical_unit": "mg/dL",
            "unit_matching": "casefold_strip",
            "missing_unit_policy": "reject",
            "unit_conversions": [{"aliases": ["mg/dL"], "multiplier": 1.0, "offset": 0.0}],
            "valid_range_mg_dl": {"minimum": 0.1, "maximum": 30, "boundary": "both"},
            "deduplication": {
                "ordering": "labevent_id",
                "exact": {
                    "key": [
                        "subject_id",
                        "hadm_id",
                        "specimen_time",
                        "creatinine_mg_dl",
                    ],
                    "policy": "keep_first",
                },
                "simultaneous": {
                    "enabled": True,
                    "key": ["subject_id", "hadm_id", "specimen_time"],
                    "value_tolerance_mg_dl": 0,
                    "concordant_policy": "keep_first",
                    "conflicting_policy": "reject_all",
                },
            },
        },
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
            "max_length": 128,
            "truncation": "error",
            "value_transform": "raw",
            "time_features": ["elapsed_hours", "delta_hours"],
            "summary_features": [
                "count",
                "distinct_count",
                "min",
                "max",
                "mean",
                "std",
                "median",
                "range",
                "repeated_min_count",
                "repeated_max_count",
                "duration_hours",
                "gap_mean_hours",
                "gap_std_hours",
                "gap_min_hours",
                "gap_max_hours",
            ],
            "slope_hours_per_unit": 24,
            "permutation_seeds": [101, 202, 303],
        },
        "matching": {
            "class_a": "transient",
            "class_b": "persistent",
            "selection_seed": 404,
            "coarsening": {
                "count": {"method": "exact"},
                "baseline_creatinine_mg_dl": {"method": "width", "width": 0.25, "origin": 0},
                "peak_value": {"method": "width", "width": 0.25, "origin": 0},
                "duration_hours": {
                    "method": "edges",
                    "edges": [0, 48, 96, 240],
                    "right": True,
                },
            },
            "minimum_pairs_per_split": {"train": 1, "validation": 1, "test": 1},
            "maximum_absolute_smd": 0.1,
        },
        "splitting": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "stratification": "phenotype_signature",
            "split_seed": 505,
            "rare_stratum_policy": "error",
        },
        "models": {
            "seeds": [11, 22, 33],
            "tabular_preprocessing": {"scaling": "standard", "missing_values": "median"},
            "sequence_preprocessing": {"scaling": "standard", "missing_values": "error"},
            "sequence_training": {
                "batch_size": 32,
                "max_epochs": 20,
                "learning_rate": 0.001,
                "weight_decay": 0,
                "optimizer": "adamw",
                "patience": 3,
                "min_delta": 0,
                "early_stopping_metric": "validation_loss",
                "device": "cpu",
                "num_workers": 0,
                "pin_memory": False,
                "deterministic_algorithms": True,
                "class_weight": None,
                "gradient_clip_norm": None,
                "optimizer_parameters": {},
            },
            "lstm": {
                "hidden_dim": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "bidirectional": False,
                "projection_dim": None,
            },
            "transformer": {
                "d_model": 32,
                "nhead": 4,
                "num_layers": 1,
                "dim_feedforward": 64,
                "dropout": 0.1,
                "max_sequence_length": 128,
                "positional_encoding": "sinusoidal",
                "pooling": "mean",
            },
            "logistic_regression": {
                "C": 1,
                "penalty": "l2",
                "solver": "liblinear",
                "max_iter": 500,
                "tol": 0.0001,
                "fit_intercept": True,
                "class_weight": None,
                "n_jobs": None,
            },
            "xgboost": {
                "parameters": {
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "min_child_weight": 1,
                    "subsample": 1,
                    "colsample_bytree": 1,
                    "reg_alpha": 0,
                    "reg_lambda": 1,
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "n_jobs": -1,
                    "tree_method": "hist",
                    "verbosity": 0,
                },
                "fit_parameters": {},
                "class_weight": None,
            },
        },
        "metrics": {
            "binary": {
                "labels": ["transient", "persistent"],
                "threshold": 0.5,
                "positive_label": "persistent",
                "missing_class_policy": "raise",
                "zero_division": 0,
            },
            "multiclass": {
                "labels": ["transient", "persistent", "relapsing"],
                "decision_rule": "argmax",
                "missing_class_policy": "raise",
                "zero_division": 0,
            },
            "aggregation": {"ddof": 1},
            "bootstrap": {
                "n_resamples": 100,
                "confidence_level": 0.95,
                "seed": 606,
                "missing_class_policy": "skip",
                "interval_method": "percentile",
            },
        },
    }


def test_complete_protocol_is_hash_stable_and_detached() -> None:
    mapping = _valid_mapping()
    first = aki_config_from_mapping(mapping)
    second = aki_config_from_mapping(deepcopy(mapping))

    assert first.config_hash == second.config_hash
    assert config_digest(first) == first.config_hash
    mapping["episodes"]["onset"]["absolute_rise_mg_dl"] = 99
    assert first.episodes["onset"]["absolute_rise_mg_dl"] == 0.3
    with pytest.raises(TypeError):
        first.episodes["onset"]["absolute_rise_mg_dl"] = 1.0
    with pytest.raises(TypeError):
        first.representations["time_features"][0] = "delta_hours"
    detached = first.as_dict()
    detached["episodes"]["onset"]["absolute_rise_mg_dl"] = 77
    assert first.episodes["onset"]["absolute_rise_mg_dl"] == 0.3


def test_invalid_admission_censor_choice_fails_during_config_load() -> None:
    mapping = _valid_mapping()
    mapping["episodes"]["admission_censoring"]["horizon_boundary"] = "guess"

    with pytest.raises(AkiConfigError, match="horizon_boundary"):
        aki_config_from_mapping(mapping)


def test_protocol_requires_three_unique_model_and_permutation_seeds() -> None:
    mapping = _valid_mapping()
    mapping["models"]["seeds"] = [11, 11, 33]

    with pytest.raises(AkiConfigError, match="three unique integer seeds"):
        aki_config_from_mapping(mapping)


def test_explicit_class_weights_must_cover_both_task_label_universe() -> None:
    mapping = _valid_mapping()
    mapping["models"]["logistic_regression"]["class_weight"] = {
        "transient": 1.0,
        "persistent": 2.0,
    }

    with pytest.raises(AkiConfigError, match="full binary/multiclass label universe"):
        aki_config_from_mapping(mapping)


def test_xgboost_objective_must_produce_class_probabilities() -> None:
    mapping = _valid_mapping()
    mapping["models"]["xgboost"]["parameters"]["objective"] = "multi:softmax"

    with pytest.raises(AkiConfigError, match="multi:softprob"):
        aki_config_from_mapping(mapping)


@pytest.mark.parametrize(
    ("section", "nested"),
    [
        ("episodes", "baseline"),
        ("representations", None),
    ],
)
def test_unknown_scientific_keys_fail_fast(section: str, nested: str | None) -> None:
    mapping = _valid_mapping()
    target = mapping[section] if nested is None else mapping[section][nested]
    target["typo_scientific_choice"] = 123

    with pytest.raises(AkiConfigError, match="unknown keys"):
        aki_config_from_mapping(mapping)


def test_matching_smd_override_keys_must_be_coarsened_features() -> None:
    mapping = _valid_mapping()
    mapping["matching"]["maximum_absolute_smd_overrides"] = {"not_coarsened": 0.15}

    with pytest.raises(AkiConfigError, match="configured matching.coarsening features"):
        aki_config_from_mapping(mapping)


@pytest.mark.parametrize("threshold", [-0.01, "0.15", True, float("inf")])
def test_matching_smd_overrides_must_be_finite_nonnegative_numbers(threshold: object) -> None:
    mapping = _valid_mapping()
    mapping["matching"]["maximum_absolute_smd_overrides"] = {
        "duration_hours": threshold
    }

    with pytest.raises(AkiConfigError, match="maximum_absolute_smd_overrides.duration_hours"):
        aki_config_from_mapping(mapping)


def test_matching_smd_override_changes_protocol_hash() -> None:
    original = _valid_mapping()
    amended = deepcopy(original)
    amended["matching"]["maximum_absolute_smd_overrides"] = {"duration_hours": 0.15}

    assert (
        aki_config_from_mapping(original).config_hash
        != aki_config_from_mapping(amended).config_hash
    )


def test_impossible_representation_window_column_fails_before_data_access() -> None:
    mapping = _valid_mapping()
    mapping["representations"]["window_start_column"] = "definitely_not_an_episode_column"

    with pytest.raises(AkiConfigError, match="guaranteed on every included episode"):
        aki_config_from_mapping(mapping)


def test_nullable_recurrence_time_cannot_bound_shared_model_window() -> None:
    mapping = _valid_mapping()
    mapping["representations"]["window_end_column"] = "recurrence_time"

    with pytest.raises(AkiConfigError, match="guaranteed on every included episode"):
        aki_config_from_mapping(mapping)
