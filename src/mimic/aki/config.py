"""Strict protocol configuration for the MIMIC-IV AKI trajectory audit.

Clinical and scientific choices have no defaults in this module.  A protocol
must name every required choice, even when the choice is to disable a filter.
This prevents a missing YAML key from silently changing the cohort or target.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = 1
CREATININE_ITEMID = 50912
CREATININE_IDENTITY = {
    "label": "Creatinine",
    "fluid": "Blood",
    "category": "Chemistry",
}


class AkiConfigError(ValueError):
    """Raised when an AKI audit protocol is incomplete or invalid."""


def require_keys(
    value: Mapping[str, Any],
    keys: Sequence[str],
    context: str,
    *,
    allow_extra: bool = True,
) -> None:
    """Require mapping keys and optionally reject keys outside the schema."""

    missing = [key for key in keys if key not in value]
    if missing:
        raise AkiConfigError(f"{context} is missing required keys: {missing}")
    if not allow_extra:
        extra = sorted(set(value).difference(keys))
        if extra:
            raise AkiConfigError(f"{context} contains unknown keys: {extra}")


def require_path(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Return a required nested value, raising with its full dotted path."""

    current: Any = value
    dotted = ".".join(path)
    for index, key in enumerate(path):
        if not isinstance(current, Mapping) or key not in current:
            parent = ".".join(path[:index]) or "configuration"
            raise AkiConfigError(f"{dotted} is required (missing below {parent})")
        current = current[key]
    if current is None:
        raise AkiConfigError(f"{dotted} is required and cannot be null")
    return current


# These paths are explicitly nullable model API choices.  No clinical field is
# nullable.  Absence and null remain different: each path is still required.
ALLOWED_NULL_PATHS = frozenset(
    {
        "models.sequence_training.class_weight",
        "models.sequence_training.gradient_clip_norm",
        "models.lstm.projection_dim",
        "models.logistic_regression.penalty",
        "models.logistic_regression.class_weight",
        "models.logistic_regression.n_jobs",
        "models.xgboost.class_weight",
    }
)

LABEL_KEYED_MAPPING_PATHS = frozenset(
    {
        "models.sequence_training.class_weight",
        "models.logistic_regression.class_weight",
        "models.xgboost.class_weight",
    }
)


def validate_no_null_scientific_values(
    value: Any,
    *,
    path: str = "",
    allowed_null_paths: frozenset[str] = ALLOWED_NULL_PATHS,
) -> None:
    """Recursively reject null and non-finite protocol values.

    This helper is public so downstream configuration extensions can enforce
    the same fail-fast contract.
    """

    if value is None:
        if path not in allowed_null_paths:
            raise AkiConfigError(f"{path or 'configuration'} cannot be null")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise AkiConfigError(f"{path or 'configuration'} must be finite")
    if isinstance(value, (str, int, float)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            label_key_allowed = path in LABEL_KEYED_MAPPING_PATHS
            valid_label_key = (
                label_key_allowed
                and not isinstance(key, bool)
                and isinstance(key, (str, int, float))
            )
            if not valid_label_key and (not isinstance(key, str) or not key):
                raise AkiConfigError(f"{path or 'configuration'} has an invalid mapping key")
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            validate_no_null_scientific_values(
                child, path=child_path, allowed_null_paths=allowed_null_paths
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            validate_no_null_scientific_values(
                child,
                path=f"{path}[{index}]",
                allowed_null_paths=allowed_null_paths,
            )
        return
    raise AkiConfigError(
        f"{path or 'configuration'} has unsupported value type {type(value).__name__}"
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AkiConfigError(f"{path} must be a mapping")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AkiConfigError(f"{path} must be a non-empty string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise AkiConfigError(f"{path} must be a boolean")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AkiConfigError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise AkiConfigError(f"{path} must be finite")
    if minimum is not None and result < minimum:
        raise AkiConfigError(f"{path} must be >= {minimum}")
    return result


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    number = _number(value, path)
    if not number.is_integer():
        raise AkiConfigError(f"{path} must be an integer")
    result = int(number)
    if minimum is not None and result < minimum:
        raise AkiConfigError(f"{path} must be >= {minimum}")
    return result


def _choice(value: Any, path: str, choices: set[str]) -> str:
    result = _string(value, path)
    if result not in choices:
        allowed = ", ".join(sorted(choices))
        raise AkiConfigError(f"{path} must be one of {{{allowed}}}, got {result!r}")
    return result


def _string_list(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise AkiConfigError(f"{path} must be {qualifier} of strings")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise AkiConfigError(f"{path} contains duplicate values")
    return result


@dataclass(frozen=True)
class AdultConfig:
    minimum_age_years: float
    minimum_age_inclusive: bool
    age_calculation: str


@dataclass(frozen=True)
class AdmissionConfig:
    eligible_types: str | tuple[str, ...]
    type_matching: str
    minimum_stay_hours: float
    minimum_stay_inclusive: bool
    charttime_boundary: str
    null_hadm_policy: str
    administrative_end_policy: str
    invalid_death_time_policy: str


@dataclass(frozen=True)
class CohortConfig:
    creatinine_itemid: int
    adult: AdultConfig
    admission: AdmissionConfig
    minimum_measurements: int
    measurement_count_basis: str
    missing_specimen_id_policy: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CohortConfig":
        require_keys(
            value,
            (
                "creatinine_itemid",
                "adult",
                "admission",
                "minimum_measurements",
                "measurement_count_basis",
                "missing_specimen_id_policy",
            ),
            "cohort",
            allow_extra=False,
        )
        itemid = _integer(value["creatinine_itemid"], "cohort.creatinine_itemid")
        if itemid != CREATININE_ITEMID:
            raise AkiConfigError(
                f"cohort.creatinine_itemid is locked to {CREATININE_ITEMID}, got {itemid}"
            )

        adult_raw = _mapping(value["adult"], "cohort.adult")
        require_keys(
            adult_raw,
            ("minimum_age_years", "minimum_age_inclusive", "age_calculation"),
            "cohort.adult",
            allow_extra=False,
        )
        adult = AdultConfig(
            minimum_age_years=_number(
                adult_raw["minimum_age_years"], "cohort.adult.minimum_age_years", minimum=0
            ),
            minimum_age_inclusive=_boolean(
                adult_raw["minimum_age_inclusive"], "cohort.adult.minimum_age_inclusive"
            ),
            age_calculation=_choice(
                adult_raw["age_calculation"],
                "cohort.adult.age_calculation",
                {"anchor_year_delta"},
            ),
        )

        admission_raw = _mapping(value["admission"], "cohort.admission")
        require_keys(
            admission_raw,
            (
                "eligible_types",
                "type_matching",
                "minimum_stay_hours",
                "minimum_stay_inclusive",
                "charttime_boundary",
                "null_hadm_policy",
                "administrative_end_policy",
                "invalid_death_time_policy",
            ),
            "cohort.admission",
            allow_extra=False,
        )
        eligible_raw = admission_raw["eligible_types"]
        if eligible_raw == "all":
            eligible_types: str | tuple[str, ...] = "all"
        else:
            eligible_types = _string_list(eligible_raw, "cohort.admission.eligible_types")
        admission = AdmissionConfig(
            eligible_types=eligible_types,
            type_matching=_choice(
                admission_raw["type_matching"],
                "cohort.admission.type_matching",
                {"exact", "casefold"},
            ),
            minimum_stay_hours=_number(
                admission_raw["minimum_stay_hours"],
                "cohort.admission.minimum_stay_hours",
                minimum=0,
            ),
            minimum_stay_inclusive=_boolean(
                admission_raw["minimum_stay_inclusive"],
                "cohort.admission.minimum_stay_inclusive",
            ),
            charttime_boundary=_choice(
                admission_raw["charttime_boundary"],
                "cohort.admission.charttime_boundary",
                {"both", "left", "right", "neither"},
            ),
            null_hadm_policy=_choice(
                admission_raw["null_hadm_policy"],
                "cohort.admission.null_hadm_policy",
                {"reject", "link_if_unique"},
            ),
            administrative_end_policy=_choice(
                admission_raw["administrative_end_policy"],
                "cohort.admission.administrative_end_policy",
                {"discharge_only", "earliest_discharge_or_death"},
            ),
            invalid_death_time_policy=_choice(
                admission_raw["invalid_death_time_policy"],
                "cohort.admission.invalid_death_time_policy",
                {"exclude_admission", "ignore"},
            ),
        )
        minimum_measurements = _integer(
            value["minimum_measurements"], "cohort.minimum_measurements", minimum=3
        )
        measurement_count_basis = _choice(
            value["measurement_count_basis"],
            "cohort.measurement_count_basis",
            {"rows", "distinct_timestamps", "distinct_specimens"},
        )
        missing_specimen_id_policy = _choice(
            value["missing_specimen_id_policy"],
            "cohort.missing_specimen_id_policy",
            {"reject_measurement", "exclude_from_count", "count_as_unique"},
        )
        return cls(
            creatinine_itemid=itemid,
            adult=adult,
            admission=admission,
            minimum_measurements=minimum_measurements,
            measurement_count_basis=measurement_count_basis,
            missing_specimen_id_policy=missing_specimen_id_policy,
        )


@dataclass(frozen=True)
class UnitConversion:
    aliases: tuple[str, ...]
    multiplier: float
    offset: float


@dataclass(frozen=True)
class DeduplicationConfig:
    ordering: str
    exact_key: tuple[str, ...]
    exact_policy: str
    simultaneous_enabled: bool
    simultaneous_key: tuple[str, ...]
    value_tolerance_mg_dl: float
    concordant_policy: str
    conflicting_policy: str


@dataclass(frozen=True)
class CleaningConfig:
    numeric_value_source: str
    canonical_unit: str
    unit_matching: str
    missing_unit_policy: str
    unit_conversions: tuple[UnitConversion, ...]
    minimum_value_mg_dl: float
    maximum_value_mg_dl: float
    value_boundary: str
    deduplication: DeduplicationConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CleaningConfig":
        require_keys(
            value,
            (
                "numeric_value_source",
                "canonical_unit",
                "unit_matching",
                "missing_unit_policy",
                "unit_conversions",
                "valid_range_mg_dl",
                "deduplication",
            ),
            "cleaning",
            allow_extra=False,
        )
        canonical_unit = _string(value["canonical_unit"], "cleaning.canonical_unit")
        if canonical_unit != "mg/dL":
            raise AkiConfigError("cleaning.canonical_unit is locked to 'mg/dL' for the AKI audit")
        unit_matching = _choice(
            value["unit_matching"],
            "cleaning.unit_matching",
            {"exact", "strip", "casefold_strip"},
        )

        conversion_raw = value["unit_conversions"]
        if not isinstance(conversion_raw, list) or not conversion_raw:
            raise AkiConfigError("cleaning.unit_conversions must be a non-empty list")
        conversions: list[UnitConversion] = []
        for index, item in enumerate(conversion_raw):
            path = f"cleaning.unit_conversions[{index}]"
            row = _mapping(item, path)
            require_keys(
                row,
                ("aliases", "multiplier", "offset"),
                path,
                allow_extra=False,
            )
            conversions.append(
                UnitConversion(
                    aliases=_string_list(row["aliases"], f"{path}.aliases"),
                    multiplier=_number(row["multiplier"], f"{path}.multiplier"),
                    offset=_number(row["offset"], f"{path}.offset"),
                )
            )
        aliases: dict[str, tuple[float, float]] = {}
        for conversion in conversions:
            for alias in conversion.aliases:
                normalized = alias
                if unit_matching in {"strip", "casefold_strip"}:
                    normalized = normalized.strip()
                if unit_matching == "casefold_strip":
                    normalized = normalized.casefold()
                parameters = (conversion.multiplier, conversion.offset)
                if normalized in aliases and aliases[normalized] != parameters:
                    raise AkiConfigError(
                        f"cleaning.unit_conversions alias {alias!r} has conflicting conversions"
                    )
                aliases[normalized] = parameters
        canonical_alias = canonical_unit
        if unit_matching in {"strip", "casefold_strip"}:
            canonical_alias = canonical_alias.strip()
        if unit_matching == "casefold_strip":
            canonical_alias = canonical_alias.casefold()
        if aliases.get(canonical_alias) != (1.0, 0.0):
            raise AkiConfigError(
                "cleaning.unit_conversions must map canonical 'mg/dL' to multiplier=1, offset=0"
            )

        range_raw = _mapping(value["valid_range_mg_dl"], "cleaning.valid_range_mg_dl")
        require_keys(
            range_raw,
            ("minimum", "maximum", "boundary"),
            "cleaning.valid_range_mg_dl",
            allow_extra=False,
        )
        minimum = _number(range_raw["minimum"], "cleaning.valid_range_mg_dl.minimum")
        maximum = _number(range_raw["maximum"], "cleaning.valid_range_mg_dl.maximum")
        if minimum >= maximum:
            raise AkiConfigError("cleaning.valid_range_mg_dl.minimum must be < maximum")

        dedup_raw = _mapping(value["deduplication"], "cleaning.deduplication")
        require_keys(
            dedup_raw,
            ("ordering", "exact", "simultaneous"),
            "cleaning.deduplication",
            allow_extra=False,
        )
        exact_raw = _mapping(dedup_raw["exact"], "cleaning.deduplication.exact")
        require_keys(
            exact_raw,
            ("key", "policy"),
            "cleaning.deduplication.exact",
            allow_extra=False,
        )
        simultaneous_raw = _mapping(
            dedup_raw["simultaneous"], "cleaning.deduplication.simultaneous"
        )
        require_keys(
            simultaneous_raw,
            (
                "enabled",
                "key",
                "value_tolerance_mg_dl",
                "concordant_policy",
                "conflicting_policy",
            ),
            "cleaning.deduplication.simultaneous",
            allow_extra=False,
        )
        allowed_keys = {
            "labevent_id",
            "subject_id",
            "hadm_id",
            "specimen_id",
            "specimen_time",
            "creatinine_mg_dl",
            "raw_value",
            "raw_valuenum",
            "raw_unit",
        }
        exact_key = _string_list(exact_raw["key"], "cleaning.deduplication.exact.key")
        simultaneous_key = _string_list(
            simultaneous_raw["key"], "cleaning.deduplication.simultaneous.key"
        )
        for path, keys in (
            ("cleaning.deduplication.exact.key", exact_key),
            ("cleaning.deduplication.simultaneous.key", simultaneous_key),
        ):
            unknown = sorted(set(keys).difference(allowed_keys))
            if unknown:
                raise AkiConfigError(f"{path} has unsupported columns: {unknown}")
            if not {"subject_id", "hadm_id"}.issubset(keys):
                raise AkiConfigError(f"{path} must contain subject_id and hadm_id")

        deduplication = DeduplicationConfig(
            ordering=_choice(
                dedup_raw["ordering"],
                "cleaning.deduplication.ordering",
                {"labevent_id", "storetime"},
            ),
            exact_key=exact_key,
            exact_policy=_choice(
                exact_raw["policy"],
                "cleaning.deduplication.exact.policy",
                {"keep_first", "keep_last", "reject_all", "error"},
            ),
            simultaneous_enabled=_boolean(
                simultaneous_raw["enabled"],
                "cleaning.deduplication.simultaneous.enabled",
            ),
            simultaneous_key=simultaneous_key,
            value_tolerance_mg_dl=_number(
                simultaneous_raw["value_tolerance_mg_dl"],
                "cleaning.deduplication.simultaneous.value_tolerance_mg_dl",
                minimum=0,
            ),
            concordant_policy=_choice(
                simultaneous_raw["concordant_policy"],
                "cleaning.deduplication.simultaneous.concordant_policy",
                {"keep_first", "keep_last", "mean", "reject_all", "error"},
            ),
            conflicting_policy=_choice(
                simultaneous_raw["conflicting_policy"],
                "cleaning.deduplication.simultaneous.conflicting_policy",
                {"keep_first", "keep_last", "mean", "reject_all", "error"},
            ),
        )
        return cls(
            numeric_value_source=_choice(
                value["numeric_value_source"],
                "cleaning.numeric_value_source",
                {"valuenum_only", "valuenum_then_value"},
            ),
            canonical_unit=canonical_unit,
            unit_matching=unit_matching,
            missing_unit_policy=_choice(
                value["missing_unit_policy"],
                "cleaning.missing_unit_policy",
                {"reject", "assume_canonical"},
            ),
            unit_conversions=tuple(conversions),
            minimum_value_mg_dl=minimum,
            maximum_value_mg_dl=maximum,
            value_boundary=_choice(
                range_raw["boundary"],
                "cleaning.valid_range_mg_dl.boundary",
                {"both", "left", "right", "neither"},
            ),
            deduplication=deduplication,
        )


REQUIRED_SECTION_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "episodes": (
        ("columns", "timestamp"),
        ("columns", "value"),
        ("columns", "subject_id"),
        ("columns", "hadm_id"),
        ("admission_bounded",),
        ("baseline", "method"),
        ("baseline", "lookback_hours"),
        ("baseline", "minimum_measurements"),
        ("onset", "absolute_rise_mg_dl"),
        ("onset", "window_hours"),
        ("onset", "boundary"),
        ("onset", "tie_breaker"),
        ("recovery", "threshold_above_baseline_mg_dl"),
        ("recovery", "window_hours"),
        ("recovery", "time_boundary"),
        ("recovery", "value_comparison"),
        ("recovery", "sustained_hours"),
        ("follow_up", "duration_days"),
        ("follow_up", "boundary"),
        ("peak", "window_end"),
        ("peak", "boundary"),
        ("peak", "tie_breaker"),
        ("relapse", "window_days"),
        ("relapse", "anchor"),
        ("relapse", "comparator"),
        ("relapse", "boundary"),
        ("observation", "max_gap_hours"),
        ("observation", "endpoint_tolerance_hours"),
        ("observation", "minimum_post_onset_measurements"),
        ("duplicates", "exact"),
        ("duplicates", "conflicting_timestamp"),
        ("admission_censoring", "hadm_id_column"),
        ("admission_censoring", "discharge_time_column"),
        ("admission_censoring", "death_time_column"),
        ("admission_censoring", "endpoint_precedence"),
        ("admission_censoring", "horizon_boundary"),
        ("admission_censoring", "unknown_endpoint_reason"),
        ("recovery_timing_sensitivity", "enabled"),
        ("recovery_timing_sensitivity", "reference_time_column"),
        ("recovery_timing_sensitivity", "baseline_column"),
        ("recovery_timing_sensitivity", "threshold_above_baseline_mg_dl"),
        ("recovery_timing_sensitivity", "value_comparison"),
        ("recovery_timing_sensitivity", "fast_max_hours"),
        ("recovery_timing_sensitivity", "intermediate_max_hours"),
        ("recovery_timing_sensitivity", "time_boundary"),
        ("recovery_timing_sensitivity", "sustained_hours"),
        ("recovery_timing_sensitivity", "max_gap_hours"),
        ("recovery_timing_sensitivity", "endpoint_tolerance_hours"),
        ("recovery_timing_sensitivity", "minimum_measurements"),
        ("recovery_timing_sensitivity", "duplicate_policy"),
    ),
    "representations": (
        ("window_start_column",),
        ("window_end_column",),
        ("window_start_boundary",),
        ("window_end_boundary",),
        ("max_length",),
        ("truncation",),
        ("value_transform",),
        ("time_features",),
        ("summary_features",),
        ("slope_hours_per_unit",),
        ("permutation_seeds",),
    ),
    "matching": (
        ("class_a",),
        ("class_b",),
        ("selection_seed",),
        ("coarsening",),
        ("minimum_pairs_per_split",),
        ("maximum_absolute_smd",),
    ),
    "splitting": (
        ("train_fraction",),
        ("validation_fraction",),
        ("test_fraction",),
        ("stratification",),
        ("split_seed",),
        ("rare_stratum_policy",),
    ),
    "models": (
        ("seeds",),
        ("tabular_preprocessing", "scaling"),
        ("tabular_preprocessing", "missing_values"),
        ("sequence_preprocessing", "scaling"),
        ("sequence_preprocessing", "missing_values"),
        ("sequence_training", "batch_size"),
        ("sequence_training", "max_epochs"),
        ("sequence_training", "learning_rate"),
        ("sequence_training", "weight_decay"),
        ("sequence_training", "optimizer"),
        ("sequence_training", "patience"),
        ("sequence_training", "min_delta"),
        ("sequence_training", "early_stopping_metric"),
        ("sequence_training", "device"),
        ("sequence_training", "num_workers"),
        ("sequence_training", "pin_memory"),
        ("sequence_training", "deterministic_algorithms"),
        ("sequence_training", "class_weight"),
        ("sequence_training", "gradient_clip_norm"),
        ("sequence_training", "optimizer_parameters"),
        ("lstm", "hidden_dim"),
        ("lstm", "num_layers"),
        ("lstm", "dropout"),
        ("lstm", "bidirectional"),
        ("lstm", "projection_dim"),
        ("transformer", "d_model"),
        ("transformer", "nhead"),
        ("transformer", "num_layers"),
        ("transformer", "dim_feedforward"),
        ("transformer", "dropout"),
        ("transformer", "max_sequence_length"),
        ("transformer", "positional_encoding"),
        ("transformer", "pooling"),
        ("logistic_regression", "C"),
        ("logistic_regression", "penalty"),
        ("logistic_regression", "solver"),
        ("logistic_regression", "max_iter"),
        ("logistic_regression", "tol"),
        ("logistic_regression", "fit_intercept"),
        ("logistic_regression", "class_weight"),
        ("logistic_regression", "n_jobs"),
        ("xgboost", "parameters", "n_estimators"),
        ("xgboost", "parameters", "max_depth"),
        ("xgboost", "parameters", "learning_rate"),
        ("xgboost", "parameters", "min_child_weight"),
        ("xgboost", "parameters", "subsample"),
        ("xgboost", "parameters", "colsample_bytree"),
        ("xgboost", "parameters", "reg_alpha"),
        ("xgboost", "parameters", "reg_lambda"),
        ("xgboost", "parameters", "objective"),
        ("xgboost", "parameters", "eval_metric"),
        ("xgboost", "parameters", "n_jobs"),
        ("xgboost", "parameters", "tree_method"),
        ("xgboost", "parameters", "verbosity"),
        ("xgboost", "fit_parameters"),
        ("xgboost", "class_weight"),
    ),
    "metrics": (
        ("binary", "labels"),
        ("binary", "threshold"),
        ("binary", "positive_label"),
        ("binary", "missing_class_policy"),
        ("binary", "zero_division"),
        ("multiclass", "labels"),
        ("multiclass", "decision_rule"),
        ("multiclass", "missing_class_policy"),
        ("multiclass", "zero_division"),
        ("aggregation", "ddof"),
        ("bootstrap", "n_resamples"),
        ("bootstrap", "confidence_level"),
        ("bootstrap", "seed"),
        ("bootstrap", "missing_class_policy"),
        ("bootstrap", "interval_method"),
    ),
}


OPTIONAL_SECTION_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "matching": (("maximum_absolute_smd_overrides",),),
    "models": (("sequence_training", "momentum"),),
}


def _path_tree(paths: Sequence[Sequence[str]]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for path in paths:
        current = tree
        for key in path:
            current = current.setdefault(key, {})
    return tree


def _reject_unknown_keys(value: Any, tree: Mapping[str, Any], context: str) -> None:
    section = _mapping(value, context)
    extra = sorted(set(section).difference(tree))
    if extra:
        raise AkiConfigError(f"{context} contains unknown keys: {extra}")
    for key, child_tree in tree.items():
        if child_tree and key in section and section[key] is not None:
            _reject_unknown_keys(section[key], child_tree, f"{context}.{key}")


def _validate_known_protocol_keys(raw: Mapping[str, Any]) -> None:
    for section, required_paths in REQUIRED_SECTION_PATHS.items():
        paths = (*required_paths, *OPTIONAL_SECTION_PATHS.get(section, ()))
        _reject_unknown_keys(raw[section], _path_tree(paths), section)

    training = _mapping(raw["models"]["sequence_training"], "models.sequence_training")
    if training.get("optimizer") != "sgd" and "momentum" in training:
        raise AkiConfigError(
            "models.sequence_training.momentum is only valid when optimizer is 'sgd'"
        )


def _validate_downstream_sections(raw: Mapping[str, Any]) -> None:
    _validate_known_protocol_keys(raw)
    for section, relative_paths in REQUIRED_SECTION_PATHS.items():
        section_value = _mapping(raw[section], section)
        if not section_value:
            raise AkiConfigError(f"{section} must not be empty")
        for relative_path in relative_paths:
            full_path = (section, *relative_path)
            dotted = ".".join(full_path)
            try:
                require_path(raw, full_path)
            except AkiConfigError:
                if dotted in ALLOWED_NULL_PATHS:
                    # Nullable paths must still exist.
                    current: Any = raw
                    for key in full_path:
                        if not isinstance(current, Mapping) or key not in current:
                            raise
                        current = current[key]
                else:
                    raise

    sequence_training = _mapping(raw["models"]["sequence_training"], "models.sequence_training")
    optimizer = sequence_training["optimizer"]
    if optimizer == "sgd":
        require_path(raw, ("models", "sequence_training", "momentum"))
    for key in (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "objective",
        "eval_metric",
        "n_jobs",
        "tree_method",
        "verbosity",
    ):
        require_path(raw, ("models", "xgboost", "parameters", key))
    binary = _mapping(raw["metrics"]["binary"], "metrics.binary")
    labels = binary["labels"]
    if not isinstance(labels, list) or len(labels) != 2 or len(set(labels)) != 2:
        raise AkiConfigError("metrics.binary.labels must contain exactly two unique labels")
    if binary["positive_label"] not in labels:
        raise AkiConfigError("metrics.binary.positive_label must occur in metrics.binary.labels")
    for path in (("models", "seeds"), ("representations", "permutation_seeds")):
        values = require_path(raw, path)
        dotted = ".".join(path)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
            or len(set(values)) != 3
        ):
            raise AkiConfigError(f"{dotted} must contain exactly three unique integer seeds")

    _validate_downstream_values(raw)


def _validate_class_weight(value: Any, path: str) -> None:
    if value is None or value == "balanced":
        return
    if not isinstance(value, Mapping) or not value:
        raise AkiConfigError(
            f"{path} must be null, 'balanced', or a non-empty label-to-weight mapping"
        )
    for label, weight in value.items():
        _number(weight, f"{path}.{label}", minimum=0)
        if float(weight) <= 0:
            raise AkiConfigError(f"{path}.{label} must be > 0")


def _validate_downstream_values(raw: Mapping[str, Any]) -> None:
    """Validate every non-cohort protocol choice before any MIMIC data access."""

    # Reuse the pure episode protocol parser so the state machine and loader
    # cannot drift on boundary/comparator choices.
    from .episodes import EPISODE_COLUMNS, EpisodeConfigError, _Protocol

    try:
        _Protocol.from_config(raw)
    except EpisodeConfigError as exc:
        raise AkiConfigError(str(exc)) from exc
    from .sensitivity import validate_recovery_timing_sensitivity_config

    try:
        validate_recovery_timing_sensitivity_config(raw)
    except ValueError as exc:
        raise AkiConfigError(str(exc)) from exc

    censor = _mapping(raw["episodes"]["admission_censoring"], "episodes.admission_censoring")
    require_keys(
        censor,
        (
            "hadm_id_column",
            "discharge_time_column",
            "death_time_column",
            "endpoint_precedence",
            "horizon_boundary",
            "unknown_endpoint_reason",
        ),
        "episodes.admission_censoring",
        allow_extra=False,
    )
    for key in (
        "hadm_id_column",
        "discharge_time_column",
        "death_time_column",
        "unknown_endpoint_reason",
    ):
        _string(censor[key], f"episodes.admission_censoring.{key}")
    _choice(
        censor["endpoint_precedence"],
        "episodes.admission_censoring.endpoint_precedence",
        {"death_then_discharge", "discharge_then_death"},
    )
    _choice(
        censor["horizon_boundary"],
        "episodes.admission_censoring.horizon_boundary",
        {"inclusive", "exclusive"},
    )

    representations = _mapping(raw["representations"], "representations")
    for key in ("window_start_column", "window_end_column"):
        _string(representations[key], f"representations.{key}")
    allowed_time_columns = {column for column in EPISODE_COLUMNS if column.endswith("_time")}
    guaranteed_representation_bounds = {
        "baseline_start_time",
        "baseline_end_time",
        "onset_time",
        "peak_time",
        "follow_up_end_time",
        "required_observation_end_time",
        "last_observation_time",
    }
    for key in ("window_start_column", "window_end_column"):
        if representations[key] not in guaranteed_representation_bounds:
            raise AkiConfigError(
                f"representations.{key} must name a timestamp guaranteed on every "
                "included episode; "
                f"got {representations[key]!r}"
            )
    for key in ("window_start_boundary", "window_end_boundary"):
        _choice(
            representations[key],
            f"representations.{key}",
            {"inclusive", "exclusive"},
        )
    sensitivity = _mapping(
        raw["episodes"]["recovery_timing_sensitivity"],
        "episodes.recovery_timing_sensitivity",
    )
    if sensitivity["reference_time_column"] not in allowed_time_columns:
        raise AkiConfigError(
            "episodes.recovery_timing_sensitivity.reference_time_column must name "
            "a persisted episode timestamp column"
        )
    allowed_value_columns = {
        "baseline_value",
        "onset_value",
        "onset_rise",
        "peak_value",
        "recovery_value",
        "recurrence_value",
    }
    if sensitivity["baseline_column"] not in allowed_value_columns:
        raise AkiConfigError(
            "episodes.recovery_timing_sensitivity.baseline_column must name a "
            "persisted episode value column"
        )
    _integer(representations["max_length"], "representations.max_length", minimum=1)
    _choice(
        representations["truncation"],
        "representations.truncation",
        {"head", "tail", "error"},
    )
    _choice(
        representations["value_transform"],
        "representations.value_transform",
        {"raw", "delta_from_baseline", "ratio_to_baseline", "log"},
    )
    time_features = _string_list(
        representations["time_features"],
        "representations.time_features",
        allow_empty=True,
    )
    unsupported_time = set(time_features).difference(
        {"elapsed_hours", "delta_hours", "time_since_onset_hours"}
    )
    if unsupported_time:
        raise AkiConfigError(
            f"representations.time_features contains unsupported values: {sorted(unsupported_time)}"
        )
    summaries = _string_list(
        representations["summary_features"], "representations.summary_features"
    )
    supported_summaries = {
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
        "q05",
        "q10",
        "q25",
        "q75",
        "q90",
        "q95",
    }
    unsupported_summaries = set(summaries).difference(supported_summaries)
    if unsupported_summaries:
        raise AkiConfigError(
            "representations.summary_features contains unsupported values: "
            f"{sorted(unsupported_summaries)}"
        )
    required_invariant_controls = {
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
    }
    missing_invariant_controls = required_invariant_controls.difference(summaries)
    if missing_invariant_controls:
        raise AkiConfigError(
            "representations.summary_features must include the locked strong "
            "order-invariant controls: "
            f"{sorted(missing_invariant_controls)}"
        )
    slope_scale = _number(
        representations["slope_hours_per_unit"],
        "representations.slope_hours_per_unit",
        minimum=0,
    )
    if slope_scale <= 0:
        raise AkiConfigError("representations.slope_hours_per_unit must be > 0")

    splitting = _mapping(raw["splitting"], "splitting")
    fractions = [
        _number(splitting[key], f"splitting.{key}", minimum=0)
        for key in ("train_fraction", "validation_fraction", "test_fraction")
    ]
    if any(value <= 0 for value in fractions) or not math.isclose(sum(fractions), 1.0):
        raise AkiConfigError("split fractions must be positive and sum to one")
    _choice(
        splitting["stratification"],
        "splitting.stratification",
        {"none", "single_label", "phenotype_signature"},
    )
    _integer(splitting["split_seed"], "splitting.split_seed")
    _choice(
        splitting["rare_stratum_policy"],
        "splitting.rare_stratum_policy",
        {"error", "unstratified"},
    )

    matching = _mapping(raw["matching"], "matching")
    primary = {str(matching["class_a"]), str(matching["class_b"])}
    if primary != {"transient", "persistent"}:
        raise AkiConfigError(
            "matching.class_a/class_b must be transient and persistent for the locked primary task"
        )
    _integer(matching["selection_seed"], "matching.selection_seed")
    coarsening = _mapping(matching["coarsening"], "matching.coarsening")
    if not coarsening:
        raise AkiConfigError("matching.coarsening must not be empty")
    required_audit_features = {
        "baseline_creatinine_mg_dl",
        "peak_value",
        "duration_hours",
        "count",
    }
    missing_audit_features = required_audit_features.difference(coarsening)
    if missing_audit_features:
        raise AkiConfigError(
            "matching.coarsening must include the locked audit-population features: "
            f"{sorted(missing_audit_features)}"
        )
    allowed_matching_features = set(summaries) | {
        "baseline_creatinine_mg_dl",
        "peak_value",
    }
    disallowed_matching_features = set(coarsening).difference(allowed_matching_features)
    if disallowed_matching_features:
        raise AkiConfigError(
            "matching.coarsening may contain only configured order-invariant summaries, "
            "baseline_creatinine_mg_dl, and peak_value; disallowed: "
            f"{sorted(disallowed_matching_features)}"
        )
    for feature, raw_spec in coarsening.items():
        spec = _mapping(raw_spec, f"matching.coarsening.{feature}")
        method = _choice(
            spec.get("method"),
            f"matching.coarsening.{feature}.method",
            {"exact", "edges", "width"},
        )
        if method == "exact":
            require_keys(spec, ("method",), f"matching.coarsening.{feature}", allow_extra=False)
        elif method == "edges":
            require_keys(
                spec,
                ("method", "edges", "right"),
                f"matching.coarsening.{feature}",
                allow_extra=False,
            )
            edges = spec["edges"]
            if not isinstance(edges, list) or len(edges) < 2:
                raise AkiConfigError(f"matching.coarsening.{feature}.edges needs >=2 values")
            numeric_edges = [
                _number(value, f"matching.coarsening.{feature}.edges[{index}]")
                for index, value in enumerate(edges)
            ]
            if any(right <= left for left, right in zip(numeric_edges, numeric_edges[1:])):
                raise AkiConfigError(f"matching.coarsening.{feature}.edges must increase")
            _boolean(spec["right"], f"matching.coarsening.{feature}.right")
        else:
            require_keys(
                spec,
                ("method", "width", "origin"),
                f"matching.coarsening.{feature}",
                allow_extra=False,
            )
            width = _number(spec["width"], f"matching.coarsening.{feature}.width", minimum=0)
            if width <= 0:
                raise AkiConfigError(f"matching.coarsening.{feature}.width must be > 0")
            _number(spec["origin"], f"matching.coarsening.{feature}.origin")
    minimum_pairs = _mapping(
        matching["minimum_pairs_per_split"], "matching.minimum_pairs_per_split"
    )
    require_keys(
        minimum_pairs,
        ("train", "validation", "test"),
        "matching.minimum_pairs_per_split",
        allow_extra=False,
    )
    for split, value in minimum_pairs.items():
        _integer(value, f"matching.minimum_pairs_per_split.{split}", minimum=1)
    _number(matching["maximum_absolute_smd"], "matching.maximum_absolute_smd", minimum=0)
    maximum_absolute_smd_overrides = _mapping(
        matching.get("maximum_absolute_smd_overrides", {}),
        "matching.maximum_absolute_smd_overrides",
    )
    unknown_smd_override_features = set(maximum_absolute_smd_overrides).difference(coarsening)
    if unknown_smd_override_features:
        raise AkiConfigError(
            "matching.maximum_absolute_smd_overrides keys must be configured "
            "matching.coarsening features; unknown: "
            f"{sorted(unknown_smd_override_features)}"
        )
    for feature, threshold in maximum_absolute_smd_overrides.items():
        _number(
            threshold,
            f"matching.maximum_absolute_smd_overrides.{feature}",
            minimum=0,
        )

    models = _mapping(raw["models"], "models")
    for name in ("tabular_preprocessing", "sequence_preprocessing"):
        preprocessing = _mapping(models[name], f"models.{name}")
        _choice(preprocessing["scaling"], f"models.{name}.scaling", {"none", "standard"})
        _choice(
            preprocessing["missing_values"], f"models.{name}.missing_values", {"error", "median"}
        )
    training = _mapping(models["sequence_training"], "models.sequence_training")
    _integer(training["batch_size"], "models.sequence_training.batch_size", minimum=1)
    _integer(training["max_epochs"], "models.sequence_training.max_epochs", minimum=1)
    learning_rate = _number(
        training["learning_rate"], "models.sequence_training.learning_rate", minimum=0
    )
    if learning_rate <= 0:
        raise AkiConfigError("models.sequence_training.learning_rate must be > 0")
    _number(training["weight_decay"], "models.sequence_training.weight_decay", minimum=0)
    optimizer = _choice(
        training["optimizer"], "models.sequence_training.optimizer", {"adam", "adamw", "sgd"}
    )
    _integer(training["patience"], "models.sequence_training.patience", minimum=0)
    _number(training["min_delta"], "models.sequence_training.min_delta", minimum=0)
    _choice(
        training["early_stopping_metric"],
        "models.sequence_training.early_stopping_metric",
        {"validation_loss"},
    )
    _string(training["device"], "models.sequence_training.device")
    _integer(training["num_workers"], "models.sequence_training.num_workers", minimum=0)
    _boolean(training["pin_memory"], "models.sequence_training.pin_memory")
    _boolean(
        training["deterministic_algorithms"],
        "models.sequence_training.deterministic_algorithms",
    )
    _validate_class_weight(training["class_weight"], "models.sequence_training.class_weight")
    optimizer_parameters = _mapping(
        training["optimizer_parameters"],
        "models.sequence_training.optimizer_parameters",
    )
    duplicated_optimizer_keys = sorted(
        set(optimizer_parameters).intersection({"lr", "weight_decay", "momentum"})
    )
    if duplicated_optimizer_keys:
        raise AkiConfigError(
            "models.sequence_training.optimizer_parameters duplicates controlled keys: "
            f"{duplicated_optimizer_keys}"
        )
    clip = training["gradient_clip_norm"]
    if (
        clip is not None
        and _number(clip, "models.sequence_training.gradient_clip_norm", minimum=0) <= 0
    ):
        raise AkiConfigError("models.sequence_training.gradient_clip_norm must be null or > 0")
    if optimizer == "sgd":
        _number(training["momentum"], "models.sequence_training.momentum", minimum=0)

    lstm = _mapping(models["lstm"], "models.lstm")
    for key in ("hidden_dim", "num_layers"):
        _integer(lstm[key], f"models.lstm.{key}", minimum=1)
    dropout = _number(lstm["dropout"], "models.lstm.dropout", minimum=0)
    if dropout >= 1:
        raise AkiConfigError("models.lstm.dropout must be < 1")
    _boolean(lstm["bidirectional"], "models.lstm.bidirectional")
    projection = lstm["projection_dim"]
    if projection is not None:
        _integer(projection, "models.lstm.projection_dim", minimum=1)

    transformer = _mapping(models["transformer"], "models.transformer")
    for key in (
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "max_sequence_length",
    ):
        _integer(transformer[key], f"models.transformer.{key}", minimum=1)
    if int(transformer["d_model"]) % int(transformer["nhead"]) != 0:
        raise AkiConfigError("models.transformer.d_model must be divisible by nhead")
    if int(transformer["max_sequence_length"]) < int(representations["max_length"]):
        raise AkiConfigError(
            "models.transformer.max_sequence_length must cover representations.max_length"
        )
    transformer_dropout = _number(transformer["dropout"], "models.transformer.dropout", minimum=0)
    if transformer_dropout >= 1:
        raise AkiConfigError("models.transformer.dropout must be < 1")
    _choice(
        transformer["positional_encoding"],
        "models.transformer.positional_encoding",
        {"learned", "sinusoidal", "none"},
    )
    _choice(transformer["pooling"], "models.transformer.pooling", {"mean", "first", "last"})

    logistic = _mapping(models["logistic_regression"], "models.logistic_regression")
    for key in ("C", "tol"):
        if _number(logistic[key], f"models.logistic_regression.{key}", minimum=0) <= 0:
            raise AkiConfigError(f"models.logistic_regression.{key} must be > 0")
    _integer(logistic["max_iter"], "models.logistic_regression.max_iter", minimum=1)
    if logistic["penalty"] is not None:
        _string(logistic["penalty"], "models.logistic_regression.penalty")
    _string(logistic["solver"], "models.logistic_regression.solver")
    _boolean(logistic["fit_intercept"], "models.logistic_regression.fit_intercept")
    _validate_class_weight(logistic["class_weight"], "models.logistic_regression.class_weight")
    if logistic["n_jobs"] is not None:
        _integer(logistic["n_jobs"], "models.logistic_regression.n_jobs")

    xgboost = _mapping(models["xgboost"], "models.xgboost")
    xgb_parameters = _mapping(xgboost["parameters"], "models.xgboost.parameters")
    _integer(
        xgb_parameters["n_estimators"],
        "models.xgboost.parameters.n_estimators",
        minimum=1,
    )
    for key in ("max_depth", "verbosity"):
        _integer(xgb_parameters[key], f"models.xgboost.parameters.{key}", minimum=0)
    if _integer(xgb_parameters["n_jobs"], "models.xgboost.parameters.n_jobs") == 0:
        raise AkiConfigError("models.xgboost.parameters.n_jobs cannot be zero")
    for key in ("learning_rate", "min_child_weight", "reg_alpha", "reg_lambda"):
        value = _number(xgb_parameters[key], f"models.xgboost.parameters.{key}", minimum=0)
        if key == "learning_rate" and value <= 0:
            raise AkiConfigError("models.xgboost.parameters.learning_rate must be > 0")
    for key in ("subsample", "colsample_bytree"):
        value = _number(xgb_parameters[key], f"models.xgboost.parameters.{key}", minimum=0)
        if value <= 0 or value > 1:
            raise AkiConfigError(f"models.xgboost.parameters.{key} must be in (0, 1]")
    objective = _string(xgb_parameters["objective"], "models.xgboost.parameters.objective")
    if objective != "multi:softprob":
        raise AkiConfigError(
            "models.xgboost.parameters.objective must be 'multi:softprob' to produce "
            "probabilities for both locked tasks"
        )
    if "num_class" in xgb_parameters:
        raise AkiConfigError(
            "models.xgboost.parameters.num_class must be omitted; "
            "task class count is resolved per run"
        )
    for key in ("eval_metric", "tree_method"):
        _string(xgb_parameters[key], f"models.xgboost.parameters.{key}")
    _mapping(xgboost["fit_parameters"], "models.xgboost.fit_parameters")
    _validate_class_weight(xgboost["class_weight"], "models.xgboost.class_weight")

    metrics = _mapping(raw["metrics"], "metrics")
    binary = _mapping(metrics["binary"], "metrics.binary")
    if {str(value) for value in binary["labels"]} != {"transient", "persistent"}:
        raise AkiConfigError("metrics.binary.labels must contain transient and persistent")
    threshold = _number(binary["threshold"], "metrics.binary.threshold", minimum=0)
    if threshold > 1:
        raise AkiConfigError("metrics.binary.threshold must be in [0, 1]")
    _choice(
        binary["missing_class_policy"],
        "metrics.binary.missing_class_policy",
        {"raise", "nan", "skip"},
    )
    if binary["zero_division"] not in (0, 1, "warn"):
        raise AkiConfigError("metrics.binary.zero_division must be 0, 1, or 'warn'")
    multiclass = _mapping(metrics["multiclass"], "metrics.multiclass")
    multi_labels = multiclass["labels"]
    if (
        not isinstance(multi_labels, list)
        or len(multi_labels) != 3
        or {str(value) for value in multi_labels} != {"transient", "persistent", "relapsing"}
    ):
        raise AkiConfigError(
            "metrics.multiclass.labels must contain transient, persistent, and relapsing"
        )
    label_universe = set(multi_labels)
    for path, specification in (
        ("models.sequence_training.class_weight", training["class_weight"]),
        ("models.logistic_regression.class_weight", logistic["class_weight"]),
        ("models.xgboost.class_weight", xgboost["class_weight"]),
    ):
        if isinstance(specification, Mapping) and set(specification) != label_universe:
            missing = sorted(label_universe.difference(specification))
            extra = sorted(set(specification).difference(label_universe))
            raise AkiConfigError(
                f"{path} must cover the full binary/multiclass label universe; "
                f"missing={missing}, extra={extra}"
            )
    _choice(multiclass["decision_rule"], "metrics.multiclass.decision_rule", {"argmax"})
    _choice(
        multiclass["missing_class_policy"],
        "metrics.multiclass.missing_class_policy",
        {"raise", "nan", "skip"},
    )
    if multiclass["zero_division"] not in (0, 1, "warn"):
        raise AkiConfigError("metrics.multiclass.zero_division must be 0, 1, or 'warn'")
    _integer(metrics["aggregation"]["ddof"], "metrics.aggregation.ddof", minimum=0)
    bootstrap = _mapping(metrics["bootstrap"], "metrics.bootstrap")
    _integer(bootstrap["n_resamples"], "metrics.bootstrap.n_resamples", minimum=1)
    confidence = _number(
        bootstrap["confidence_level"], "metrics.bootstrap.confidence_level", minimum=0
    )
    if not 0 < confidence < 1:
        raise AkiConfigError("metrics.bootstrap.confidence_level must be in (0, 1)")
    _integer(bootstrap["seed"], "metrics.bootstrap.seed")
    _choice(
        bootstrap["missing_class_policy"],
        "metrics.bootstrap.missing_class_policy",
        {"raise", "skip"},
    )
    _choice(bootstrap["interval_method"], "metrics.bootstrap.interval_method", {"percentile"})


def _canonical_hash_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        return {
            "__mapping__": [
                [type(key).__name__, repr(key), _canonical_hash_tree(child)] for key, child in items
            ]
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_hash_tree(child) for child in value]
    return value


def _deep_freeze(value: Any) -> Any:
    """Recursively make a validated protocol immutable."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(child) for child in value)
    return value


def _deep_thaw(value: Any) -> Any:
    """Return a detached JSON/YAML-compatible copy of a frozen protocol."""

    if isinstance(value, Mapping):
        return {key: _deep_thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(child) for child in value]
    return deepcopy(value)


@dataclass(frozen=True)
class AkiAuditConfig:
    """Validated immutable top-level view of one AKI audit protocol."""

    schema_version: int
    cohort: CohortConfig
    cleaning: CleaningConfig
    episodes: Mapping[str, Any]
    representations: Mapping[str, Any]
    matching: Mapping[str, Any]
    splitting: Mapping[str, Any]
    models: Mapping[str, Any]
    metrics: Mapping[str, Any]
    config_hash: str
    source_path: Path | None
    _raw: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a detached serializable copy of the resolved protocol."""

        return _deep_thaw(self._raw)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise AkiConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def aki_config_from_mapping(
    value: Mapping[str, Any], *, source_path: Path | None = None
) -> AkiAuditConfig:
    """Validate and materialize an AKI protocol mapping."""

    raw = deepcopy(dict(_mapping(value, "configuration")))
    top_keys = (
        "schema_version",
        "cohort",
        "cleaning",
        "episodes",
        "representations",
        "matching",
        "splitting",
        "models",
        "metrics",
    )
    require_keys(raw, top_keys, "configuration", allow_extra=False)
    validate_no_null_scientific_values(raw)
    schema_version = _integer(raw["schema_version"], "schema_version", minimum=1)
    if schema_version != SCHEMA_VERSION:
        raise AkiConfigError(
            f"unsupported schema_version {schema_version}; expected {SCHEMA_VERSION}"
        )
    cohort = CohortConfig.from_mapping(_mapping(raw["cohort"], "cohort"))
    cleaning = CleaningConfig.from_mapping(_mapping(raw["cleaning"], "cleaning"))
    _validate_downstream_sections(raw)

    serialized = json.dumps(
        _canonical_hash_tree(raw),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    config_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return AkiAuditConfig(
        schema_version=schema_version,
        cohort=cohort,
        cleaning=cleaning,
        episodes=_deep_freeze(raw["episodes"]),
        representations=_deep_freeze(raw["representations"]),
        matching=_deep_freeze(raw["matching"]),
        splitting=_deep_freeze(raw["splitting"]),
        models=_deep_freeze(raw["models"]),
        metrics=_deep_freeze(raw["metrics"]),
        config_hash=config_hash,
        source_path=source_path,
        _raw=_deep_freeze(raw),
    )


def load_aki_config(path: str | Path) -> AkiAuditConfig:
    """Load a YAML protocol and fail before data access if any choice is absent."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"AKI configuration not found: {resolved}")
    try:
        loaded = yaml.load(resolved.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise AkiConfigError(f"invalid YAML in {resolved}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise AkiConfigError(f"AKI configuration must be a mapping: {resolved}")
    return aki_config_from_mapping(loaded, source_path=resolved)


__all__ = [
    "AdmissionConfig",
    "AdultConfig",
    "AkiAuditConfig",
    "AkiConfigError",
    "ALLOWED_NULL_PATHS",
    "CREATININE_IDENTITY",
    "CREATININE_ITEMID",
    "CleaningConfig",
    "CohortConfig",
    "DeduplicationConfig",
    "REQUIRED_SECTION_PATHS",
    "SCHEMA_VERSION",
    "UnitConversion",
    "aki_config_from_mapping",
    "load_aki_config",
    "require_keys",
    "require_path",
    "validate_no_null_scientific_values",
]
