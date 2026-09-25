"""Chunked MIMIC-IV serum-creatinine cohort extraction with audit artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator
import uuid

import numpy as np
import pandas as pd

from .config import (
    AkiAuditConfig,
    CREATININE_IDENTITY,
    CohortConfig,
    CleaningConfig,
    load_aki_config,
)


LABEVENT_COLUMNS = [
    "labevent_id",
    "subject_id",
    "hadm_id",
    "specimen_id",
    "itemid",
    "charttime",
    "storetime",
    "value",
    "valuenum",
    "valueuom",
]

MEASUREMENT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "specimen_time",
    "creatinine_mg_dl",
    "creatinine_unit",
    "specimen_id",
    "labevent_id",
    "admittime",
    "dischtime",
    "deathtime",
    "administrative_end_time",
    "administrative_end_type",
    "admission_type",
    "age_at_admission",
    "storetime",
    "source_hadm_id",
    "raw_value",
    "raw_valuenum",
    "raw_unit",
    "config_hash",
]


class CohortExtractionError(RuntimeError):
    """Raised when source data violate an extraction invariant."""


@dataclass(frozen=True)
class CohortExtractionResult:
    """Paths and headline counts emitted by :func:`extract_creatinine_cohort`."""

    measurements_path: Path
    rejected_measurements_path: Path
    admission_audit_path: Path
    flow_audit_path: Path
    unit_audit_path: Path
    config_hash: str
    labevents_rows_scanned: int
    creatinine_rows_found: int
    retained_measurements: int
    retained_admissions: int
    retained_subjects: int


def _resolve_table(raw_dir: Path, stem: str) -> Path:
    for name in (f"{stem}.csv", f"{stem}.csv.gz"):
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"MIMIC-IV table {stem!r} not found under {raw_dir} as CSV or CSV.GZ")


def validate_creatinine_labitem(path: str | Path, itemid: int) -> dict[str, Any]:
    """Validate that the locked item ID has the expected MIMIC dictionary identity."""

    frame = pd.read_csv(
        path,
        usecols=["itemid", "label", "fluid", "category"],
        dtype={"itemid": "Int64", "label": "string", "fluid": "string", "category": "string"},
    )
    selected = frame.loc[frame["itemid"] == itemid]
    if len(selected) != 1:
        raise CohortExtractionError(
            f"d_labitems must contain itemid {itemid} exactly once; found {len(selected)} rows"
        )
    row = selected.iloc[0]
    mismatches = {}
    for column, expected in CREATININE_IDENTITY.items():
        observed = row[column]
        if pd.isna(observed) or str(observed).strip().casefold() != expected.casefold():
            mismatches[column] = {
                "expected": expected,
                "observed": None if pd.isna(observed) else str(observed),
            }
    if mismatches:
        raise CohortExtractionError(
            f"itemid {itemid} does not match the locked serum-creatinine identity: {mismatches}"
        )
    return {
        "itemid": int(row["itemid"]),
        "label": str(row["label"]),
        "fluid": str(row["fluid"]),
        "category": str(row["category"]),
    }


def _append_reason(series: pd.Series, mask: pd.Series | np.ndarray, reason: str) -> pd.Series:
    result = series.copy()
    selected = pd.Series(mask, index=series.index).fillna(False).astype(bool)
    if not selected.any():
        return result
    current = result.loc[selected].fillna("").astype(str)
    result.loc[selected] = np.where(current.eq(""), reason, current + ";" + reason)
    return result


def _closed_interval_mask(values: Any, starts: Any, ends: Any, boundary: str) -> Any:
    left = values >= starts if boundary in {"both", "left"} else values > starts
    right = values <= ends if boundary in {"both", "right"} else values < ends
    return left & right


def _prepare_admission_audit(
    admissions_path: Path,
    patients_path: Path,
    cohort: CohortConfig,
    config_hash: str,
) -> pd.DataFrame:
    patients = pd.read_csv(
        patients_path,
        usecols=["subject_id", "anchor_age", "anchor_year"],
        dtype={"subject_id": "Int64", "anchor_age": "Float64", "anchor_year": "Int64"},
    )
    if patients["subject_id"].isna().any() or patients["subject_id"].duplicated().any():
        raise CohortExtractionError("patients.csv must contain one non-null row per subject_id")
    admissions = pd.read_csv(
        admissions_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "deathtime",
            "admission_type",
        ],
        dtype={
            "subject_id": "Int64",
            "hadm_id": "Int64",
            "admission_type": "string",
        },
    )
    if admissions["hadm_id"].isna().any() or admissions["hadm_id"].duplicated().any():
        raise CohortExtractionError("admissions.csv must contain one non-null row per hadm_id")
    raw_death = admissions["deathtime"].copy()
    raw_death_present = raw_death.notna() & raw_death.astype("string").str.strip().ne("")
    for column in ("admittime", "dischtime", "deathtime"):
        admissions[column] = pd.to_datetime(admissions[column], errors="coerce")
    admissions["death_time_invalid"] = raw_death_present & admissions["deathtime"].isna()
    admissions["death_before_admission"] = (
        admissions["deathtime"].notna()
        & admissions["admittime"].notna()
        & (admissions["deathtime"] < admissions["admittime"])
    )
    if cohort.admission.administrative_end_policy == "discharge_only":
        admissions["administrative_end_time"] = admissions["dischtime"]
        admissions["administrative_end_type"] = "discharge"
    else:
        death_precedes_discharge = admissions["deathtime"].notna() & (
            admissions["dischtime"].isna() | (admissions["deathtime"] < admissions["dischtime"])
        )
        admissions["administrative_end_time"] = admissions["dischtime"].where(
            ~death_precedes_discharge, admissions["deathtime"]
        )
        admissions["administrative_end_type"] = np.where(
            death_precedes_discharge, "death", "discharge"
        )
    admissions = admissions.merge(patients, on="subject_id", how="left", validate="many_to_one")
    admissions["age_at_admission"] = (
        admissions["anchor_age"] + admissions["admittime"].dt.year - admissions["anchor_year"]
    ).astype("Float64")
    admissions["stay_hours"] = (
        (admissions["administrative_end_time"] - admissions["admittime"]).dt.total_seconds()
        / 3600.0
    ).astype("Float64")
    admissions["base_exclusion_reasons"] = ""
    reasons = admissions["base_exclusion_reasons"]
    reasons = _append_reason(reasons, admissions["subject_id"].isna(), "missing_subject_id")
    reasons = _append_reason(
        reasons,
        admissions[["anchor_age", "anchor_year"]].isna().any(axis=1),
        "missing_age_anchor",
    )
    reasons = _append_reason(reasons, admissions["admittime"].isna(), "invalid_admittime")
    reasons = _append_reason(reasons, admissions["dischtime"].isna(), "invalid_dischtime")
    if cohort.admission.invalid_death_time_policy == "exclude_admission":
        reasons = _append_reason(reasons, admissions["death_time_invalid"], "invalid_deathtime")
    reasons = _append_reason(
        reasons, admissions["death_before_admission"], "death_before_admission"
    )
    reasons = _append_reason(
        reasons,
        admissions["stay_hours"].notna() & (admissions["stay_hours"] < 0),
        "negative_length_of_stay",
    )
    if cohort.adult.minimum_age_inclusive:
        underage = admissions["age_at_admission"] < cohort.adult.minimum_age_years
    else:
        underage = admissions["age_at_admission"] <= cohort.adult.minimum_age_years
    reasons = _append_reason(reasons, underage, "below_adult_age_threshold")

    eligible_types = cohort.admission.eligible_types
    if eligible_types != "all":
        if cohort.admission.type_matching == "casefold":
            allowed = {value.casefold() for value in eligible_types}
            observed = admissions["admission_type"].str.strip().str.casefold()
        else:
            allowed = set(eligible_types)
            observed = admissions["admission_type"]
        reasons = _append_reason(
            reasons,
            admissions["admission_type"].isna() | ~observed.isin(allowed),
            "ineligible_admission_type",
        )

    if cohort.admission.minimum_stay_inclusive:
        short_stay = admissions["stay_hours"] < cohort.admission.minimum_stay_hours
    else:
        short_stay = admissions["stay_hours"] <= cohort.admission.minimum_stay_hours
    reasons = _append_reason(
        reasons, admissions["stay_hours"].notna() & short_stay, "stay_too_short"
    )
    admissions["base_exclusion_reasons"] = reasons
    admissions["base_eligible"] = reasons.eq("")
    admissions["config_hash"] = config_hash
    return admissions


def _unit_key(value: Any, matching: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if matching in {"strip", "casefold_strip"}:
        text = text.strip()
    if matching == "casefold_strip":
        text = text.casefold()
    return text


def _unit_conversion_lookup(cleaning: CleaningConfig) -> dict[str, tuple[float, float]]:
    lookup: dict[str, tuple[float, float]] = {}
    for rule in cleaning.unit_conversions:
        for alias in rule.aliases:
            key = _unit_key(alias, cleaning.unit_matching)
            assert key is not None
            conversion = (rule.multiplier, rule.offset)
            if key in lookup and lookup[key] != conversion:
                raise CohortExtractionError(f"unit alias {alias!r} maps to conflicting conversions")
            lookup[key] = conversion
    canonical_key = _unit_key(cleaning.canonical_unit, cleaning.unit_matching)
    if canonical_key not in lookup:
        raise CohortExtractionError(
            "unit_conversions must explicitly include the canonical 'mg/dL' unit"
        )
    if lookup[canonical_key] != (1.0, 0.0):
        raise CohortExtractionError("the canonical 'mg/dL' unit must use multiplier=1 and offset=0")
    return lookup


def stream_creatinine_candidates(
    path: str | Path,
    *,
    itemid: int,
    cleaning: CleaningConfig,
    chunksize: int,
    scan_counts: dict[str, int] | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream item 50912 rows and apply value/unit cleaning within each chunk."""

    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    conversion_lookup = _unit_conversion_lookup(cleaning)
    dtype = {
        "labevent_id": "Int64",
        "subject_id": "Int64",
        "hadm_id": "Int64",
        "specimen_id": "Int64",
        "itemid": "Int32",
        "value": "string",
        "valuenum": "Float64",
        "valueuom": "string",
    }
    reader = pd.read_csv(
        path,
        usecols=LABEVENT_COLUMNS,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        if scan_counts is not None:
            scan_counts["labevents_rows_scanned"] += len(chunk)
        frame = chunk.loc[chunk["itemid"] == itemid].copy()
        if frame.empty:
            continue
        if scan_counts is not None:
            scan_counts["creatinine_rows_found"] += len(frame)
        frame = frame.rename(
            columns={
                "value": "raw_value",
                "valuenum": "raw_valuenum",
                "valueuom": "raw_unit",
                "hadm_id": "source_hadm_id",
            }
        )
        frame["hadm_id"] = frame["source_hadm_id"].astype("Int64")
        frame["specimen_time"] = pd.to_datetime(frame.pop("charttime"), errors="coerce")
        frame["storetime"] = pd.to_datetime(frame["storetime"], errors="coerce")
        frame["rejection_reasons"] = ""
        reasons = frame["rejection_reasons"]
        reasons = _append_reason(reasons, frame["subject_id"].isna(), "missing_subject_id")
        reasons = _append_reason(reasons, frame["specimen_time"].isna(), "invalid_charttime")

        numeric = pd.to_numeric(frame["raw_valuenum"], errors="coerce")
        if cleaning.numeric_value_source == "valuenum_then_value":
            numeric = numeric.fillna(pd.to_numeric(frame["raw_value"], errors="coerce"))
        numeric = numeric.astype(float)
        reasons = _append_reason(reasons, numeric.isna(), "missing_or_nonnumeric_value")
        reasons = _append_reason(
            reasons, numeric.notna() & ~np.isfinite(numeric), "nonfinite_value"
        )

        raw_unit_keys = frame["raw_unit"].map(
            lambda value: _unit_key(value, cleaning.unit_matching)
        )
        missing_unit = raw_unit_keys.isna()
        if cleaning.missing_unit_policy == "reject":
            reasons = _append_reason(reasons, missing_unit, "missing_unit")
        conversion_keys = raw_unit_keys.copy()
        if cleaning.missing_unit_policy == "assume_canonical":
            conversion_keys.loc[missing_unit] = _unit_key(
                cleaning.canonical_unit, cleaning.unit_matching
            )
        unknown_unit = conversion_keys.notna() & ~conversion_keys.isin(conversion_lookup)
        reasons = _append_reason(reasons, unknown_unit, "unrecognized_unit")
        multipliers = conversion_keys.map(
            {key: value[0] for key, value in conversion_lookup.items()}
        ).astype(float)
        offsets = conversion_keys.map(
            {key: value[1] for key, value in conversion_lookup.items()}
        ).astype(float)
        frame["creatinine_mg_dl"] = numeric * multipliers + offsets
        frame["creatinine_unit"] = cleaning.canonical_unit
        converted = frame["creatinine_mg_dl"]
        reasons = _append_reason(
            reasons,
            converted.notna() & ~np.isfinite(converted),
            "nonfinite_converted_value",
        )
        if cleaning.value_boundary in {"both", "left"}:
            below = converted < cleaning.minimum_value_mg_dl
        else:
            below = converted <= cleaning.minimum_value_mg_dl
        if cleaning.value_boundary in {"both", "right"}:
            above = converted > cleaning.maximum_value_mg_dl
        else:
            above = converted >= cleaning.maximum_value_mg_dl
        reasons = _append_reason(reasons, converted.notna() & below, "value_below_valid_range")
        reasons = _append_reason(reasons, converted.notna() & above, "value_above_valid_range")
        frame["rejection_reasons"] = reasons
        yield frame


def _infer_null_hadm_ids(
    frame: pd.DataFrame,
    admission_audit: pd.DataFrame,
    cohort: CohortConfig,
) -> None:
    null_mask = frame["source_hadm_id"].isna()
    if not null_mask.any():
        return
    if cohort.admission.null_hadm_policy == "reject":
        frame["rejection_reasons"] = _append_reason(
            frame["rejection_reasons"], null_mask, "missing_hadm_id"
        )
        return

    probe = frame.loc[
        null_mask & frame["subject_id"].notna() & frame["specimen_time"].notna(),
        ["subject_id", "specimen_time"],
    ].copy()
    probe["__candidate_index"] = probe.index
    eligible = admission_audit.loc[
        admission_audit["base_eligible"],
        ["subject_id", "hadm_id", "admittime", "administrative_end_time"],
    ]
    if probe.empty or eligible.empty:
        frame["rejection_reasons"] = _append_reason(
            frame["rejection_reasons"],
            null_mask,
            "null_hadm_no_eligible_admission",
        )
        return
    matches = probe.merge(eligible, on="subject_id", how="left")
    in_window = _closed_interval_mask(
        matches["specimen_time"],
        matches["admittime"],
        matches["administrative_end_time"],
        cohort.admission.charttime_boundary,
    ).fillna(False)
    matches = matches.loc[in_window]
    grouped = matches.groupby("__candidate_index", sort=False)["hadm_id"].agg(list)
    no_match = null_mask & ~frame.index.isin(grouped.index)
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"], no_match, "null_hadm_no_eligible_admission"
    )
    unique = grouped[grouped.map(len) == 1]
    if not unique.empty:
        resolved = unique.map(lambda values: values[0]).astype("Int64")
        frame.loc[resolved.index, "hadm_id"] = resolved
    ambiguous_indices = grouped.index[grouped.map(len) > 1]
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"],
        frame.index.isin(ambiguous_indices),
        "null_hadm_ambiguous_admission",
    )


def _link_admissions(
    frame: pd.DataFrame,
    admission_audit: pd.DataFrame,
    cohort: CohortConfig,
) -> pd.DataFrame:
    frame = frame.copy()
    lookup = admission_audit.set_index("hadm_id", drop=False)
    direct = frame["source_hadm_id"].notna()
    mapped_subject = frame["hadm_id"].map(lookup["subject_id"])
    known = mapped_subject.notna()
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"], direct & ~known, "hadm_id_not_found"
    )
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"],
        direct & known & mapped_subject.ne(frame["subject_id"]),
        "hadm_subject_mismatch",
    )
    mapped_eligible = frame["hadm_id"].map(lookup["base_eligible"])
    mapped_eligible = mapped_eligible.eq(True).fillna(False)
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"], direct & known & ~mapped_eligible, "admission_ineligible"
    )
    direct_start = frame["hadm_id"].map(lookup["admittime"])
    direct_end = frame["hadm_id"].map(lookup["administrative_end_time"])
    direct_in_window = _closed_interval_mask(
        frame["specimen_time"],
        direct_start,
        direct_end,
        cohort.admission.charttime_boundary,
    ).fillna(False)
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"],
        direct & known & mapped_eligible & ~direct_in_window,
        "charttime_outside_admission",
    )
    _infer_null_hadm_ids(frame, admission_audit, cohort)

    for column in (
        "admittime",
        "dischtime",
        "deathtime",
        "administrative_end_time",
        "administrative_end_type",
        "admission_type",
        "age_at_admission",
        "base_exclusion_reasons",
    ):
        frame[column] = frame["hadm_id"].map(lookup[column])
    return frame


def _sort_for_dedup(frame: pd.DataFrame, cleaning: CleaningConfig) -> pd.DataFrame:
    if cleaning.deduplication.ordering == "storetime":
        columns = ["storetime", "labevent_id"]
    else:
        columns = ["labevent_id"]
    return frame.sort_values(columns, kind="mergesort", na_position="last").reset_index(drop=True)


def _apply_group_policy(
    frame: pd.DataFrame,
    member_mask: pd.Series,
    key: list[str],
    policy: str,
    reason: str,
) -> None:
    subset = frame.loc[member_mask]
    if subset.empty:
        return
    if policy == "error":
        example = subset[key].head(5).to_dict(orient="records")
        raise CohortExtractionError(f"{reason} groups found; examples: {example}")
    if policy == "reject_all":
        frame["rejection_reasons"] = _append_reason(frame["rejection_reasons"], member_mask, reason)
        return
    keep = "first" if policy in {"keep_first", "mean"} else "last"
    dropped = subset.duplicated(key, keep=keep)
    dropped_indices = subset.index[dropped]
    if policy == "mean":
        means = subset.groupby(key, dropna=False, sort=False)["creatinine_mg_dl"].transform("mean")
        retained_indices = subset.index[~dropped]
        frame.loc[retained_indices, "creatinine_mg_dl"] = means.loc[retained_indices]
    frame["rejection_reasons"] = _append_reason(
        frame["rejection_reasons"], frame.index.isin(dropped_indices), reason
    )


def _deduplicate(frame: pd.DataFrame, cleaning: CleaningConfig) -> pd.DataFrame:
    frame = _sort_for_dedup(frame, cleaning)
    dedup = cleaning.deduplication
    active = frame["rejection_reasons"].eq("")
    exact_key = list(dedup.exact_key)
    exact_groups = active & frame.loc[active].duplicated(exact_key, keep=False).reindex(
        frame.index, fill_value=False
    )
    _apply_group_policy(
        frame,
        exact_groups,
        exact_key,
        dedup.exact_policy,
        "exact_duplicate",
    )
    if not dedup.simultaneous_enabled:
        return frame

    active = frame["rejection_reasons"].eq("")
    key = list(dedup.simultaneous_key)
    subset = frame.loc[active]
    if subset.empty:
        return frame
    grouped = subset.groupby(key, dropna=False, sort=False)["creatinine_mg_dl"]
    counts = grouped.transform("size")
    spans = grouped.transform("max") - grouped.transform("min")
    duplicate = counts > 1
    concordant_indices = subset.index[duplicate & (spans <= dedup.value_tolerance_mg_dl)]
    conflicting_indices = subset.index[duplicate & (spans > dedup.value_tolerance_mg_dl)]
    _apply_group_policy(
        frame,
        frame.index.isin(concordant_indices),
        key,
        dedup.concordant_policy,
        "concordant_simultaneous_duplicate",
    )
    _apply_group_policy(
        frame,
        frame.index.isin(conflicting_indices) & frame["rejection_reasons"].eq(""),
        key,
        dedup.conflicting_policy,
        "conflicting_simultaneous_measurement",
    )
    return frame


def _measurement_counts(frame: pd.DataFrame, cohort: CohortConfig) -> pd.Series:
    active = frame.loc[frame["rejection_reasons"].eq("")]
    if cohort.measurement_count_basis == "rows":
        return active.groupby("hadm_id", dropna=False).size().astype("Int64")
    if cohort.measurement_count_basis == "distinct_timestamps":
        return active.groupby("hadm_id", dropna=False)["specimen_time"].nunique().astype("Int64")

    missing_specimen = active["specimen_id"].isna()
    if cohort.missing_specimen_id_policy == "reject_measurement":
        indices = active.index[missing_specimen]
        frame["rejection_reasons"] = _append_reason(
            frame["rejection_reasons"],
            frame.index.isin(indices),
            "missing_specimen_id_for_count_basis",
        )
        active = frame.loc[frame["rejection_reasons"].eq("")]
        return active.groupby("hadm_id", dropna=False)["specimen_id"].nunique().astype("Int64")
    known = active.loc[~missing_specimen].groupby("hadm_id")["specimen_id"].nunique()
    if cohort.missing_specimen_id_policy == "exclude_from_count":
        return known.astype("Int64")
    missing = active.loc[missing_specimen].groupby("hadm_id").size()
    return known.add(missing, fill_value=0).astype("Int64")


def _reason_counts(
    frame: pd.DataFrame, column: str, *, entity: str, stage: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exploded = frame.loc[frame[column].fillna("").ne(""), column].str.split(";").explode()
    for reason, count in exploded.value_counts().sort_index().items():
        rows.append({"entity": entity, "stage": stage, "reason": reason, "count": int(count)})
    return rows


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def extract_creatinine_cohort(
    config: AkiAuditConfig | str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    chunksize: int = 500_000,
    overwrite: bool = False,
) -> CohortExtractionResult:
    """Extract the admission-bounded adult creatinine cohort and full audits.

    The large ``labevents`` table is read in chunks.  Only item 50912 rows are
    retained in memory.  No output is replaced unless ``overwrite=True``.
    """

    protocol = load_aki_config(config) if isinstance(config, (str, Path)) else config
    if not isinstance(protocol, AkiAuditConfig):
        raise TypeError("config must be an AkiAuditConfig or YAML path")
    raw = Path(raw_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    artifacts = {
        "measurements": destination / "aki_creatinine_measurements.parquet",
        "rejections": destination / "aki_creatinine_rejections.parquet",
        "admissions": destination / "aki_admission_audit.parquet",
        "flow": destination / "aki_cohort_flow.csv",
        "units": destination / "aki_unit_audit.csv",
    }
    existing = [str(path) for path in artifacts.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"AKI extraction artifacts already exist: {existing}")

    labevents_path = _resolve_table(raw, "labevents")
    admissions_path = _resolve_table(raw, "admissions")
    patients_path = _resolve_table(raw, "patients")
    d_labitems_path = _resolve_table(raw, "d_labitems")
    validate_creatinine_labitem(d_labitems_path, protocol.cohort.creatinine_itemid)
    admission_audit = _prepare_admission_audit(
        admissions_path,
        patients_path,
        protocol.cohort,
        protocol.config_hash,
    )

    scan_counts = {"labevents_rows_scanned": 0, "creatinine_rows_found": 0}
    candidate_parts: list[pd.DataFrame] = []
    for chunk in stream_creatinine_candidates(
        labevents_path,
        itemid=protocol.cohort.creatinine_itemid,
        cleaning=protocol.cleaning,
        chunksize=chunksize,
        scan_counts=scan_counts,
    ):
        candidate_parts.append(_link_admissions(chunk, admission_audit, protocol.cohort))
    if candidate_parts:
        candidates = pd.concat(candidate_parts, ignore_index=True)
    else:
        candidates = pd.DataFrame(
            columns=[
                *LABEVENT_COLUMNS,
                "source_hadm_id",
                "specimen_time",
                "raw_value",
                "raw_valuenum",
                "raw_unit",
                "creatinine_mg_dl",
                "creatinine_unit",
                "rejection_reasons",
            ]
        )
    candidates = _deduplicate(candidates, protocol.cleaning)
    candidate_rows_by_admission = candidates.groupby("hadm_id", dropna=False).size()
    counts = _measurement_counts(candidates, protocol.cohort)
    valid_rows_before_cohort = (
        candidates.loc[candidates["rejection_reasons"].eq("")]
        .groupby("hadm_id", dropna=False)
        .size()
    )

    admission_audit["creatinine_candidate_rows"] = (
        admission_audit["hadm_id"].map(candidate_rows_by_admission).fillna(0).astype("Int64")
    )
    admission_audit["valid_measurement_rows_before_cohort"] = (
        admission_audit["hadm_id"].map(valid_rows_before_cohort).fillna(0).astype("Int64")
    )
    admission_audit["invalid_or_duplicate_measurement_rows"] = (
        admission_audit["creatinine_candidate_rows"]
        - admission_audit["valid_measurement_rows_before_cohort"]
    )
    admission_audit["clean_measurement_count"] = (
        admission_audit["hadm_id"].map(counts).fillna(0).astype("Int64")
    )
    admission_audit["measurement_count_basis"] = protocol.cohort.measurement_count_basis
    admission_audit["minimum_measurements_required"] = protocol.cohort.minimum_measurements
    admission_audit["cohort_exclusion_reasons"] = admission_audit["base_exclusion_reasons"].copy()
    below_minimum = admission_audit["base_eligible"] & (
        admission_audit["clean_measurement_count"] < protocol.cohort.minimum_measurements
    )
    admission_audit["cohort_exclusion_reasons"] = _append_reason(
        admission_audit["cohort_exclusion_reasons"],
        below_minimum,
        "insufficient_creatinine_measurements",
    )
    admission_audit["cohort_included"] = admission_audit["cohort_exclusion_reasons"].eq("")
    included_hadm = set(admission_audit.loc[admission_audit["cohort_included"], "hadm_id"])
    clean_but_excluded = candidates["rejection_reasons"].eq("") & ~candidates["hadm_id"].isin(
        included_hadm
    )
    candidates["rejection_reasons"] = _append_reason(
        candidates["rejection_reasons"],
        clean_but_excluded,
        "admission_excluded_from_cohort",
    )
    candidates["config_hash"] = protocol.config_hash
    retained = candidates.loc[candidates["rejection_reasons"].eq("")].copy()
    retained = retained.sort_values(
        ["subject_id", "hadm_id", "specimen_time", "labevent_id"], kind="mergesort"
    ).reset_index(drop=True)
    for column in MEASUREMENT_COLUMNS:
        if column not in retained:
            retained[column] = pd.NA
    retained = retained[MEASUREMENT_COLUMNS]
    rejected = candidates.loc[candidates["rejection_reasons"].ne("")].copy()
    rejected = rejected.sort_values(
        ["subject_id", "specimen_time", "labevent_id"], kind="mergesort"
    )

    final_counts = (
        retained.groupby("hadm_id").size() if not retained.empty else pd.Series(dtype=int)
    )
    admission_audit["retained_measurement_rows"] = (
        admission_audit["hadm_id"].map(final_counts).fillna(0).astype("Int64")
    )
    admission_audit = admission_audit.sort_values(["subject_id", "admittime", "hadm_id"])

    flow_rows = [
        {
            "entity": "labevent",
            "stage": "source",
            "reason": "rows_scanned",
            "count": scan_counts["labevents_rows_scanned"],
        },
        {
            "entity": "measurement",
            "stage": "item_filter",
            "reason": f"itemid_{protocol.cohort.creatinine_itemid}",
            "count": scan_counts["creatinine_rows_found"],
        },
        {
            "entity": "admission",
            "stage": "base_eligibility",
            "reason": "eligible",
            "count": int(admission_audit["base_eligible"].sum()),
        },
        {
            "entity": "admission",
            "stage": "final_cohort",
            "reason": "included",
            "count": int(admission_audit["cohort_included"].sum()),
        },
        {
            "entity": "subject",
            "stage": "final_cohort",
            "reason": "included",
            "count": int(retained["subject_id"].nunique()),
        },
        {
            "entity": "measurement",
            "stage": "final_cohort",
            "reason": "included",
            "count": len(retained),
        },
    ]
    flow_rows.extend(
        _reason_counts(candidates, "rejection_reasons", entity="measurement", stage="rejection")
    )
    flow_rows.extend(
        _reason_counts(
            admission_audit,
            "cohort_exclusion_reasons",
            entity="admission",
            stage="exclusion",
        )
    )
    flow = pd.DataFrame(flow_rows)
    denominators = {
        ("labevent", "source"): scan_counts["labevents_rows_scanned"],
        ("measurement", "item_filter"): scan_counts["labevents_rows_scanned"],
        ("admission", "base_eligibility"): len(admission_audit),
        ("admission", "final_cohort"): len(admission_audit),
        ("subject", "final_cohort"): int(admission_audit["subject_id"].nunique()),
        ("measurement", "final_cohort"): scan_counts["creatinine_rows_found"],
        ("measurement", "rejection"): scan_counts["creatinine_rows_found"],
        ("admission", "exclusion"): len(admission_audit),
    }
    flow["denominator"] = [denominators[(row.entity, row.stage)] for row in flow.itertuples()]
    flow["fraction"] = flow["count"].div(flow["denominator"].replace(0, np.nan))
    flow["config_hash"] = protocol.config_hash

    if candidates.empty:
        unit_audit = pd.DataFrame(
            columns=["raw_unit", "candidate_rows", "rejected_rows", "retained_rows"]
        )
    else:
        unit_source = candidates.assign(
            raw_unit=candidates["raw_unit"].astype("string").fillna("<MISSING>"),
            rejected=candidates["rejection_reasons"].ne("").astype(int),
            retained=candidates["rejection_reasons"].eq("").astype(int),
        )
        unit_audit = (
            unit_source.groupby("raw_unit", dropna=False)
            .agg(
                candidate_rows=("labevent_id", "size"),
                rejected_rows=("rejected", "sum"),
                retained_rows=("retained", "sum"),
            )
            .reset_index()
        )
    unit_audit["config_hash"] = protocol.config_hash

    destination.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(retained, artifacts["measurements"])
    _atomic_parquet(rejected, artifacts["rejections"])
    _atomic_parquet(admission_audit, artifacts["admissions"])
    _atomic_csv(flow, artifacts["flow"])
    _atomic_csv(unit_audit, artifacts["units"])
    return CohortExtractionResult(
        measurements_path=artifacts["measurements"],
        rejected_measurements_path=artifacts["rejections"],
        admission_audit_path=artifacts["admissions"],
        flow_audit_path=artifacts["flow"],
        unit_audit_path=artifacts["units"],
        config_hash=protocol.config_hash,
        labevents_rows_scanned=scan_counts["labevents_rows_scanned"],
        creatinine_rows_found=scan_counts["creatinine_rows_found"],
        retained_measurements=len(retained),
        retained_admissions=int(admission_audit["cohort_included"].sum()),
        retained_subjects=int(retained["subject_id"].nunique()),
    )


__all__ = [
    "CohortExtractionError",
    "CohortExtractionResult",
    "extract_creatinine_cohort",
    "stream_creatinine_candidates",
    "validate_creatinine_labitem",
]
