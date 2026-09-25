from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mimic.aki.cohort import (
    CohortExtractionError,
    _prepare_admission_audit,
    extract_creatinine_cohort,
    validate_creatinine_labitem,
)
from mimic.aki.config import (
    AdultConfig,
    AdmissionConfig,
    AkiAuditConfig,
    CleaningConfig,
    CohortConfig,
    DeduplicationConfig,
    UnitConversion,
)


def _protocol() -> AkiAuditConfig:
    cohort = CohortConfig(
        creatinine_itemid=50912,
        adult=AdultConfig(
            minimum_age_years=18,
            minimum_age_inclusive=True,
            age_calculation="anchor_year_delta",
        ),
        admission=AdmissionConfig(
            eligible_types="all",
            type_matching="exact",
            minimum_stay_hours=0,
            minimum_stay_inclusive=True,
            charttime_boundary="both",
            null_hadm_policy="link_if_unique",
            administrative_end_policy="earliest_discharge_or_death",
            invalid_death_time_policy="exclude_admission",
        ),
        minimum_measurements=3,
        measurement_count_basis="rows",
        missing_specimen_id_policy="exclude_from_count",
    )
    cleaning = CleaningConfig(
        numeric_value_source="valuenum_only",
        canonical_unit="mg/dL",
        unit_matching="casefold_strip",
        missing_unit_policy="reject",
        # Deliberately synthetic conversion for a transparent assertion; this
        # is test protocol data, not a production scientific default.
        unit_conversions=(
            UnitConversion(aliases=("mg/dL",), multiplier=1.0, offset=0.0),
            UnitConversion(aliases=("synthetic-u/L",), multiplier=0.01, offset=0.0),
        ),
        minimum_value_mg_dl=0.1,
        maximum_value_mg_dl=30.0,
        value_boundary="both",
        deduplication=DeduplicationConfig(
            ordering="labevent_id",
            exact_key=(
                "subject_id",
                "hadm_id",
                "specimen_id",
                "specimen_time",
                "creatinine_mg_dl",
            ),
            exact_policy="keep_first",
            simultaneous_enabled=True,
            simultaneous_key=("subject_id", "hadm_id", "specimen_time"),
            value_tolerance_mg_dl=0.0,
            concordant_policy="keep_first",
            conflicting_policy="reject_all",
        ),
    )
    return AkiAuditConfig(
        schema_version=1,
        cohort=cohort,
        cleaning=cleaning,
        episodes={},
        representations={},
        matching={},
        splitting={},
        models={},
        metrics={},
        config_hash="synthetic-test-protocol",
        source_path=None,
        _raw={},
    )


def _write_mini_mimic(raw: Path) -> None:
    raw.mkdir()
    pd.DataFrame(
        [{"itemid": 50912, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"}]
    ).to_csv(raw / "d_labitems.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": 1, "anchor_age": 45, "anchor_year": 2020},
            {"subject_id": 2, "anchor_age": 17, "anchor_year": 2020},
            {"subject_id": 3, "anchor_age": 50, "anchor_year": 2020},
        ]
    ).to_csv(raw / "patients.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": subject_id,
                "hadm_id": subject_id * 100,
                "admittime": "2020-01-01 00:00:00",
                "dischtime": "2020-01-10 00:00:00",
                "deathtime": pd.NA,
                "admission_type": "URGENT",
            }
            for subject_id in (1, 2, 3)
        ]
    ).to_csv(raw / "admissions.csv", index=False)

    rows: list[dict[str, object]] = []

    def measurement(
        labevent_id: int,
        subject_id: int,
        hadm_id: int | None,
        specimen_id: int,
        time: str,
        value: float,
        unit: str,
    ) -> None:
        rows.append(
            {
                "labevent_id": labevent_id,
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "specimen_id": specimen_id,
                "itemid": 50912,
                "order_provider_id": pd.NA,
                "charttime": time,
                "storetime": time,
                "value": str(value),
                "valuenum": value,
                "valueuom": unit,
                "ref_range_lower": pd.NA,
                "ref_range_upper": pd.NA,
                "flag": pd.NA,
                "priority": pd.NA,
                "comments": pd.NA,
            }
        )

    # Adult admission: three valid measurements. The second row exercises an
    # explicit unit conversion and the third exercises unique null-hadm linkage.
    measurement(1, 1, 100, 1, "2020-01-02", 1.0, "mg/dL")
    measurement(2, 1, 100, 2, "2020-01-03", 200.0, "synthetic-u/L")
    measurement(3, 1, None, 3, "2020-01-04", 1.5, "mg/dL")
    measurement(4, 1, 100, 3, "2020-01-04", 1.5, "mg/dL")  # exact duplicate
    measurement(5, 1, 100, 4, "2020-01-05", 1.0, "unknown")
    measurement(6, 1, 100, 5, "2020-01-06", 99.0, "mg/dL")

    # Child admission has enough measurements but fails the adult gate.
    for offset in range(3):
        measurement(10 + offset, 2, 200, 20 + offset, f"2020-01-0{offset + 2}", 1.0, "mg/dL")
    # Adult admission has only two measurements and fails the count gate.
    for offset in range(2):
        measurement(20 + offset, 3, 300, 30 + offset, f"2020-01-0{offset + 2}", 1.0, "mg/dL")

    pd.DataFrame(rows).to_csv(raw / "labevents.csv", index=False)


def test_chunked_extraction_cleans_and_audits_cohort(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_mini_mimic(raw)

    result = extract_creatinine_cohort(_protocol(), raw, tmp_path / "out", chunksize=2)

    measurements = pd.read_parquet(result.measurements_path)
    assert len(measurements) == 3
    assert set(measurements["subject_id"]) == {1}
    assert set(measurements["hadm_id"]) == {100}
    assert measurements["creatinine_unit"].eq("mg/dL").all()
    assert measurements.loc[measurements["labevent_id"] == 2, "creatinine_mg_dl"].item() == 2.0
    assert measurements.loc[measurements["labevent_id"] == 3, "hadm_id"].item() == 100
    assert {"admittime", "dischtime", "deathtime"}.issubset(measurements.columns)

    rejected = pd.read_parquet(result.rejected_measurements_path).set_index("labevent_id")
    assert "exact_duplicate" in rejected.loc[4, "rejection_reasons"]
    assert "unrecognized_unit" in rejected.loc[5, "rejection_reasons"]
    assert "value_above_valid_range" in rejected.loc[6, "rejection_reasons"]

    admissions = pd.read_parquet(result.admission_audit_path).set_index("hadm_id")
    assert bool(admissions.loc[100, "cohort_included"])
    assert "below_adult_age_threshold" in admissions.loc[200, "cohort_exclusion_reasons"]
    assert "insufficient_creatinine_measurements" in admissions.loc[300, "cohort_exclusion_reasons"]
    flow = pd.read_csv(result.flow_audit_path)
    assert {"denominator", "fraction"}.issubset(flow.columns)
    assert ((flow["reason"] == "included") & (flow["entity"] == "admission")).any()


def test_creatinine_dictionary_identity_is_validated(tmp_path: Path) -> None:
    dictionary = tmp_path / "d_labitems.csv"
    pd.DataFrame(
        [{"itemid": 50912, "label": "Not Creatinine", "fluid": "Blood", "category": "Chemistry"}]
    ).to_csv(dictionary, index=False)

    with pytest.raises(CohortExtractionError, match="locked serum-creatinine identity"):
        validate_creatinine_labitem(dictionary, 50912)


def test_administrative_endpoint_and_invalid_death_time_are_explicit(tmp_path: Path) -> None:
    patients = tmp_path / "patients.csv"
    admissions = tmp_path / "admissions.csv"
    pd.DataFrame(
        [
            {"subject_id": 1, "anchor_age": 60, "anchor_year": 2020},
            {"subject_id": 2, "anchor_age": 60, "anchor_year": 2020},
        ]
    ).to_csv(patients, index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 100,
                "admittime": "2020-01-01",
                "dischtime": "2020-01-10",
                "deathtime": "2020-01-03",
                "admission_type": "URGENT",
            },
            {
                "subject_id": 2,
                "hadm_id": 200,
                "admittime": "2020-01-01",
                "dischtime": "2020-01-10",
                "deathtime": "not-a-timestamp",
                "admission_type": "URGENT",
            },
        ]
    ).to_csv(admissions, index=False)

    audit = _prepare_admission_audit(
        admissions, patients, _protocol().cohort, "synthetic-test-protocol"
    ).set_index("hadm_id")

    assert audit.loc[100, "administrative_end_time"] == pd.Timestamp("2020-01-03")
    assert audit.loc[100, "administrative_end_type"] == "death"
    assert bool(audit.loc[200, "death_time_invalid"])
    assert "invalid_deathtime" in audit.loc[200, "base_exclusion_reasons"]
