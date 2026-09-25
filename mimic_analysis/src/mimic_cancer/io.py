from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd


PATIENTS_COLS = [
    "subject_id",
    "gender",
    "anchor_age",
    "anchor_year",
    "anchor_year_group",
    "dod",
]

ADMISSIONS_COLS = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "deathtime",
    "admission_type",
    "admission_location",
    "discharge_location",
    "insurance",
    "language",
    "marital_status",
    "race",
]

DIAGNOSES_COLS = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
DX_DICT_COLS = ["icd_code", "icd_version", "long_title"]

RADIOLOGY_COLS = [
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
]

RADIOLOGY_META_COLS = [c for c in RADIOLOGY_COLS if c != "text"]


def load_patients(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=PATIENTS_COLS,
        dtype={
            "subject_id": "int32",
            "anchor_age": "int16",
            "anchor_year": "int16",
        },
        parse_dates=["dod"],
    )
    for c in ("gender", "anchor_year_group"):
        df[c] = df[c].astype("category")
    return df


def load_admissions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=ADMISSIONS_COLS,
        dtype={"subject_id": "int32", "hadm_id": "Int32"},
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    for c in (
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "race",
    ):
        df[c] = df[c].astype("category")
    return df


def load_diagnoses(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=DIAGNOSES_COLS,
        dtype={
            "subject_id": "int32",
            "hadm_id": "Int32",
            "seq_num": "Int16",
            "icd_code": "string",
            "icd_version": "int8",
        },
    )
    df["icd_code"] = df["icd_code"].str.strip()
    return df


def load_diagnoses_dict(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        compression="gzip",
        usecols=DX_DICT_COLS,
        dtype={"icd_code": "string", "icd_version": "int8", "long_title": "string"},
    )
    df["icd_code"] = df["icd_code"].str.strip()
    return df


def stream_radiology_chunks(
    path: Path,
    chunksize: int = 50_000,
    include_text: bool = True,
) -> Iterator[pd.DataFrame]:
    usecols = RADIOLOGY_COLS if include_text else RADIOLOGY_META_COLS
    dtype = {
        "subject_id": "int32",
        "hadm_id": "Int32",
        "note_seq": "Int16",
        "note_type": "string",
        "note_id": "string",
    }
    if include_text:
        dtype["text"] = "string"
    reader = pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        dtype=dtype,
        parse_dates=["charttime", "storetime"],
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        yield chunk


def load_radiology_detail_wide(path: Path) -> pd.DataFrame:
    """Read the long-format radiology_detail table and pivot to one row per note_id.

    Output columns: note_id, subject_id, exam_code, exam_name, cpt_code,
    parent_note_id, addendum_note_id. Takes the first value per (note_id, field_name).
    """
    long_df = pd.read_csv(
        path,
        compression="gzip",
        dtype={
            "note_id": "string",
            "subject_id": "int32",
            "field_name": "string",
            "field_value": "string",
            "field_ordinal": "Int16",
        },
    )
    keep_fields = {
        "exam_code",
        "exam_name",
        "cpt_code",
        "parent_note_id",
        "addendum_note_id",
    }
    long_df = long_df[long_df["field_name"].isin(keep_fields)].copy()
    long_df = long_df.sort_values(["note_id", "field_name", "field_ordinal"])
    first = long_df.drop_duplicates(["note_id", "field_name"], keep="first")
    wide = first.pivot(index="note_id", columns="field_name", values="field_value")
    for col in keep_fields:
        if col not in wide.columns:
            wide[col] = pd.NA
    wide = wide.reset_index()
    # Attach subject_id (stable per note_id)
    subj = first.drop_duplicates("note_id")[["note_id", "subject_id"]]
    wide = wide.merge(subj, on="note_id", how="left")
    wide.columns.name = None
    return wide[
        [
            "note_id",
            "subject_id",
            "exam_code",
            "exam_name",
            "cpt_code",
            "parent_note_id",
            "addendum_note_id",
        ]
    ]


def resolve_event_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add `event_time = charttime or storetime`, drop rows where both are null."""
    out = df.copy()
    out["event_time"] = out["charttime"].fillna(out["storetime"])
    out = out.dropna(subset=["event_time"])
    return out
