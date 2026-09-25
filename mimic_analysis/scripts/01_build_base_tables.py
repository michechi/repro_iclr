"""Step 1: load MIMIC core tables and build the adult inpatient base note set.

Outputs (parquet):
    data/intermediate/cohort/patients.parquet
    data/intermediate/cohort/admissions.parquet
    data/intermediate/cohort/diagnoses_icd.parquet
    data/intermediate/cohort/d_icd_diagnoses.parquet
    data/intermediate/cohort/base_adult_inpatient_notes.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from mimic_cancer.io import (  # noqa: E402
    load_admissions,
    load_diagnoses,
    load_diagnoses_dict,
    load_patients,
    stream_radiology_chunks,
)
from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import (  # noqa: E402
    ensure_directories,
    load_cohort_config,
    load_paths,
)

logger = setup_logging("01_build_base_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build base MIMIC tables + adult inpatient notes.")
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--cohort-config", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--max-notes",
        type=int,
        default=None,
        help="Cap on total radiology metadata rows retained (for smoke tests).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    cohort_cfg = load_cohort_config(args.cohort_config)
    ensure_directories(paths)
    min_age = int(cohort_cfg.get("min_anchor_age", 18))
    require_hadm = bool(cohort_cfg.get("require_hadm_id", True))

    logger.info("loading patients...")
    patients = load_patients(paths.patients_csv)
    patients.to_parquet(paths.cohort_dir / "patients.parquet")
    logger.info("  %d patients", len(patients))

    logger.info("loading admissions...")
    admissions = load_admissions(paths.admissions_csv)
    admissions.to_parquet(paths.cohort_dir / "admissions.parquet")
    logger.info("  %d admissions", len(admissions))

    logger.info("loading diagnoses_icd...")
    diagnoses = load_diagnoses(paths.diagnoses_icd_csv)
    diagnoses.to_parquet(paths.cohort_dir / "diagnoses_icd.parquet")
    logger.info("  %d diagnosis rows", len(diagnoses))

    logger.info("loading d_icd_diagnoses...")
    dx_dict = load_diagnoses_dict(paths.d_icd_diagnoses_csv)
    dx_dict.to_parquet(paths.cohort_dir / "d_icd_diagnoses.parquet")
    logger.info("  %d ICD dictionary rows", len(dx_dict))

    adult_subjects = set(
        patients.loc[patients["anchor_age"] >= min_age, "subject_id"].tolist()
    )
    logger.info("adult subjects (age >= %d): %d", min_age, len(adult_subjects))

    logger.info("streaming radiology metadata (no text) to build base note table...")
    parts: list[pd.DataFrame] = []
    total_raw = 0
    total_kept = 0
    for chunk in stream_radiology_chunks(
        paths.radiology_csv, chunksize=args.chunksize, include_text=False
    ):
        total_raw += len(chunk)
        chunk = chunk[chunk["subject_id"].isin(adult_subjects)]
        if require_hadm:
            chunk = chunk[chunk["hadm_id"].notna()]
        chunk["event_time"] = chunk["charttime"].fillna(chunk["storetime"])
        chunk = chunk.dropna(subset=["event_time"])
        total_kept += len(chunk)
        parts.append(chunk)
        if args.max_notes is not None and total_kept >= args.max_notes:
            break

    base = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if args.max_notes is not None and len(base) > args.max_notes:
        base = base.head(args.max_notes).copy()

    out_path = paths.cohort_dir / "base_adult_inpatient_notes.parquet"
    base.to_parquet(out_path)
    logger.info(
        "raw radiology rows=%d retained=%d -> %s", total_raw, len(base), out_path
    )
    log_peak_memory(logger, "build_base_tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
