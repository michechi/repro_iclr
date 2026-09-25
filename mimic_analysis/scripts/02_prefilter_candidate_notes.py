"""Step 2: build the high-recall cancer candidate note set.

Streams radiology.csv.gz in chunks, applies the cancer text prefilter and the
ICD-based admission/patient seed, and writes two parquet files:

  - candidate_radiology_notes_with_text.parquet  (with raw text; for inference)
  - candidate_radiology_notes.parquet            (without text; for joins)

Both are written via pyarrow ParquetWriter so we stream row groups into one
file per output without holding the whole result in memory.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from mimic_cancer.cancer_icd_seed import icd_cancer_seed_mask  # noqa: E402
from mimic_cancer.cancer_text_filter import text_cancer_seed_series  # noqa: E402
from mimic_cancer.io import load_diagnoses, load_patients, stream_radiology_chunks  # noqa: E402
from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import ensure_directories, load_paths  # noqa: E402

logger = setup_logging("02_prefilter_candidate_notes")

META_COLS = [
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text_cancer_seed",
    "icd_cancer_seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--max-notes", type=int, default=None)
    return parser.parse_args()


def build_icd_seed_sets(paths) -> tuple[set[int], set[int]]:
    """Return (seed_hadms, seed_subjects) — admissions and patients marked
    by any ICD neoplasm code."""
    logger.info("loading diagnoses for ICD seed...")
    diagnoses = load_diagnoses(paths.diagnoses_icd_csv)
    mask = icd_cancer_seed_mask(diagnoses)
    seeded = diagnoses[mask]
    seed_hadms = set(seeded["hadm_id"].dropna().astype("int64").tolist())
    seed_subjects = set(seeded["subject_id"].astype("int64").tolist())
    logger.info(
        "icd seed: %d unique admissions, %d unique patients",
        len(seed_hadms),
        len(seed_subjects),
    )
    # Keep RAM free
    del diagnoses, seeded
    return seed_hadms, seed_subjects


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    ensure_directories(paths)

    # Restrict to adult subjects
    patients = load_patients(paths.patients_csv)
    adult_subjects = set(patients.loc[patients["anchor_age"] >= 18, "subject_id"].astype("int64").tolist())
    del patients

    seed_hadms, seed_subjects = build_icd_seed_sets(paths)

    out_with_text = paths.notes_dir / "candidate_radiology_notes_with_text.parquet"
    out_no_text = paths.notes_dir / "candidate_radiology_notes.parquet"
    for p in (out_with_text, out_no_text):
        if p.exists():
            p.unlink()

    writer_full: pq.ParquetWriter | None = None
    writer_meta: pq.ParquetWriter | None = None
    total_raw = 0
    total_kept = 0
    try:
        for chunk in stream_radiology_chunks(
            paths.radiology_csv, chunksize=args.chunksize, include_text=True
        ):
            total_raw += len(chunk)
            chunk = chunk[chunk["subject_id"].isin(adult_subjects)]
            if chunk.empty:
                continue
            chunk = chunk[chunk["hadm_id"].notna()].copy()
            if chunk.empty:
                continue

            chunk["text_cancer_seed"] = text_cancer_seed_series(chunk["text"]).astype("bool")
            hadm_series = chunk["hadm_id"].astype("Int64")
            subj_series = chunk["subject_id"].astype("int64")
            chunk["icd_cancer_seed"] = (
                hadm_series.isin(seed_hadms) | subj_series.isin(seed_subjects)
            ).astype("bool")

            keep = chunk["text_cancer_seed"] | chunk["icd_cancer_seed"]
            chunk = chunk[keep].copy()
            if chunk.empty:
                continue

            total_kept += len(chunk)

            full_table = pa.Table.from_pandas(
                chunk[
                    [
                        "note_id",
                        "subject_id",
                        "hadm_id",
                        "note_type",
                        "note_seq",
                        "charttime",
                        "storetime",
                        "text",
                        "text_cancer_seed",
                        "icd_cancer_seed",
                    ]
                ],
                preserve_index=False,
            )
            meta_table = pa.Table.from_pandas(chunk[META_COLS], preserve_index=False)

            if writer_full is None:
                writer_full = pq.ParquetWriter(out_with_text, full_table.schema, compression="snappy")
            writer_full.write_table(full_table)
            if writer_meta is None:
                writer_meta = pq.ParquetWriter(out_no_text, meta_table.schema, compression="snappy")
            writer_meta.write_table(meta_table)

            if args.max_notes is not None and total_kept >= args.max_notes:
                logger.info("reached max_notes=%d early", args.max_notes)
                break
    finally:
        if writer_full is not None:
            writer_full.close()
        if writer_meta is not None:
            writer_meta.close()

    logger.info(
        "scanned %d radiology rows, kept %d candidate notes -> %s (with text) + %s (metadata)",
        total_raw,
        total_kept,
        out_with_text,
        out_no_text,
    )
    log_peak_memory(logger, "prefilter_candidate_notes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
