"""Step 4: merge addenda, deduplicate, and build the note-level oncology trajectory.

Input:
  data/intermediate/predictions/radiology_cancer_predictions.parquet
  data/intermediate/cohort/admissions.parquet
  data/intermediate/notes/candidate_radiology_notes.parquet
  (plus radiology_detail.csv.gz for addendum metadata)

Output:
  data/intermediate/sequences/note_level_trajectories.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from mimic_cancer.io import load_radiology_detail_wide  # noqa: E402
from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import (  # noqa: E402
    ensure_directories,
    load_cohort_config,
    load_paths,
)
from mimic_cancer.trajectory import (  # noqa: E402
    build_note_level_table,
    compile_modality_map,
    merge_addenda,
)

logger = setup_logging("04_build_note_sequences")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--cohort-config", type=Path, default=None)
    parser.add_argument("--max-notes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    cohort_cfg = load_cohort_config(args.cohort_config)
    ensure_directories(paths)

    modality_map = compile_modality_map(cohort_cfg.get("modality_map", []))
    min_notes = int(cohort_cfg.get("min_retained_notes_per_patient", 2))

    preds_path = paths.predictions_dir / "radiology_cancer_predictions.parquet"
    meta_path = paths.notes_dir / "candidate_radiology_notes.parquet"
    if not preds_path.exists():
        logger.error("missing %s -- run step 03 first", preds_path)
        return 2
    if not meta_path.exists():
        logger.error("missing %s -- run step 02 first", meta_path)
        return 2

    logger.info("loading predictions...")
    preds = pd.read_parquet(preds_path)
    logger.info("  %d prediction rows", len(preds))

    logger.info("loading radiology metadata...")
    meta = pd.read_parquet(meta_path)
    logger.info("  %d metadata rows", len(meta))

    logger.info("loading radiology_detail (pivoted to wide)...")
    detail_wide = load_radiology_detail_wide(paths.radiology_detail_csv)
    logger.info("  %d distinct note_ids", len(detail_wide))

    logger.info("merging addenda (parent <- child OR of flags, max of probs)...")
    preds = merge_addenda(preds, detail_wide)
    logger.info("  %d predictions post-addendum", len(preds))

    logger.info("loading admissions...")
    admissions = pd.read_parquet(paths.cohort_dir / "admissions.parquet")

    logger.info("building note-level trajectory table...")
    notes = build_note_level_table(
        preds=preds,
        radiology_meta=meta,
        detail_wide=detail_wide,
        admissions=admissions,
        modality_map=modality_map,
        min_retained_notes_per_patient=min_notes,
    )

    if args.max_notes is not None and len(notes) > args.max_notes:
        notes = notes.head(args.max_notes).copy()

    out_path = paths.sequences_dir / "note_level_trajectories.parquet"
    notes.to_parquet(out_path)
    logger.info(
        "wrote %d retained cancer-positive notes across %d patients -> %s",
        len(notes),
        notes["subject_id"].nunique() if not notes.empty else 0,
        out_path,
    )
    log_peak_memory(logger, "build_note_sequences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
