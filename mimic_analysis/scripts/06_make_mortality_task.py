"""Step 6: build Dataset B (1-year mortality from last cancer note's admission)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import (  # noqa: E402
    ensure_directories,
    load_paths,
    load_thresholds_config,
)
from mimic_cancer.splits import make_patient_splits  # noqa: E402
from mimic_cancer.trajectory import build_dataset_b  # noqa: E402

logger = setup_logging("06_make_mortality_task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--thresholds-config", type=Path, default=None)
    parser.add_argument("--max-notes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    thr = load_thresholds_config(args.thresholds_config)
    ensure_directories(paths)
    seed = int(thr.get("split_seed", 2026))
    train = float(thr.get("train_frac", 0.70))
    val = float(thr.get("val_frac", 0.15))
    test = float(thr.get("test_frac", 0.15))

    notes_path = paths.sequences_dir / "note_level_trajectories.parquet"
    if not notes_path.exists():
        logger.error("missing %s -- run step 04 first", notes_path)
        return 2

    logger.info("loading note-level trajectories + patients + admissions...")
    notes = pd.read_parquet(notes_path)
    if args.max_notes is not None and len(notes) > args.max_notes:
        notes = notes.head(args.max_notes)

    patients = pd.read_parquet(paths.cohort_dir / "patients.parquet")
    admissions = pd.read_parquet(paths.cohort_dir / "admissions.parquet")

    tables = build_dataset_b(notes, patients, admissions)
    labels = tables["labels"]

    if labels.empty:
        logger.warning("no patients qualify for Dataset B")
    else:
        split_df = make_patient_splits(
            subjects=labels["subject_id"].tolist(),
            strata=labels["y_death_365"].tolist(),
            seed=seed,
            train=train,
            val=val,
            test=test,
        )
        split_df = split_df.rename(columns={"split": "split_mortality_task"})
        labels = labels.drop(columns=["split"]).merge(
            split_df, on="subject_id", how="left"
        )
        splits_path = paths.splits_dir / "patient_splits_mortality.parquet"
        split_df.to_parquet(splits_path)
        logger.info("wrote %d mortality splits -> %s", len(split_df), splits_path)

    out_dir = paths.dataset_mortality365_dir
    labels.to_parquet(out_dir / "labels.parquet")
    tables["seq"].to_parquet(out_dir / "X_seq.parquet")
    tables["bag"].to_parquet(out_dir / "X_bag.parquet")
    tables["last_event"].to_parquet(out_dir / "X_last_event.parquet")

    logger.info(
        "Dataset B: %d patients, y_death_365=1 count=%d, y_death_365=0 count=%d",
        len(labels),
        int((labels["y_death_365"] == 1).sum()) if not labels.empty else 0,
        int((labels["y_death_365"] == 0).sum()) if not labels.empty else 0,
    )
    log_peak_memory(logger, "make_mortality_task")
    return 0


if __name__ == "__main__":
    sys.exit(main())
