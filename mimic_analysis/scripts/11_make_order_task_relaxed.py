"""Step 11: relaxed Dataset A (Sensitivity A from docs/mimic_cancer.md §15).

Same cohort logic as step 05 but with the strict "exactly one progression
and one response note before t_cut" filter removed. The label `y_order`
is still based on the first-occurrence order of progression vs. response.

Output directory: data/processed/dataset_order_only_relaxed/
"""
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
from mimic_cancer.trajectory import build_dataset_a  # noqa: E402

logger = setup_logging("11_make_order_task_relaxed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--thresholds-config", type=Path, default=None)
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

    logger.info("loading note-level trajectories...")
    notes = pd.read_parquet(notes_path)
    logger.info("  %d notes, %d subjects", len(notes), notes["subject_id"].nunique())

    tables = build_dataset_a(notes, strict=False)
    labels = tables["labels"]

    if labels.empty:
        logger.warning("no patients qualify for relaxed Dataset A")
    else:
        split_df = make_patient_splits(
            subjects=labels["subject_id"].tolist(),
            strata=labels["y_order"].tolist(),
            seed=seed,
            train=train,
            val=val,
            test=test,
        )
        split_df = split_df.rename(columns={"split": "split_order_task"})
        labels = labels.drop(columns=["split"]).merge(
            split_df, on="subject_id", how="left"
        )
        splits_path = paths.splits_dir / "patient_splits_order_relaxed.parquet"
        split_df.to_parquet(splits_path)
        logger.info("wrote %d relaxed order-task splits -> %s", len(split_df), splits_path)

    out_dir = paths.dataset_order_only_relaxed_dir
    labels.to_parquet(out_dir / "labels.parquet")
    tables["seq_context"].to_parquet(out_dir / "X_seq_context.parquet")
    tables["seq_core"].to_parquet(out_dir / "X_seq_core.parquet")
    tables["bag_context"].to_parquet(out_dir / "X_bag_context.parquet")
    tables["last_event"].to_parquet(out_dir / "X_last_event.parquet")

    n_strict = int((labels["strict_primary_subset"] == 1).sum()) if not labels.empty else 0
    logger.info(
        "Relaxed Dataset A: %d patients (%d strict, %d relaxed-only), y_order=1 count=%d, y_order=0 count=%d",
        len(labels),
        n_strict,
        len(labels) - n_strict,
        int((labels["y_order"] == 1).sum()) if not labels.empty else 0,
        int((labels["y_order"] == 0).sum()) if not labels.empty else 0,
    )
    log_peak_memory(logger, "make_order_task_relaxed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
