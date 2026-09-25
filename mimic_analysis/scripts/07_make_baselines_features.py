"""Step 7: materialize shared baseline feature tables + JSONL serialization.

Produces, for both datasets:
  X_seq_context.jsonl / X_seq.jsonl   (one record per patient)
  patient_splits.parquet              (combined order + mortality splits)
  shuffle_index.parquet               (within-patient permutation, fixed seed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402

from mimic_cancer.dfci_imaging_model import LABEL_ORDER  # noqa: E402
from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import (  # noqa: E402
    ensure_directories,
    load_paths,
    load_thresholds_config,
)
from mimic_cancer.splits import make_shuffle_index  # noqa: E402

logger = setup_logging("07_make_baselines_features")

FLAG_COLS = [f"flag_{label}" for label in LABEL_ORDER]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--thresholds-config", type=Path, default=None)
    return parser.parse_args()


def serialize_seq_jsonl(
    seq_df: pd.DataFrame, labels_df: pd.DataFrame, label_col: str, out_path: Path
) -> int:
    """Emit one JSON record per patient with aligned times/labels arrays."""
    if seq_df.empty or labels_df.empty:
        out_path.write_text("")
        return 0
    label_map = dict(zip(labels_df["subject_id"], labels_df[label_col]))
    groups = seq_df.sort_values(["subject_id", "seq_idx"]).groupby("subject_id")
    n = 0
    with out_path.open("w") as f:
        for subject_id, g in groups:
            if subject_id not in label_map:
                continue
            rec = {
                "subject_id": int(subject_id),
                "times": g["event_time"].astype(str).tolist(),
                "delta_days": g["delta_days_from_prev_note"].astype(float).tolist(),
                "note_ids": g["note_id"].tolist(),
                "hadm_ids": g["hadm_id"].astype("Int64").tolist(),
                "labels": [
                    {label: int(row[f"flag_{label}"]) for label in LABEL_ORDER}
                    for _, row in g.iterrows()
                ],
                "y": int(label_map[subject_id]),
            }
            f.write(json.dumps(rec, default=str) + "\n")
            n += 1
    return n


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    thr = load_thresholds_config(args.thresholds_config)
    ensure_directories(paths)
    seed = int(thr.get("split_seed", 2026))

    # Dataset A
    a_dir = paths.dataset_order_only_dir
    a_labels = pd.read_parquet(a_dir / "labels.parquet") if (a_dir / "labels.parquet").exists() else pd.DataFrame()
    a_seq = pd.read_parquet(a_dir / "X_seq_context.parquet") if (a_dir / "X_seq_context.parquet").exists() else pd.DataFrame()
    n_a = serialize_seq_jsonl(a_seq, a_labels, "y_order", a_dir / "X_seq_context.jsonl") if not a_labels.empty else 0
    logger.info("Dataset A JSONL: %d records", n_a)

    # Dataset B
    b_dir = paths.dataset_mortality365_dir
    b_labels = pd.read_parquet(b_dir / "labels.parquet") if (b_dir / "labels.parquet").exists() else pd.DataFrame()
    b_seq = pd.read_parquet(b_dir / "X_seq.parquet") if (b_dir / "X_seq.parquet").exists() else pd.DataFrame()
    n_b = serialize_seq_jsonl(b_seq, b_labels, "y_death_365", b_dir / "X_seq.jsonl") if not b_labels.empty else 0
    logger.info("Dataset B JSONL: %d records", n_b)

    # Combined patient splits
    merged = None
    order_splits_path = paths.splits_dir / "patient_splits_order.parquet"
    mortality_splits_path = paths.splits_dir / "patient_splits_mortality.parquet"
    if order_splits_path.exists():
        merged = pd.read_parquet(order_splits_path)
    if mortality_splits_path.exists():
        mdf = pd.read_parquet(mortality_splits_path)
        merged = mdf if merged is None else merged.merge(mdf, on="subject_id", how="outer")
    if merged is not None:
        merged.to_parquet(paths.splits_dir / "patient_splits.parquet")
        logger.info("wrote combined patient_splits.parquet with %d rows", len(merged))

    # Shuffled-order control
    shuffle_rows: list[pd.DataFrame] = []
    if not a_seq.empty:
        counts_a = a_seq.groupby("subject_id").size()
        shuffle_rows.append(
            make_shuffle_index(counts_a.index.tolist(), counts_a.tolist(), seed=seed)
            .assign(source="order_only_context")
        )
    if not b_seq.empty:
        counts_b = b_seq.groupby("subject_id").size()
        shuffle_rows.append(
            make_shuffle_index(counts_b.index.tolist(), counts_b.tolist(), seed=seed)
            .assign(source="mortality365")
        )
    if shuffle_rows:
        shuffled = pd.concat(shuffle_rows, ignore_index=True)
        shuffled.to_parquet(paths.splits_dir / "shuffle_index.parquet")
        logger.info("wrote shuffle_index.parquet with %d rows", len(shuffled))

    log_peak_memory(logger, "make_baselines_features")
    return 0


if __name__ == "__main__":
    sys.exit(main())
