"""Step 8: run the QC checks and emit preprocessing_report.json.

Exits non-zero if any mandatory sanity/leakage check fails.
"""
from __future__ import annotations

import argparse
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
    load_cohort_config,
    load_paths,
    load_thresholds_config,
)
from mimic_cancer.qc import (  # noqa: E402
    assert_binary,
    assert_monotonic_within,
    assert_no_duplicates,
    assert_no_null,
    build_report,
    has_failures,
    write_report,
)

logger = setup_logging("08_qc_reports")

FLAG_COLS = [f"flag_{label}" for label in LABEL_ORDER]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--cohort-config", type=Path, default=None)
    parser.add_argument("--thresholds-config", type=Path, default=None)
    return parser.parse_args()


def _try_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    cohort_cfg = load_cohort_config(args.cohort_config)
    thresholds_cfg = load_thresholds_config(args.thresholds_config)
    ensure_directories(paths)

    base = _try_read(paths.cohort_dir / "base_adult_inpatient_notes.parquet")
    candidate = _try_read(paths.notes_dir / "candidate_radiology_notes.parquet")
    preds = _try_read(paths.predictions_dir / "radiology_cancer_predictions.parquet")
    notes = _try_read(paths.sequences_dir / "note_level_trajectories.parquet")

    a_labels = _try_read(paths.dataset_order_only_dir / "labels.parquet")
    a_seq = _try_read(paths.dataset_order_only_dir / "X_seq_context.parquet")
    b_labels = _try_read(paths.dataset_mortality365_dir / "labels.parquet")
    b_seq = _try_read(paths.dataset_mortality365_dir / "X_seq.parquet")

    counts = {
        "base_adult_inpatient_notes": len(base),
        "candidate_radiology_notes": len(candidate),
        "radiology_cancer_predictions": len(preds),
        "retained_cancer_positive_notes": len(notes),
        "patients_with_trajectories": int(notes["subject_id"].nunique()) if not notes.empty else 0,
        "dataset_a_strict_patients": len(a_labels),
        "dataset_a_y_order_1": int((a_labels.get("y_order", pd.Series(dtype=int)) == 1).sum()),
        "dataset_a_y_order_0": int((a_labels.get("y_order", pd.Series(dtype=int)) == 0).sum()),
        "dataset_b_patients": len(b_labels),
        "dataset_b_y_death_1": int((b_labels.get("y_death_365", pd.Series(dtype=int)) == 1).sum()),
        "dataset_b_y_death_0": int((b_labels.get("y_death_365", pd.Series(dtype=int)) == 0).sum()),
    }

    sanity: list[str | None] = []
    if not notes.empty:
        sanity.append(assert_no_duplicates(notes, "note_id"))
        sanity.append(assert_monotonic_within(notes, "subject_id", "event_time"))
        sanity.append(assert_no_null(notes, "subject_id"))
        sanity.append(assert_no_null(notes, "event_time"))
    if not a_labels.empty:
        sanity.append(assert_binary(a_labels, "y_order"))
    if not b_labels.empty:
        sanity.append(assert_binary(b_labels, "y_death_365"))
        sanity.append(assert_no_null(b_labels, "t0"))

    leakage: list[str | None] = []
    if not a_seq.empty and not a_labels.empty:
        joined = a_seq.merge(a_labels[["subject_id", "t_cut"]], on="subject_id", how="left")
        if (joined["event_time"] > joined["t_cut"]).any():
            n = int((joined["event_time"] > joined["t_cut"]).sum())
            leakage.append(f"Dataset A: {n} notes past t_cut")
    if not b_seq.empty and not b_labels.empty:
        joined = b_seq.merge(b_labels[["subject_id", "t0"]], on="subject_id", how="left")
        if (joined["event_time"] > joined["t0"]).any():
            n = int((joined["event_time"] > joined["t0"]).sum())
            leakage.append(f"Dataset B: {n} notes past t0")

    addenda = {
        "notes_with_addendum_flag": int(notes.get("has_addendum", pd.Series(dtype=bool)).sum()) if not notes.empty else 0,
        "orphan_addenda": int(notes.get("is_orphan_addendum", pd.Series(dtype=bool)).sum()) if not notes.empty else 0,
    }

    config_snapshot = {
        "cohort": cohort_cfg,
        "thresholds": thresholds_cfg,
        "paths": {
            "intermediate": str(paths.intermediate),
            "processed": str(paths.processed),
        },
    }

    report = build_report(counts, sanity, leakage, addenda, config_snapshot)
    out_path = paths.qc_dir / "preprocessing_report.json"
    write_report(report, out_path)

    logger.info("wrote QC report -> %s", out_path)
    logger.info("counts: %s", counts)
    if report["sanity"]:
        logger.error("sanity failures: %s", report["sanity"])
    if report["leakage"]:
        logger.error("leakage failures: %s", report["leakage"])

    log_peak_memory(logger, "qc_reports")
    return 1 if has_failures(report) else 0


if __name__ == "__main__":
    sys.exit(main())
