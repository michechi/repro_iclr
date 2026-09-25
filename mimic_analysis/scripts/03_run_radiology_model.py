"""Step 3: run DFCI-imaging-student inference on candidate radiology notes.

Reads candidate_radiology_notes_with_text.parquet in chunks, runs inference
on CPU, and streams predictions into radiology_cancer_predictions.parquet.
The raw text is NEVER written to the processed artifact.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from mimic_cancer.dfci_imaging_model import (  # noqa: E402
    LABEL_ORDER,
    load_dfci_imaging,
    load_tokenizer,
    run_inference,
)
from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import (  # noqa: E402
    ensure_directories,
    load_paths,
    load_thresholds_config,
)

logger = setup_logging("03_run_radiology_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--thresholds-config", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--inference-chunk",
        type=int,
        default=5000,
        help="Rows of input parquet to read per inference batch flush.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-notes", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    thresholds_cfg = load_thresholds_config(args.thresholds_config)
    ensure_directories(paths)
    threshold = float(thresholds_cfg.get("binarization_threshold", 0.5))

    in_path = paths.notes_dir / "candidate_radiology_notes_with_text.parquet"
    out_path = paths.predictions_dir / "radiology_cancer_predictions.parquet"
    if not in_path.exists():
        logger.error("missing input %s -- run step 02 first", in_path)
        return 2
    if out_path.exists():
        out_path.unlink()

    logger.info("loading DFCI imaging model...")
    model = load_dfci_imaging(paths.dfci_imaging_weights, device=args.device)
    tokenizer = load_tokenizer()

    parquet_file = pq.ParquetFile(in_path)
    batch_iter = parquet_file.iter_batches(
        batch_size=args.inference_chunk,
        columns=["note_id", "subject_id", "hadm_id", "charttime", "storetime", "text"],
    )

    writer: pq.ParquetWriter | None = None
    total_notes = 0
    try:
        for batch in batch_iter:
            chunk_df = batch.to_pandas()
            if args.max_notes is not None and total_notes + len(chunk_df) > args.max_notes:
                chunk_df = chunk_df.head(args.max_notes - total_notes)
            if chunk_df.empty:
                break

            preds = run_inference(
                chunk_df[["note_id", "text"]],
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                batch_size=args.batch_size,
                binarization_threshold=threshold,
            )

            meta = chunk_df[["note_id", "subject_id", "hadm_id", "charttime", "storetime"]]
            joined = meta.merge(preds, on="note_id", how="inner")
            joined["event_time"] = joined["charttime"].fillna(joined["storetime"])
            joined = joined.dropna(subset=["event_time"])

            out_cols = (
                ["note_id", "subject_id", "hadm_id", "event_time", "charttime", "storetime"]
                + [f"logit_{label}" for label in LABEL_ORDER]
                + [f"prob_{label}" for label in LABEL_ORDER]
                + [f"flag_{label}" for label in LABEL_ORDER]
            )
            table = pa.Table.from_pandas(joined[out_cols], preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)

            total_notes += len(joined)
            logger.info("processed %d / ? notes", total_notes)

            del chunk_df, preds, meta, joined, table
            gc.collect()

            if args.max_notes is not None and total_notes >= args.max_notes:
                break
    finally:
        if writer is not None:
            writer.close()

    logger.info("wrote %d predictions -> %s", total_notes, out_path)
    log_peak_memory(logger, "run_radiology_model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
