"""Step 0: validate that all raw inputs exist and the DFCI model loads cleanly.

This is the integration check: it loads the DFCI imaging state dict into
LabeledModel with `strict=True` and runs a single-note forward pass. If the
class definition ever drifts from the state dict, this step fails loudly.

Usage:
    python scripts/00_validate_inputs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from mimic_cancer.dfci_imaging_model import (  # noqa: E402
    LABEL_ORDER,
    load_dfci_imaging,
    load_tokenizer,
)
from mimic_cancer.logging_utils import log_peak_memory, setup_logging  # noqa: E402
from mimic_cancer.paths import load_paths  # noqa: E402

logger = setup_logging("00_validate_inputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate pipeline inputs.")
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip the DFCI model load + forward pass smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)

    required_files = {
        "patients": paths.patients_csv,
        "admissions": paths.admissions_csv,
        "diagnoses_icd": paths.diagnoses_icd_csv,
        "d_icd_diagnoses": paths.d_icd_diagnoses_csv,
        "radiology": paths.radiology_csv,
        "radiology_detail": paths.radiology_detail_csv,
        "dfci_imaging_weights": paths.dfci_imaging_weights,
    }

    missing = []
    for label, p in required_files.items():
        if p.exists():
            logger.info("OK   %-25s %s", label, p)
        else:
            logger.error("MISS %-25s %s", label, p)
            missing.append(label)

    if missing:
        logger.error("missing required files: %s", missing)
        return 2

    if args.skip_model_load:
        logger.info("skipping model load per flag")
        log_peak_memory(logger, "validate_inputs")
        return 0

    logger.info("loading DFCI imaging student model...")
    model = load_dfci_imaging(paths.dfci_imaging_weights, device="cpu")
    tokenizer = load_tokenizer()
    logger.info("model loaded; running single-note forward pass")

    # Synthetic smoke-test text written for this check; not derived from any MIMIC-IV-Note record.
    sample_text = (
        "CT abdomen with contrast. Findings: multiple hypodense hepatic lesions consistent "
        "with metastatic disease. Impression: progression of metastatic colorectal cancer "
        "with new liver lesions. Increasing nodal disease in the mesentery."
    ).lower().replace("\n", " ")
    encoded = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(encoded["input_ids"], encoded["attention_mask"])
    logits = [float(o.item()) for o in outputs]
    probs = [1.0 / (1.0 + pow(2.718281828, -l)) for l in logits]
    for label, logit, prob in zip(LABEL_ORDER, logits, probs):
        logger.info("  %-15s logit=%+7.3f  prob=%.3f", label, logit, prob)

    log_peak_memory(logger, "validate_inputs")
    logger.info("validation OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
