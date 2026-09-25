from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
MIMIC_ANALYSIS_ROOT = REPO_ROOT / "mimic_analysis"
CONFIGS_DIR = MIMIC_ANALYSIS_ROOT / "configs"


@dataclass(frozen=True)
class Paths:
    mimic_hosp: Path
    mimic_note: Path
    dfci_imaging_weights: Path
    intermediate: Path
    processed: Path

    @property
    def patients_csv(self) -> Path:
        return self.mimic_hosp / "patients.csv.gz"

    @property
    def admissions_csv(self) -> Path:
        return self.mimic_hosp / "admissions.csv.gz"

    @property
    def diagnoses_icd_csv(self) -> Path:
        return self.mimic_hosp / "diagnoses_icd.csv.gz"

    @property
    def d_icd_diagnoses_csv(self) -> Path:
        return self.mimic_hosp / "d_icd_diagnoses.csv.gz"

    @property
    def radiology_csv(self) -> Path:
        return self.mimic_note / "radiology.csv.gz"

    @property
    def radiology_detail_csv(self) -> Path:
        return self.mimic_note / "radiology_detail.csv.gz"

    @property
    def cohort_dir(self) -> Path:
        return self.intermediate / "cohort"

    @property
    def notes_dir(self) -> Path:
        return self.intermediate / "notes"

    @property
    def predictions_dir(self) -> Path:
        return self.intermediate / "predictions"

    @property
    def sequences_dir(self) -> Path:
        return self.intermediate / "sequences"

    @property
    def splits_dir(self) -> Path:
        return self.intermediate / "splits"

    @property
    def qc_dir(self) -> Path:
        return self.intermediate / "qc"

    @property
    def dataset_order_only_dir(self) -> Path:
        return self.processed / "dataset_order_only"

    @property
    def dataset_order_only_relaxed_dir(self) -> Path:
        return self.processed / "dataset_order_only_relaxed"

    @property
    def dataset_mortality365_dir(self) -> Path:
        return self.processed / "dataset_mortality365"


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_paths(config_path: Path | None = None) -> Paths:
    if config_path is None:
        config_path = CONFIGS_DIR / "paths.yaml"
    raw = load_yaml(config_path)
    def resolve(key: str) -> Path:
        # Environment values override YAML; relative YAML paths are portable
        # across working directories and remain inside this analysis package.
        value = os.environ.get(f"MIMIC_CANCER_{key.upper()}", raw[key])
        path = Path(os.path.expandvars(value)).expanduser()
        return path if path.is_absolute() else MIMIC_ANALYSIS_ROOT / path

    return Paths(
        mimic_hosp=resolve("mimic_hosp"),
        mimic_note=resolve("mimic_note"),
        dfci_imaging_weights=resolve("dfci_imaging_weights"),
        intermediate=resolve("intermediate"),
        processed=resolve("processed"),
    )


def load_cohort_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = CONFIGS_DIR / "cohort.yaml"
    return load_yaml(config_path)


def load_thresholds_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = CONFIGS_DIR / "thresholds.yaml"
    return load_yaml(config_path)


def ensure_directories(paths: Paths) -> None:
    for d in (
        paths.cohort_dir,
        paths.notes_dir,
        paths.predictions_dir,
        paths.sequences_dir,
        paths.splits_dir,
        paths.qc_dir,
        paths.dataset_order_only_dir,
        paths.dataset_order_only_relaxed_dir,
        paths.dataset_mortality365_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
