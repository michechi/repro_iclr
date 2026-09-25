"""Command-line entry point for the MIMIC-IV AKI trajectory audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .cohort import extract_creatinine_cohort
from .config import AkiAuditConfig, load_aki_config
from .experiment import run_audit_experiments
from .pipeline import build_aki_audit_datasets, persist_pipeline_result
from .reporting import write_run_manifest


ALL_MODELS = ("logistic_regression", "xgboost", "lstm", "transformer")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_table(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"input table not found: {resolved}")
    if resolved.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if resolved.name.casefold().endswith((".csv", ".csv.gz")):
        return pd.read_csv(resolved)
    raise ValueError(f"input table must be Parquet or CSV: {resolved}")


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _pipeline_summary(result: Any) -> dict[str, Any]:
    return {
        "episodes": len(result.episodes),
        "audited_decisions": len(result.episode_audit),
        "subjects_split": int(result.patient_splits["subject_id"].nunique()),
        "primary_episodes_before_matching": len(result.primary_labels),
        "primary_matched_episodes": len(result.primary_matched_labels),
        "secondary_episodes": len(result.secondary_labels),
        "matching_estimable": bool(result.matching.estimable),
        "non_estimable_reasons": list(result.matching.non_estimable_reasons),
    }


def _require_estimable(result: Any) -> None:
    if not result.matching.estimable:
        reasons = "; ".join(result.matching.non_estimable_reasons)
        raise RuntimeError(
            "the configured primary matched analysis is not estimable; "
            f"audit artifacts were retained. Reasons: {reasons}"
        )


def _run_prepare(
    config: AkiAuditConfig,
    measurements_path: str | Path,
    admissions_path: str | Path | None,
    output_dir: str | Path,
    *,
    max_workers: int = 1,
    patient_chunk_size: int = 128,
) -> Any:
    measurements = _read_table(measurements_path)
    admissions = None if admissions_path is None else _read_table(admissions_path)
    result = build_aki_audit_datasets(
        measurements,
        config,
        admissions=admissions,
        max_workers=max_workers,
        patient_chunk_size=patient_chunk_size,
    )
    persist_pipeline_result(result, output_dir)
    return result


def _run_train(
    config: AkiAuditConfig,
    prepared_dir: str | Path,
    output_dir: str | Path,
    models: Sequence[str],
) -> Any:
    source = Path(prepared_dir).expanduser().resolve()
    protocol_path = source / "protocol_metadata.csv"
    if not protocol_path.is_file():
        raise FileNotFoundError(
            f"prepared protocol metadata is required before training: {protocol_path}"
        )
    protocol_metadata = pd.read_csv(protocol_path)
    if len(protocol_metadata) != 1:
        raise ValueError("protocol_metadata.csv must contain exactly one row")
    prepared_hash = str(protocol_metadata.iloc[0]["config_hash"])
    if prepared_hash != config.config_hash:
        raise ValueError(
            "training protocol does not match prepared data: "
            f"prepared={prepared_hash}, requested={config.config_hash}"
        )
    estimability_path = source / "matching_estimability.csv"
    if not estimability_path.is_file():
        raise FileNotFoundError(
            f"matching estimability audit is required before training: {estimability_path}"
        )
    estimability = pd.read_csv(estimability_path)
    if len(estimability) != 1:
        raise ValueError("matching_estimability.csv must contain exactly one row")
    raw_estimable = estimability.iloc[0]["estimable"]
    is_estimable = (
        raw_estimable
        if isinstance(raw_estimable, bool)
        else str(raw_estimable).strip().casefold() == "true"
    )
    if not is_estimable:
        reasons = estimability.iloc[0].get("non_estimable_reasons_json", "[]")
        raise RuntimeError(f"primary cohort is marked non-estimable: {reasons}")
    datasets = {
        "binary": (
            _read_table(source / "primary_matched_events.parquet"),
            _read_table(source / "primary_matched_labels.parquet"),
        ),
        "multiclass": (
            _read_table(source / "secondary_events.parquet"),
            _read_table(source / "secondary_labels.parquet"),
        ),
    }
    return run_audit_experiments(
        datasets,
        config,
        output_dir=output_dir,
        enabled_models=models,
    )


def _validate_command(args: argparse.Namespace) -> None:
    config = load_aki_config(args.config)
    _json_print(
        {
            "status": "valid",
            "config": str(config.source_path),
            "config_sha256": config.config_hash,
        }
    )


def _extract_command(args: argparse.Namespace) -> None:
    config = load_aki_config(args.config)
    result = extract_creatinine_cohort(
        config,
        args.raw_dir,
        args.output_dir,
        chunksize=args.chunksize,
        overwrite=args.overwrite,
    )
    _json_print(result.__dict__)


def _prepare_command(args: argparse.Namespace) -> None:
    config = load_aki_config(args.config)
    destination = Path(args.output_dir).expanduser().resolve()
    manifest = destination / "run_manifest.json"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {manifest}")
    result = _run_prepare(
        config,
        args.measurements,
        args.admissions,
        args.output_dir,
        max_workers=args.workers,
        patient_chunk_size=args.patient_chunk_size,
    )
    summary = _pipeline_summary(result)
    write_run_manifest(
        manifest,
        config=config,
        repo_root=REPO_ROOT,
        artifacts={"prepared_directory": destination},
        extra={
            "stage": "prepare",
            "runtime": {
                "population_workers": args.workers,
                "patient_chunk_size": args.patient_chunk_size,
            },
            "summary": summary,
        },
    )
    _json_print({"manifest": manifest, **summary})


def _train_command(args: argparse.Namespace) -> None:
    config = load_aki_config(args.config)
    destination = Path(args.output_dir).expanduser().resolve()
    manifest = destination / "run_manifest.json"
    if manifest.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {manifest}")
    result = _run_train(config, args.prepared_dir, args.output_dir, tuple(args.models))
    summary = {
        "seed_metric_rows": len(result.seed_metrics),
        "aggregate_metric_rows": len(result.aggregate_metrics),
        "prediction_rows": len(result.predictions),
        "contrast_rows": len(result.contrasts),
        "aggregate_contrast_rows": len(result.aggregate_contrasts),
        "task_estimability": result.task_estimability.to_dict(orient="records"),
        "condition_estimability": result.condition_estimability.to_dict(orient="records"),
    }
    write_run_manifest(
        manifest,
        config=config,
        repo_root=REPO_ROOT,
        artifacts={
            "prepared_directory": Path(args.prepared_dir).expanduser().resolve(),
            "experiment_directory": destination,
        },
        extra={"stage": "train", "enabled_models": list(args.models), "summary": summary},
    )
    _json_print({"manifest": manifest, **summary})


def _run_command(args: argparse.Namespace) -> None:
    config = load_aki_config(args.config)
    destination = Path(args.output_dir).expanduser().resolve()
    manifest_path = destination / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite run manifest: {manifest_path}")

    cohort_dir = destination / "cohort"
    prepared_dir = destination / "prepared"
    experiment_dir = destination / "experiment"
    extraction = extract_creatinine_cohort(
        config,
        args.raw_dir,
        cohort_dir,
        chunksize=args.chunksize,
        overwrite=False,
    )
    prepared = _run_prepare(
        config,
        extraction.measurements_path,
        extraction.admission_audit_path,
        prepared_dir,
        max_workers=args.workers,
        patient_chunk_size=args.patient_chunk_size,
    )
    summary = _pipeline_summary(prepared)
    artifacts: dict[str, str | Path] = {
        "cohort_directory": cohort_dir,
        "prepared_directory": prepared_dir,
    }
    if prepared.matching.estimable:
        experiment = run_audit_experiments(
            {
                "binary": (
                    prepared.primary_matched_events,
                    prepared.primary_matched_labels,
                ),
                "multiclass": (prepared.secondary_events, prepared.secondary_labels),
            },
            config,
            output_dir=experiment_dir,
            enabled_models=tuple(args.models),
        )
        summary.update(
            {
                "seed_metric_rows": len(experiment.seed_metrics),
                "aggregate_metric_rows": len(experiment.aggregate_metrics),
                "aggregate_contrast_rows": len(experiment.aggregate_contrasts),
                "task_estimability": experiment.task_estimability.to_dict(orient="records"),
                "condition_estimability": experiment.condition_estimability.to_dict(
                    orient="records"
                ),
            }
        )
        artifacts["experiment_directory"] = experiment_dir
    write_run_manifest(
        manifest_path,
        config=config,
        repo_root=REPO_ROOT,
        artifacts=artifacts,
        extra={
            "raw_directory": str(Path(args.raw_dir).expanduser().resolve()),
            "enabled_models": list(args.models),
            "runtime": {
                "population_workers": args.workers,
                "patient_chunk_size": args.patient_chunk_size,
            },
            "summary": summary,
        },
    )
    _json_print({"manifest": manifest_path, **summary})
    _require_estimable(prepared)


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Completed strict protocol YAML")


def _add_models(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
        default=list(ALL_MODELS),
        help="Models to execute; the full requested matrix is the default",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.mimic.aki",
        description="Config-driven MIMIC-IV AKI trajectory audit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    _add_config(validate)
    validate.set_defaults(handler=_validate_command)

    extract = subparsers.add_parser("extract")
    _add_config(extract)
    extract.add_argument("--raw-dir", required=True)
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--chunksize", type=int, default=500_000)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(handler=_extract_command)

    prepare = subparsers.add_parser("prepare")
    _add_config(prepare)
    prepare.add_argument("--measurements", required=True)
    prepare.add_argument(
        "--admissions",
        required=True,
        help="Admission audit/table required for explicit discharge/death censor attribution",
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Runtime-only subject workers for deterministic episode construction",
    )
    prepare.add_argument(
        "--patient-chunk-size",
        type=_positive_int,
        default=128,
        help="Runtime-only subjects per ordered multiprocessing task",
    )
    prepare.set_defaults(handler=_prepare_command)

    train = subparsers.add_parser("train")
    _add_config(train)
    train.add_argument("--prepared-dir", required=True)
    train.add_argument("--output-dir", required=True)
    _add_models(train)
    train.set_defaults(handler=_train_command)

    run = subparsers.add_parser("run")
    _add_config(run)
    run.add_argument("--raw-dir", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--chunksize", type=int, default=500_000)
    run.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Runtime-only subject workers for deterministic episode construction",
    )
    run.add_argument(
        "--patient-chunk-size",
        type=_positive_int,
        default=128,
        help="Runtime-only subjects per ordered multiprocessing task",
    )
    _add_models(run)
    run.set_defaults(handler=_run_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
