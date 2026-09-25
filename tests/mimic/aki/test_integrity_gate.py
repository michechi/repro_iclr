from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.mimic.aki.config import load_aki_config
from src.mimic.aki.experiment import (
    _aggregate,
    _aggregate_contrasts,
    _contrasts,
    _persist_experiment,
    _record_run,
    ExperimentResult,
)
from src.mimic.aki.reporting import write_run_manifest


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "mimic_aki_h200_smd015_amendment_v3.yaml"
GATE_PATH = REPO_ROOT / "scripts" / "mimic_aki_integrity_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("mimic_aki_integrity_gate", GATE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _truth(task: str) -> pd.DataFrame:
    labels = (
        ["transient", "persistent"] * 2
        if task == "binary"
        else ["transient", "persistent", "relapsing"] * 2
    )
    offset = 1_000 if task == "binary" else 2_000
    return pd.DataFrame(
        {
            "episode_id": np.arange(offset, offset + len(labels)),
            "subject_id": np.arange(offset + 10_000, offset + 10_000 + len(labels)),
            "hadm_id": np.arange(offset + 20_000, offset + 20_000 + len(labels)),
            "label": labels,
            "split": "test",
        }
    )


def _probabilities(labels: pd.Series, class_labels: tuple[str, ...], oracle: bool) -> np.ndarray:
    if oracle:
        matrix = np.zeros((len(labels), len(class_labels)), dtype=float)
        positions = {label: index for index, label in enumerate(class_labels)}
        for row, label in enumerate(labels):
            matrix[row, positions[label]] = 1.0
        return matrix
    correct = 0.7
    other = (1.0 - correct) / (len(class_labels) - 1)
    matrix = np.full((len(labels), len(class_labels)), other, dtype=float)
    positions = {label: index for index, label in enumerate(class_labels)}
    for row, label in enumerate(labels):
        matrix[row, positions[label]] = correct
    return matrix


def _synthetic_result(config) -> ExperimentResult:
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    seed_pairs = list(
        zip(config.models["seeds"], config.representations["permutation_seeds"])
    )
    logistic_features = (
        "invariant_summary",
        "count_only",
        "first_value",
        "last_value",
        "first_to_last_change",
        "linear_slope",
        "position_combined",
    )
    sequence_conditions = (
        ("ordered_train_ordered_test", "ordered"),
        ("ordered_train_shuffled_test", "shuffled"),
        ("ordered_train_reversed_test", "reversed"),
        ("shuffled_train_shuffled_test", "shuffled"),
    )
    for task in ("binary", "multiclass"):
        truth = _truth(task)
        class_labels = tuple(config.metrics[task]["labels"])
        probabilities = _probabilities(truth["label"], class_labels, False)
        for model_seed, permutation_seed in seed_pairs:
            for feature_set in logistic_features:
                _record_run(
                    metric_rows,
                    prediction_rows,
                    task=task,
                    model="logistic_regression",
                    condition="trained_and_tested_same_representation",
                    feature_set=feature_set,
                    model_seed=model_seed,
                    permutation_seed=permutation_seed,
                    test_labels=truth,
                    probabilities=probabilities,
                    class_labels=class_labels,
                    config=config,
                )
            _record_run(
                metric_rows,
                prediction_rows,
                task=task,
                model="xgboost",
                condition="trained_and_tested_same_representation",
                feature_set="invariant_summary",
                model_seed=model_seed,
                permutation_seed=permutation_seed,
                test_labels=truth,
                probabilities=probabilities,
                class_labels=class_labels,
                config=config,
            )
            for model in ("lstm", "transformer"):
                for condition, feature_set in sequence_conditions:
                    _record_run(
                        metric_rows,
                        prediction_rows,
                        task=task,
                        model=model,
                        condition=condition,
                        feature_set=feature_set,
                        model_seed=model_seed,
                        permutation_seed=permutation_seed,
                        test_labels=truth,
                        probabilities=probabilities,
                        class_labels=class_labels,
                        config=config,
                    )
        _record_run(
            metric_rows,
            prediction_rows,
            task=task,
            model="state_machine_oracle",
            condition="oracle_ceiling",
            feature_set="configured_trajectory_state_machine",
            model_seed=-1,
            permutation_seed=-1,
            test_labels=truth,
            probabilities=_probabilities(truth["label"], class_labels, True),
            class_labels=class_labels,
            config=config,
        )

    seed_metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    core_contrasts = _contrasts(seed_metrics)
    contrasts = core_contrasts.assign(
        bootstrap_estimate=core_contrasts["ordered_minus_control"],
        confidence_lower=core_contrasts["ordered_minus_control"] - 0.01,
        confidence_upper=core_contrasts["ordered_minus_control"] + 0.01,
        confidence_level=float(config.metrics["bootstrap"]["confidence_level"]),
        bootstrap_interval_method=str(config.metrics["bootstrap"]["interval_method"]),
        bootstrap_resamples_requested=int(config.metrics["bootstrap"]["n_resamples"]),
        bootstrap_resamples_valid=int(config.metrics["bootstrap"]["n_resamples"]),
        bootstrap_clustered_by_subject=True,
    )
    task_estimability = pd.DataFrame(
        [
            {"task": task, "estimable": True, "non_estimable_reasons": []}
            for task in ("binary", "multiclass")
        ]
    )
    condition_estimability = pd.DataFrame(
        [
            {
                "task": task,
                "model": "state_machine_oracle",
                "condition": "oracle_ceiling",
                "estimable": True,
                "reason": "",
            }
            for task in ("binary", "multiclass")
        ]
    )
    return ExperimentResult(
        seed_metrics=seed_metrics,
        aggregate_metrics=_aggregate(seed_metrics, config),
        predictions=predictions,
        contrasts=contrasts,
        aggregate_contrasts=_aggregate_contrasts(contrasts, config),
        task_estimability=task_estimability,
        condition_estimability=condition_estimability,
    )


def _make_private(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        path.chmod(0o700 if path.is_dir() else 0o600)


def test_complete_posttraining_gate_accepts_consistent_private_outputs(tmp_path: Path) -> None:
    gate = _load_gate_module()
    config = load_aki_config(CONFIG_PATH)
    run_root = tmp_path / "run"
    protocol = run_root / "protocol"
    prepared = run_root / "prepared"
    experiment = run_root / "experiment"
    logs = run_root / "logs"
    for directory in (protocol, prepared, logs):
        directory.mkdir(parents=True, exist_ok=True)

    frozen_config = protocol / CONFIG_PATH.name
    frozen_config.write_bytes(CONFIG_PATH.read_bytes())
    for task, name in (
        ("binary", "primary_matched_labels.parquet"),
        ("multiclass", "secondary_labels.parquet"),
    ):
        _truth(task).to_parquet(prepared / name, index=False)
    pd.DataFrame([{"config_hash": config.config_hash, "schema": 1}]).to_csv(
        prepared / "protocol_metadata.csv", index=False
    )
    (prepared / "run_manifest.json").write_text(
        json.dumps({"stage": "prepare", "config_sha256": config.config_hash}) + "\n",
        encoding="utf-8",
    )

    result = _synthetic_result(config)
    _persist_experiment(result, experiment)
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
        experiment / "run_manifest.json",
        config=config,
        repo_root=REPO_ROOT,
        artifacts={
            "prepared_directory": prepared,
            "experiment_directory": experiment,
        },
        extra={
            "stage": "train",
            "enabled_models": [
                "logistic_regression",
                "xgboost",
                "lstm",
                "transformer",
            ],
            "summary": summary,
        },
    )

    marker_lines = []
    for model in ("lstm", "transformer"):
        for seed in config.models["seeds"]:
            for _ in range(4):
                marker_lines.append(
                    f"AKI_SEQUENCE_DEVICE model={model} seed={seed} "
                    "device=cuda:0 accelerator=NVIDIA H200 NVL"
                )
                marker_lines.append(
                    f"AKI_SEQUENCE_COMPLETE model={model} seed={seed} "
                    "device=cuda:0 accelerator=NVIDIA H200 NVL best_epoch=1"
                )
    train_log = logs / "train.console.log"
    train_log.write_text("\n".join(marker_lines) + "\n", encoding="utf-8")
    telemetry = logs / "train.gpu-telemetry.csv"
    telemetry.write_text(
        "uuid,name,utilization_gpu_percent,memory_used_mib\n"
        "GPU-synthetic,NVIDIA H200 NVL,10,100\n",
        encoding="utf-8",
    )
    _make_private(run_root)

    report, status = gate.run_posttraining_gate(
        config,
        frozen_config,
        prepared,
        experiment,
        train_log,
        telemetry,
    )

    assert status == 0, report["failures"]
    assert report["overall_gate"] == "PASS"
    assert report["prediction_integrity_ok"]
    assert report["sequence_h200_markers_ok"]
