#!/usr/bin/env python3
"""Evaluate a saved KIP adapter checkpoint without modifying its training run."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.experiments.kip_data_efficiency import model_provenance
from src.experiments.kip_efficiency_models import load_classifier
from src.experiments.kip_encoder_training import (
    DynamicPadding, TokenizedDataset, atomic_json, metric_values, read_split,
    restore_trainable, seed_all, sha256, validation_threshold,
)
from src.experiments.kip_llama_training import evaluate

EXPECTED_SOURCE_ROWS = {"train": 400000, "val": 50000, "test": 50000}
COMPLETION_METHOD = "user_requested_saved_checkpoint_evaluation"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-source-root", type=Path, default=ROOT,
                        help="root of preserved original source files recorded in the training manifest")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.eval_batch_size < 1 or args.threads < 1:
        parser.error("eval-batch-size and threads must be positive")
    return args


def verify_run(run_dir: Path, checkpoint_path: Path, training_source_root: Path = ROOT):
    """Verify original provenance and immutable row IDs before model loading."""
    manifest_path = run_dir / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    config = manifest["config"]
    if config["is_smoke"] or config["tag"] != "kip_m4" or config["mode"] != "ordered":
        raise ValueError("only production ordered KIP M4 runs can be evaluated")
    if config["arm"] == "llama8b_full":
        raise ValueError("full-tuning runs do not persist a complete selected checkpoint")
    if not math.isfinite(config["fraction"]) or not 0 < config["fraction"] <= 1:
        raise ValueError("invalid source training fraction")
    if config["selection"] != "minimum_validation_loss" or config["threshold_selection"] != "validation_f1":
        raise ValueError("unexpected source checkpoint/threshold selection protocol")
    indices, checked_paths = {}, {manifest_path: manifest_hash}
    training_source_root = training_source_root.resolve()
    for name, digest in manifest["source_sha256"].items():
        path = (training_source_root / name).resolve()
        if not path.is_relative_to(training_source_root) or sha256(path) != digest:
            raise ValueError(f"source implementation changed: {name}")
        checked_paths[path] = digest
    # Training orchestration may be amended after suspension, but inference
    # must still execute the exact original architecture, tokenization and loss.
    for name in ("kip_efficiency_models.py", "kip_encoder_training.py", "kip_llama_training.py"):
        relative = f"src/experiments/{name}"
        if relative not in manifest["source_sha256"] or sha256(ROOT / relative) != manifest["source_sha256"][relative]:
            raise ValueError(f"runtime inference implementation differs from original training source: {relative}")
    for split, source_rows in EXPECTED_SOURCE_ROWS.items():
        provenance = config["data"][split]
        if provenance["rows"] != source_rows:
            raise ValueError(f"{split}: requires the full fixed source split")
        for kind in ("features", "labels"):
            path = Path(provenance[kind]["path"])
            if sha256(path) != provenance[kind]["sha256"]:
                raise ValueError(f"{split}: {kind} data hash differs from source manifest")
        path = run_dir / f"{split}_indices.npy"
        if sha256(path) != config["index_sha256"][split]:
            raise ValueError(f"{split}: selected row index hash differs from source manifest")
        index = np.load(path, allow_pickle=False)
        count = int(source_rows * config["fraction"]) if split == "train" else source_rows
        if (index.ndim != 1 or index.dtype.kind not in "iu" or len(index) != count
                or config["rows"][split] != count or np.any(index < 0) or np.any(index >= source_rows)
                or np.any(index[1:] <= index[:-1])):
            raise ValueError(f"{split}: invalid selected row indices or row count")
        if split != "train" and not np.array_equal(index, np.arange(source_rows)):
            raise ValueError(f"{split}: evaluation must use every fixed held-out row")
        indices[split] = index
        checked_paths[path] = config["index_sha256"][split]

    checkpoint_hash = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError("checkpoint changed while it was being read")
    epoch = checkpoint.get("epoch")
    if checkpoint.get("arm") != config["arm"] or type(epoch) is not int or epoch < 1:
        raise ValueError("checkpoint arm/epoch does not match the source run")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain trainable weights")
    if any(not torch.is_tensor(tensor) or tensor.dtype != torch.float32
           or not torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("checkpoint trainables must be finite FP32 tensors")
    expected_names = manifest.get("model", {}).get("trainable_parameter_names")
    if expected_names is not None and set(expected_names) != set(state):
        raise ValueError("checkpoint trainable names differ from source model metadata")
    history_path = run_dir / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text().splitlines() if line.strip()] if history_path.exists() else []
    if history:
        epochs = [record["epoch"] for record in history]
        if epochs != list(range(1, max(epochs) + 1)) or epoch not in epochs:
            raise ValueError("checkpoint epoch is absent from the contiguous completed epoch history")
        selected = min(history, key=lambda record: record["val_loss"])
        if selected["epoch"] != epoch:
            raise ValueError("checkpoint is not the minimum-validation-loss completed epoch")
        checked_paths[history_path] = sha256(history_path)
    checked_paths[checkpoint_path] = checkpoint_hash
    return manifest, indices, checkpoint, history, checked_paths


def run_evaluation(args):
    started = time.monotonic()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    checkpoint_path = (args.checkpoint or run_dir / "best_trainable.pt").resolve()
    if output == run_dir or output.is_relative_to(run_dir) or run_dir.is_relative_to(output):
        raise ValueError("evaluation output must be outside the original training run and its ancestors")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("evaluation requires a fresh empty output directory")

    def progress(status, **details):
        atomic_json({"status": status, "pid": os.getpid(), "updated_at_unix": time.time(),
                     "wall_seconds": time.monotonic() - started, **details}, output / "status.json")
        print(json.dumps({"event": status, **details}, allow_nan=False), flush=True)

    progress("verifying_source")
    manifest, indices, checkpoint, history, checked_paths = verify_run(run_dir, checkpoint_path, args.training_source_root)
    config = manifest["config"]
    source_model_path = Path(config["model_source"]["snapshot"])
    if model_provenance(source_model_path) != config["model_source"]:
        raise ValueError("pretrained model identity differs from source run")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    seed_all(config["seed"])
    progress("loading_model", arm=config["arm"], best_epoch=checkpoint["epoch"])
    model, tokenizer, runtime = load_classifier(config["arm"], source_model_path, gradient_checkpointing=False)
    restore_trainable(model, checkpoint["state_dict"])
    selected_epoch = checkpoint["epoch"]
    del checkpoint
    model.eval()
    base = {name: config[name] for name in ("arm", "fraction", "seed", "tag", "mode", "is_smoke")}
    reference = {"path": str(checkpoint_path), "sha256": checked_paths[checkpoint_path], "epoch": selected_epoch}
    evaluation_manifest = {
        "schema_version": 1, "completion_method": COMPLETION_METHOD, "source_run": str(run_dir),
        "source_manifest_sha256": checked_paths[run_dir / "manifest.json"],
        "reference_checkpoint": reference, "source_file_sha256": {str(p): digest for p, digest in checked_paths.items()},
        "training_source_root": str(args.training_source_root.resolve()),
        "original_training_source_sha256": manifest["source_sha256"],
        "runtime_source_sha256": {relative: sha256(ROOT / relative) for relative in manifest["source_sha256"]},
        "evaluation_source_sha256": sha256(Path(__file__)), "created_at_unix": time.time(),
        "eval_batch_size": args.eval_batch_size, "threads": args.threads, "model": runtime,
        "source_config": config, "test_informed_stopping": True,
        "stopping_rule": "user requested stopping this arm and skipping larger fractions iff full fixed test AUC equals 1.0",
    }
    atomic_json(evaluation_manifest, output / "manifest.json")
    atomic_json(manifest, output / "source_manifest.json")
    results = {}
    for split in ("val", "test"):
        progress("evaluating", split=split, rows=len(indices[split]), best_epoch=selected_epoch)
        source = config["data"][split]
        features_path, labels_path = Path(source["features"]["path"]), Path(source["labels"]["path"])
        if features_path.parent != labels_path.parent:
            raise ValueError("features and labels must share the original data directory")
        sequences, labels, provenance = read_split(features_path.parent, config["tag"], split)
        if provenance != source:
            raise ValueError(f"{split}: loaded data differs from verified source provenance")
        rows = indices[split]
        dataset = TokenizedDataset([sequences[i] for i in rows], labels[rows], tokenizer, config["max_length"])
        loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=0,
                            pin_memory=True, collate_fn=DynamicPadding(tokenizer.pad_token_id, config["max_length"]))
        results[split] = evaluate(model, loader, torch.device("cuda"), "bf16")
        if len(results[split][1]) != EXPECTED_SOURCE_ROWS[split] or not np.array_equal(results[split][2], labels[rows]):
            raise ValueError(f"{split}: evaluated labels/count differ from the fixed held-out rows")
        np.save(output / f"{split}_indices.npy", rows, allow_pickle=False)
        if split == "val":
            threshold = validation_threshold(results[split][2], results[split][1])
        loss, probabilities, actual_labels = results[split]
        pd.DataFrame({"row_index": rows, "label": actual_labels, "probability": probabilities,
                      "prediction": (probabilities >= threshold).astype(int),
                      "prediction_at_0_5": (probabilities >= 0.5).astype(int)}).to_csv(
                          output / f"{split}_predictions.csv", index=False)
        progress("split_complete", split=split, loss=loss,
                 auc=metric_values(actual_labels, probabilities, threshold)["auc"])

    for path, digest in checked_paths.items():
        if sha256(path) != digest:
            raise ValueError(f"source artifact changed during evaluation: {path}")
    val_loss, val_probabilities, val_labels = results["val"]
    test_loss, test_probabilities, test_labels = results["test"]
    test_metrics = metric_values(test_labels, test_probabilities, threshold)
    metrics = {**base, "status": "completed", "schema_version": 1,
        "completion_method": COMPLETION_METHOD, "rows": config["rows"], "best_epoch": selected_epoch,
        "epochs_completed": max(record["epoch"] for record in history) if history else selected_epoch,
        "early_stopped": False, "training_run_completed": False, "selected_by": "validation_loss",
        "threshold_selected_on": "validation", "test_informed_stopping": True,
        "stopping_rule": {"metric": "full_fixed_test_auc", "target": 1.0, "comparison": "exact",
                          "met": test_metrics["auc"] == 1.0,
                          "authorized_action_if_met": "stop_current_arm_and_skip_larger_training_fractions"},
        "training_seconds": sum(record.get("epoch_seconds", 0) for record in history),
        "wall_seconds": time.monotonic() - started,
        "validation": {"loss": val_loss, **metric_values(val_labels, val_probabilities, threshold)},
        "validation_at_0_5": metric_values(val_labels, val_probabilities, 0.5),
        "test": {"loss": test_loss, **test_metrics}, "test_at_0_5": metric_values(test_labels, test_probabilities, 0.5),
        "selected_weights_saved": True, "selected_weights_path": str(checkpoint_path),
        "reference_checkpoint": reference, "source_run": str(run_dir),
        "source_manifest_sha256": checked_paths[run_dir / "manifest.json"],
        "checkpoint_policy": "existing selected adapter/head checkpoint referenced by path and SHA256; original run untouched"}
    metrics["artifact_sha256"] = {str(path.relative_to(output)): sha256(path) for path in sorted(output.rglob("*"))
                                   if path.is_file() and path.name != "status.json"}
    atomic_json(metrics, output / "metrics.json")
    progress("completed", **base, best_epoch=selected_epoch, test_auc=test_metrics["auc"],
             stopping_rule_met=metrics["stopping_rule"]["met"])
    return metrics


def main(argv=None):
    args = parse_args(argv)
    try:
        run_evaluation(args)
    except Exception as error:
        if args.output_dir.is_dir() and not isinstance(error, FileExistsError):
            # Never overwrite the source run's status on an invalid destination.
            if not args.output_dir.resolve().is_relative_to(args.run_dir.resolve()) and not args.run_dir.resolve().is_relative_to(args.output_dir.resolve()):
                atomic_json({"status": "failed", "pid": os.getpid(), "updated_at_unix": time.time(),
                             "error_type": type(error).__name__, "error": str(error)}, args.output_dir / "status.json")
        raise


if __name__ == "__main__":
    main()
