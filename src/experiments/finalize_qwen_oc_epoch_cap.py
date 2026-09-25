"""Evaluate the existing validation-selected OC checkpoint after a user epoch cap."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.analysis.kip_checkpoint_attention import load_checkpoint, read_json, write_json, digest
from src.experiments.kip_encoder_training import (
    DynamicPadding, TokenizedDataset, metric_values, seed_all, validation_threshold,
)
from src.experiments.kip_llama_training import evaluate


def finalize(run, record_path, device="cuda"):
    run, record_path = Path(run), Path(record_path)
    record = read_json(record_path)
    started = time.monotonic()
    if (run / "metrics.json").exists():
        raise FileExistsError("completed fit must not be overwritten")
    manifest = read_json(run / "manifest.json")
    cfg = manifest["config"]
    for name, expected in record["fit_file_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError("stopped fit artifact changed: " + name)
    history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
    cap = record["epoch_cap"]
    if [r["epoch"] for r in history] != list(range(1, cap + 1)):
        raise ValueError("exact completed epoch history required")
    best = min(history, key=lambda r: r["val_loss"])
    if best["epoch"] != record["best_epoch"]:
        raise ValueError("selected epoch differs from recorded validation minimum")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    seed_all(cfg["seed"])

    def progress(status, **detail):
        write_json(run / "status.json", dict(status=status, pid=os.getpid(),
            updated_at_unix=time.time(), user_epoch_cap=cap, best_epoch=best["epoch"], **detail))

    progress("loading_selected_checkpoint")
    job = dict(kind="efficiency", run=str(run), checkpoint=str(run / "best_trainable.pt"),
               arm=cfg["arm"], model_source=cfg["model_source"]["snapshot"])
    model, tokenizer, metadata = load_checkpoint(job, device=device)
    if metadata["selected_epoch"] != best["epoch"]:
        raise ValueError("checkpoint epoch does not match the validation-selected epoch")
    results, selected = {}, {}
    for split in ("val", "test"):
        source = cfg["data"][split]
        for key in ("features", "labels"):
            if digest(source[key]["path"]) != source[key]["sha256"]:
                raise ValueError("evaluation data changed")
        indices_path = run / f"{split}_indices.npy"
        if digest(indices_path) != cfg["index_sha256"][split]:
            raise ValueError("evaluation row order changed")
        indices = np.load(indices_path, allow_pickle=False)
        sequences = pd.read_csv(source["features"]["path"], dtype=str, keep_default_na=False)["Sequences"].tolist()
        labels = pd.read_csv(source["labels"]["path"])["Outcome"].to_numpy(dtype=np.int64)
        if len(indices) != cfg["rows"][split] or not np.array_equal(indices, np.arange(len(labels))):
            raise ValueError("requires the complete original evaluation split")
        data = TokenizedDataset([sequences[i] for i in indices], labels[indices], tokenizer, cfg["max_length"])
        loader = DataLoader(data, batch_size=cfg["eval_batch_size"], shuffle=False,
            collate_fn=DynamicPadding(tokenizer.pad_token_id, cfg["max_length"]), num_workers=0,
            pin_memory=device == "cuda")
        progress("evaluating", split=split)
        results[split] = evaluate(model, loader, torch.device(device))
        selected[split] = indices
        if split == "val":
            loss, probabilities, y = results[split]
            auc_error = abs(metric_values(y, probabilities, .5)["auc"] - best["val_auc"])
            loss_error = abs(loss - best["val_loss"])
            if auc_error > 1e-6 or loss_error > 1e-5:
                raise ValueError(f"selected validation result did not reproduce: loss={loss_error}, AUC={auc_error}")
    threshold = validation_threshold(results["val"][2], results["val"][1])
    for split, (_, probabilities, labels) in results.items():
        pd.DataFrame(dict(row_index=selected[split], label=labels, probability=probabilities,
            prediction=(probabilities >= threshold).astype(int),
            prediction_at_0_5=(probabilities >= .5).astype(int))).to_csv(run / f"{split}_predictions.csv", index=False)
    tokenizer.save_pretrained(run / "tokenizer")
    write_json(run / "user_epoch_cap.json", record)
    metrics = {k:cfg[k] for k in ("arm", "fraction", "seed", "tag", "mode", "is_smoke")}
    metrics.update(status="completed", schema_version=1, rows=cfg["rows"], best_epoch=best["epoch"],
        epochs_completed=cap, early_stopped=True, selected_by="validation_loss",
        stop_reason="user_epoch_cap", stopping_rule="user_requested_epoch_cap", user_epoch_cap=cap,
        original_max_epochs=cfg["epochs"], learning_rate_schedule="original schedule preserved through epoch cap",
        test_informed_stopping=False, threshold_selected_on="validation",
        training_seconds=sum(r["epoch_seconds"] for r in history), finalization_seconds=time.monotonic()-started,
        wall_seconds=record["fit_status_at_stop"].get("wall_seconds", 0)+time.monotonic()-started,
        selected_weights_saved=True, checkpoint_policy=cfg["checkpoint_policy"],
        validation_reproduction=dict(loss_absolute_error=loss_error, auc_absolute_error=auc_error),
        user_stop_record_sha256=digest(record_path))
    for split, (loss, probabilities, labels) in results.items():
        name = "validation" if split == "val" else "test"
        metrics[name] = dict(loss=loss, **metric_values(labels, probabilities, threshold))
        metrics[name + "_at_0_5"] = metric_values(labels, probabilities, .5)
    metrics["artifact_sha256"] = {str(p.relative_to(run)):digest(p) for p in run.rglob("*")
        if p.is_file() and p.name not in ("status.json", "metrics.json")}
    write_json(run / "metrics.json", metrics)
    progress("completed", test_auc=metrics["test"]["auc"], epochs_completed=cap)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.run, args.record)
