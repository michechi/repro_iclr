"""Full Llama sequence classification on KIP, with one selected-model export.

Best weights live in CPU RAM. There is deliberately no epoch/optimizer checkpoint:
an interrupted unfinished run must restart in a fresh output directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.experiments.kip_encoder_training import (
    SEP, DynamicPadding, EarlyStopping, TokenizedDataset, atomic_json, autocast,
    metric_values, read_split, seed_all, sha256, subset_indices, validation_threshold,
)

CHECKPOINT_POLICY = "selected_model_only; interrupted unfinished run must restart"


def classifier_loss(model, batch, device, precision="bf16"):
    """Autocast the network only; compute binary outcome CE in FP32."""
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    labels = moved.pop("labels")
    with autocast(device, precision):
        logits = model(**moved).logits
    return F.cross_entropy(logits.float(), labels), logits.float()


@torch.no_grad()
def evaluate(model, loader, device, precision="bf16"):
    model.eval()
    losses, count, probabilities, labels = 0.0, 0, [], []
    for batch in loader:
        loss, logits = classifier_loss(model, batch, device, precision)
        losses += float(loss.item()) * len(batch["labels"])
        count += len(batch["labels"])
        probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy())
    probability = np.concatenate(probabilities)
    if not math.isfinite(losses) or not np.isfinite(probability).all():
        raise ValueError("non-finite validation/test output")
    return losses / count, probability, np.concatenate(labels)


def validate_token_ids(datasets, pad_token_id):
    # Llama pools at the last non-pad token. EOS-as-pad is safe only when EOS
    # never appears in the unpadded narrative, including its special tokens.
    for split, dataset in datasets.items():
        if any(pad_token_id in ids for ids in dataset.input_ids):
            raise ValueError(f"{split}: EOS/padding token occurs inside unpadded content")


def accumulation_samples(index, batches, batch_size, rows, accumulation):
    start = index // accumulation * accumulation
    stop = min(start + accumulation, batches)
    return stop, min(stop * batch_size, rows) - start * batch_size


def save_selected_model(model, tokenizer, output_dir):
    destination, staging = output_dir / "best_model", output_dir / ".best_model.staging"
    if destination.exists() or staging.exists():
        raise FileExistsError("selected-model destination/staging exists; refusing to overwrite")
    parameter_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    needed = math.ceil(parameter_bytes * 1.02) + 64 * 1024**2
    available = shutil.disk_usage(output_dir).free
    if available < needed:
        raise OSError(f"selected-model export requires {needed} free bytes, available {available}")
    model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(staging)
    staging.rename(destination)
    return destination


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tag", choices=["kip_m4", "kip_m6"], required=True)
    parser.add_argument("--mode", choices=["ordered", "shuffled_train"], required=True)
    parser.add_argument("--seed", type=int, default=9950)
    for name in ("ordered-dir", "shuffled-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    for name, default in (("batch-size", 16), ("eval-batch-size", 64),
                          ("gradient-accumulation-steps", 1), ("patience", 3),
                          ("max-length", 64), ("threads", 8), ("progress-steps", 100)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--optimizer-offload", choices=["none", "cpu"], default="none")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save-model", action="store_true")
    args = parser.parse_args(argv)
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 20)
    if args.no_save_model and not args.smoke:
        parser.error("--no-save-model is permitted only with --smoke")
    for name in ("batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs",
                 "patience", "max_length", "threads", "progress_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.warmup_ratio < 1:
        parser.error("learning rate must be positive and warmup ratio in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    return args


def train(args):
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    started = time.monotonic()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested BF16 autocast is unsupported")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"pretrained snapshot does not exist: {args.model_path}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, metrics_path = output_dir / "manifest.json", output_dir / "metrics.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None and not args.skip_completed:
        raise FileExistsError("run manifest exists; --skip-completed can verify a completed run")
    if previous is not None and not metrics_path.exists():
        raise FileExistsError("unfinished run must restart in a fresh output directory; no epoch resume exists")
    if previous is None and any(output_dir.iterdir()):
        raise FileExistsError("output directory is not empty and has no manifest")

    def progress(state, **details):
        atomic_json({"status": "completed" if state == "complete" else state, "state": state,
                     "pid": os.getpid(), "updated_at_unix": time.time(),
                     "wall_seconds_this_process": time.monotonic() - started,
                     "model_name": args.model_name, "tag": args.tag, "mode": args.mode,
                     "tuning": "full", "seed": args.seed, **details}, output_dir / "status.json")

    if previous is None:
        progress("loading_data")
    selected, provenance, counts = {}, {}, {}
    for split in ("train", "val", "test"):
        ordered, labels, ordered_meta = read_split(args.ordered_dir, args.tag, split)
        shuffled, shuffled_labels, shuffled_meta = read_split(args.shuffled_dir, args.tag, split)
        if not np.array_equal(labels, shuffled_labels):
            raise ValueError(f"{split}: ordered/shuffled labels or row counts differ")
        if any(sorted(left.split(SEP)) != sorted(right.split(SEP))
               for left, right in zip(ordered, shuffled)):
            raise ValueError(f"{split}: shuffled rows do not preserve corresponding event multisets")
        limit = (256 if split == "train" else 128) if args.smoke else None
        indices = subset_indices(labels, limit, args.seed)
        source_rows = ordered if args.mode == "ordered" else shuffled
        selected[split] = ([source_rows[i] for i in indices], labels[indices], indices)
        if split == "test" and args.mode == "ordered":
            selected["shuffled_test"] = ([shuffled[i] for i in indices], labels[indices], indices)
        provenance[split] = {"ordered": ordered_meta, "shuffled": shuffled_meta}
        counts[split] = len(indices)
    if not args.smoke and counts != {"train": 400000, "val": 50000, "test": 50000}:
        raise ValueError(f"production requires complete KIP 400000/50000/50000 splits, got {counts}")
    source = str(args.model_path.resolve())
    source_hashes = {path.name: sha256(path) for path in sorted(args.model_path.iterdir())
                     if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin", ".model"}}
    config = {"schema_version": 1, "model_name": args.model_name, "model_source": source,
              "source_file_sha256": source_hashes, "tag": args.tag, "mode": args.mode,
              "tuning": "full", "seed": args.seed, "batch_size": args.batch_size,
              "eval_batch_size": args.eval_batch_size,
              "gradient_accumulation_steps": args.gradient_accumulation_steps,
              "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
              "patience": args.patience, "warmup_ratio": args.warmup_ratio,
              "max_length": args.max_length, "precision": args.precision,
              "parameter_dtype": "float32", "optimizer_state_dtype": "float32",
              "loss": "binary_outcome_cross_entropy_float32", "use_cache": False,
              "padding_side": "right", "pad_token": "eos", "lora": None,
              "prompt": "Sequential events: " + " ".join(["<event>"] * 20),
              "is_smoke": args.smoke, "save_model": not args.no_save_model,
              "selected_rows": counts, "data": provenance,
              "selection": "minimum_validation_loss", "threshold_selection": "validation_f1",
              "checkpoint_policy": CHECKPOINT_POLICY}
    # Keep the original default manifest compatible with completed 1B runs.
    if args.optimizer_offload != "none":
        config["optimizer_offload"] = args.optimizer_offload
    if args.gradient_checkpointing:
        config["gradient_checkpointing"] = True
    if previous is not None:
        if previous["config"] != config:
            changed = sorted(key for key in set(config) | set(previous["config"])
                             if config.get(key) != previous["config"].get(key))
            raise ValueError(f"completed run configuration differs: {changed}")
        metrics = json.loads(metrics_path.read_text())
        for name, digest in metrics["artifact_sha256"].items():
            artifact = output_dir / name
            if not artifact.is_file() or sha256(artifact) != digest:
                raise ValueError(f"completed artifact missing or changed: {name}")
        if not metrics["artifact_sha256"] or metrics["is_smoke"] != args.smoke:
            raise ValueError("invalid completed run artifacts/metadata")
        return metrics
    load_kwargs = {"local_files_only": True}
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    progress("loading_model")
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    if tokenizer.eos_token_id is None:
        raise ValueError("Llama tokenizer must define EOS for padding")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2,
                                                               torch_dtype=torch.float32, **load_kwargs)
    if model.config.model_type != "llama":
        raise ValueError(f"expected Llama, got {model.config.model_type}")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.problem_type = "single_label_classification"
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.max_length > min(tokenizer.model_max_length, model.config.max_position_embeddings):
        raise ValueError("max length exceeds supported context")
    if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("full training requires every parameter trainable and FP32")
    model.to(device)
    progress("tokenizing")
    datasets = {split: TokenizedDataset(sequences, labels, tokenizer, args.max_length)
                for split, (sequences, labels, _) in selected.items()}
    validate_token_ids(datasets, tokenizer.pad_token_id)
    lengths = {split: {"min": data.min_tokens, "max": data.max_tokens} for split, data in datasets.items()}
    parameters = sum(p.numel() for p in model.parameters())
    manifest = {"config": config, "created_at_unix": time.time(), "token_lengths": lengths,
                "total_parameters": parameters, "trainable_parameters": parameters,
                "pretrained_commit": getattr(model.config, "_commit_hash", None),
                "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                                "device": str(device), "gpu": torch.cuda.get_device_name(device)
                                if device.type == "cuda" else None}}
    atomic_json(manifest, manifest_path)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(data, batch_size=args.batch_size if split == "train" else args.eval_batch_size,
                                shuffle=split == "train", generator=generator if split == "train" else None,
                                collate_fn=DynamicPadding(tokenizer.pad_token_id, args.max_length),
                                num_workers=0, pin_memory=device.type == "cuda")
               for split, data in datasets.items()}
    if args.optimizer_offload == "cpu":
        from src.experiments.kip_cpu_offload import CPUOffloadAdamW
        optimizer = CPUOffloadAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stopper, best_state, history, optimizer_steps = EarlyStopping(args.patience), None, [], 0
    training_started = time.monotonic()
    print(json.dumps({"event": "ready", "token_lengths": lengths, "trainable_parameters": parameters,
                      "config": {key: val for key, val in config.items()
                                 if key not in {"data", "source_file_sha256"}}}), flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = torch.zeros((), device=device, dtype=torch.float64), 0
        progress("training", epoch=epoch, epochs=args.epochs, batch=0, batches=len(loaders["train"]))
        for index, batch in enumerate(loaders["train"]):
            stop, group_samples = accumulation_samples(index, len(loaders["train"]), args.batch_size,
                                                       len(datasets["train"]), args.gradient_accumulation_steps)
            loss, _ = classifier_loss(model, batch, device, args.precision)
            batch_size = len(batch["labels"])
            (loss * batch_size / group_samples).backward()
            loss_sum += loss.detach().double() * batch_size
            seen += batch_size
            if index + 1 == stop:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if (index + 1) % args.progress_steps == 0 or index + 1 == len(loaders["train"]):
                average = float(loss_sum.item()) / seen
                if not math.isfinite(average):
                    raise ValueError(f"non-finite training loss at epoch={epoch}, batch={index + 1}")
                progress("training", epoch=epoch, epochs=args.epochs, batch=index + 1,
                         batches=len(loaders["train"]), train_loss=average,
                         examples_per_second=seen / max(time.monotonic() - epoch_started, 1e-9),
                         optimizer_steps=optimizer_steps, learning_rate=scheduler.get_last_lr()[0],
                         best_epoch=stopper.best_epoch)
        progress("validating", epoch=epoch, epochs=args.epochs)
        val_loss, probabilities, labels = evaluate(model, loaders["val"], device, args.precision)
        improved = stopper.update(val_loss, epoch)
        if improved:
            if best_state is None:
                best_state = {name: value.detach().to("cpu", copy=True)
                              for name, value in model.state_dict().items()}
            else:
                # Avoid retaining two full CPU checkpoints during an 8B update.
                for name, value in model.state_dict().items():
                    best_state[name].copy_(value.detach())
        record = {"epoch": epoch, "train_loss": float(loss_sum.item()) / seen,
                  "val_loss": val_loss, "val_auc": metric_values(labels, probabilities, 0.5)["auc"],
                  "best_epoch": stopper.best_epoch, "improved": improved, "bad_epochs": stopper.bad_epochs,
                  "epoch_seconds": time.monotonic() - epoch_started, "optimizer_steps": optimizer_steps,
                  "learning_rate": scheduler.get_last_lr()[0]}
        history.append(record)
        with (output_dir / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
        if stopper.stopped:
            break
    if best_state is None:
        raise RuntimeError("no validation-selected epoch exists")
    training_seconds = time.monotonic() - training_started
    model.load_state_dict(best_state, strict=True)
    del best_state, optimizer, scheduler
    progress("evaluating", best_epoch=stopper.best_epoch)
    val_loss, val_probabilities, val_labels = evaluate(model, loaders["val"], device, args.precision)
    threshold = validation_threshold(val_labels, val_probabilities)
    metrics = {"schema_version": 1, "model_name": args.model_name, "tag": args.tag, "mode": args.mode,
               "tuning": "full", "seed": args.seed, "is_smoke": args.smoke, "rows": counts,
               "best_epoch": stopper.best_epoch, "epochs_completed": history[-1]["epoch"],
               "early_stopped": stopper.stopped, "selected_by": "validation_loss",
               "threshold_selected_on": "validation", "checkpoint_policy": CHECKPOINT_POLICY,
               "validation": {"loss": val_loss, **metric_values(val_labels, val_probabilities, threshold)},
               "validation_at_0_5": metric_values(val_labels, val_probabilities, 0.5),
               "manifest": str(manifest_path), "training_seconds": training_seconds, "best_model": None}
    for split in ["test"] + (["shuffled_test"] if args.mode == "ordered" else []):
        loss, probabilities, labels = evaluate(model, loaders[split], device, args.precision)
        metrics[split] = {"loss": loss, **metric_values(labels, probabilities, threshold)}
        metrics[f"{split}_at_0_5"] = metric_values(labels, probabilities, 0.5)
        pd.DataFrame({"row_index": selected[split][2], "label": labels, "probability": probabilities,
                      "prediction": (probabilities >= threshold).astype(int),
                      "prediction_at_0_5": (probabilities >= 0.5).astype(int)}).to_csv(
                          output_dir / f"{split}_predictions.csv", index=False)
    if not args.no_save_model:
        progress("saving_best_model", best_epoch=stopper.best_epoch)
        metrics["best_model"] = str(save_selected_model(model, tokenizer, output_dir))
    artifacts = sorted(output_dir.glob("*_predictions.csv")) + sorted((output_dir / "best_model").glob("*"))
    metrics["artifact_sha256"] = {str(path.relative_to(output_dir)): sha256(path)
                                  for path in artifacts if path.is_file()}
    metrics["wall_seconds_this_process"] = time.monotonic() - started
    atomic_json(metrics, metrics_path)  # Completion marker is written after all artifacts.
    progress("complete", best_epoch=stopper.best_epoch, epochs_completed=history[-1]["epoch"],
             test_auc=metrics["test"]["auc"], is_smoke=args.smoke)
    print(json.dumps({"event": "complete", "metrics": metrics}, allow_nan=False), flush=True)
    return metrics


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except Exception as error:
        if args.output_dir.is_dir() and not (args.output_dir / "metrics.json").exists() and not isinstance(error, FileExistsError):
            atomic_json({"status": "failed", "state": "failed", "pid": os.getpid(),
                         "updated_at_unix": time.time(), "error_type": type(error).__name__,
                         "error": str(error)}, args.output_dir / "status.json")
        raise


if __name__ == "__main__":
    main()
