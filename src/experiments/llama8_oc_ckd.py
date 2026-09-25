"""Fresh Llama-3.1-8B LoRA classifiers on fixed OC and paired CKD splits.

The BF16 pretrained backbone is frozen; FP32 LoRA weights and the last-token
classification head are selected using validation loss. Test probabilities are
computed once after selection. Smoke test samples exercise checkpoint audits;
their scores never select recipes or stop production training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.experiments.kip_data_efficiency import model_provenance
from src.experiments.kip_encoder_training import (
    SEP, DynamicPadding, EarlyStopping, atomic_checkpoint, atomic_json,
    metric_values, restore_trainable, seed_all, sequence_prompt, sha256,
    subset_indices, trainable_state, validation_threshold,
)
from src.experiments.kip_llama_training import accumulation_samples, classifier_loss, evaluate

ROOT = Path(__file__).resolve().parents[2]
ARM = "llama8b_lora"
MODEL_NAME = "meta-llama/Llama-3.1-8B"


def prompt(sequence, task):
    if task == "oc":
        return sequence_prompt(sequence)
    if task != "ckd":
        raise ValueError("task must be oc or ckd")
    events = sequence.split(SEP)
    if not events or any(re.fullmatch(r"CCS_[0-9]+|OTHER", event) is None for event in events):
        raise ValueError("CKD requires nonempty CCS_<number>/OTHER events separated by U+001F")
    return "Sequential events: " + " ".join(events) + "\nOutcome (0 or 1):"


def read_data(plan, task, condition, split):
    source = plan["datasets"][task][condition][split]
    for key in ("features", "labels"):
        if sha256(Path(source[key])) != source[key + "_sha256"]:
            raise ValueError("frozen input changed: " + source[key])
    features = pd.read_csv(source["features"], dtype=str, keep_default_na=False)
    outcomes = pd.read_csv(source["labels"])
    if "Sequences" not in features or "Outcome" not in outcomes:
        raise ValueError("expected Sequences and Outcome columns")
    sequences, labels = features["Sequences"].tolist(), outcomes["Outcome"].to_numpy()
    if (len(sequences) != source["rows"] or len(labels) != len(sequences)
            or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2):
        raise ValueError("invalid split rows or binary labels")
    for sequence in sequences:
        prompt(sequence, task)
    return sequences, labels.astype(np.int64)


class TextDataset:
    """Tokenize complete narratives; reject overlength rows instead of truncating."""

    def __init__(self, sequences, labels, tokenizer, task, max_length):
        if not sequences or len(sequences) != len(labels):
            raise ValueError("nonempty matching sequences and labels required")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_ids = []
        self.max_length = max_length
        for start in range(0, len(sequences), 128 if task == "ckd" else 2048):
            count = 128 if task == "ckd" else 2048
            texts = [prompt(sequence, task) for sequence in sequences[start:start + count]]
            encoded = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True,
                                return_attention_mask=False, return_token_type_ids=False)
            self.input_ids.extend(encoded["input_ids"])
        if len(self.input_ids) != len(self.labels):
            raise ValueError("tokenizer changed the row count")
        self.lengths = np.asarray(list(map(len, self.input_ids)), dtype=np.int64)
        if self.lengths.min() < 1 or self.lengths.max() > max_length:
            raise ValueError(f"token lengths {self.lengths.min()}..{self.lengths.max()} exceed "
                             f"allowed 1..{max_length}; truncation is forbidden")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], int(self.labels[index])

    def subset(self, indices):
        result = object.__new__(TextDataset)
        result.labels = self.labels[indices]
        result.input_ids = [self.input_ids[index] for index in indices]
        result.lengths = self.lengths[indices]
        result.max_length = self.max_length
        return result

    @property
    def audit(self):
        return {"rows": len(self), "min": int(self.lengths.min()), "max": int(self.lengths.max()),
                "min_full_tokens": int(self.lengths.min()), "max_full_tokens": int(self.lengths.max()),
                "max_input_tokens": int(self.lengths.max()), "median": float(np.median(self.lengths)),
                "p95": float(np.percentile(self.lengths, 95)), "tokens": int(self.lengths.sum()),
                "truncated_rows": 0, "truncated_fraction": 0., "context_policy": "no truncation"}


def smoke_indices(labels, lengths, limit, seed, protected_indices=()):
    """Keep stratified class counts and include each condition's longest sample."""
    chosen = set(map(int, subset_indices(labels, min(limit, len(labels)), seed)))
    protected = set(map(int, protected_indices)) | {int(np.argmax(lengths))}
    for index in sorted(protected):
        if index in chosen:
            continue
        candidates = [other for other in sorted(chosen)
                      if other not in protected and labels[other] == labels[index]]
        if not candidates:
            raise ValueError("smoke cannot preserve class counts while including longest rows")
        chosen.remove(candidates[0])
        chosen.add(index)
    return np.asarray(sorted(chosen), dtype=np.int64)


def split_indices(labels, split, recipe, smoke=False):
    limit = (64 if split == "train" else 32) if smoke else recipe.get("train_rows") if split == "train" else None
    if not smoke and limit is not None and not 0 < limit <= len(labels):
        raise ValueError("requested training rows exceed the available split or are nonpositive")
    return subset_indices(labels, limit, recipe["seed"])


def verify_reference_indices(indices, recipe):
    reference = recipe.get("reference_train_indices")
    if reference is None:
        return
    path = Path(reference["path"])
    if sha256(path) != reference["sha256"]:
        raise ValueError("reference training indices changed")
    if not np.array_equal(indices, np.load(path, allow_pickle=False)):
        raise ValueError("training sample differs from the fixed encoder OC sample")


def classification_metrics(labels, probabilities, threshold):
    values = metric_values(labels, probabilities, threshold)
    ap = float(average_precision_score(labels, probabilities))
    return {**values, "auroc": values["auc"], "average_precision": ap, "auprc": ap,
            "auprc_definition": "average_precision", "rows": len(labels),
            "positive_rows": int(np.sum(labels)), "prevalence": float(np.mean(labels))}


def save_trainable_checkpoint(model, output, epoch):
    state = trainable_state(model)
    needed = math.ceil(sum(value.numel() * value.element_size() for value in state.values()) * 1.05)
    needed += 64 * 1024**2
    if shutil.disk_usage(output).free < needed:
        raise OSError(f"selected trainable checkpoint requires {needed} free bytes")
    atomic_checkpoint({"arm": ARM, "epoch": epoch, "state_dict": state}, output / "best_trainable.pt")
    return state


def actual_recipe(plan, job, args):
    recipe = dict(plan["recipes"][job["task"]])
    recipe.setdefault("max_length", 16384 if job["task"] == "ckd" else 64)
    recipe.setdefault("weight_decay", .01)
    recipe.setdefault("warmup_ratio", .06)
    if job["smoke"]:
        recipe.update(epochs=1, min_epochs=1)
    if args.batch_size is not None:
        recipe["batch_size"] = args.batch_size
        recipe["eval_batch_size"] = min(recipe["eval_batch_size"], args.batch_size)
    if args.eval_batch_size is not None:
        recipe["eval_batch_size"] = args.eval_batch_size
    for key in ("batch_size", "eval_batch_size", "effective_batch_size", "epochs", "min_epochs", "patience", "max_length"):
        if not isinstance(recipe[key], int) or recipe[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if recipe["effective_batch_size"] % recipe["batch_size"]:
        raise ValueError("microbatch must divide the fixed effective batch size")
    if recipe["min_epochs"] > recipe["epochs"]:
        raise ValueError("minimum epochs exceeds maximum epochs")
    if not math.isfinite(recipe["lr"]) or recipe["lr"] <= 0:
        raise ValueError("learning rate must be finite and positive")
    if recipe["weight_decay"] != .01 or recipe["warmup_ratio"] != .06:
        raise ValueError("recipe must retain weight decay 0.01 and warmup ratio 0.06")
    recipe["gradient_accumulation_steps"] = recipe["effective_batch_size"] // recipe["batch_size"]
    return recipe


def train(args):
    from transformers import AutoTokenizer

    plan = json.loads((args.run_root / "launch_plan.json").read_text())
    job = next(job for job in plan["jobs"] if job["key"] == args.job)
    if job["kind"] != "fit" or job["model"] != ARM:
        raise ValueError("expected a llama8b_lora fit job")
    if job["task"] not in {"oc", "ckd"} or job["condition"] not in {"ordered", "shuffled"}:
        raise ValueError("invalid task or condition")
    recipe = actual_recipe(plan, job, args)
    output = (args.output_dir or Path(job["output_dir"])).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh fit requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def progress(status, **detail):
        record = dict(status=status, job=args.job, pid=os.getpid(), updated_at_unix=time.time(),
                      wall_seconds=time.monotonic() - started, **detail)
        atomic_json(record, output / "status.json")
        print(json.dumps(record, allow_nan=False), flush=True)

    for relative, expected in plan["source_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("frozen runtime source changed: " + relative)
    progress("loading_data")
    record = plan["models"][ARM]
    model_path = Path(record["path"])
    if record.get("name", MODEL_NAME) != MODEL_NAME:
        raise ValueError("model must be the pinned Llama-3.1-8B base")
    for filename, expected in record["files_sha256"].items():
        if sha256(model_path / filename) != expected:
            raise ValueError("pretrained model changed: " + filename)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                              trust_remote_code=False, padding_side="right")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer needs an EOS token for padding")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    datasets, selected, source_tokens, provenance = {}, {}, {}, {}
    for split in ("train", "val", "test"):
        progress("tokenizing", split=split)
        sequences, labels = read_data(plan, job["task"], job["condition"], split)
        dataset = TextDataset(sequences, labels, tokenizer, job["task"], recipe["max_length"])
        source_tokens[split] = dataset.audit
        if job["smoke"]:
            lengths, protected = dataset.lengths, [int(np.argmax(dataset.lengths))]
            if job["task"] == "ckd":
                other = "shuffled" if job["condition"] == "ordered" else "ordered"
                other_sequences, other_labels = read_data(plan, "ckd", other, split)
                if not np.array_equal(labels, other_labels) or any(
                    sorted(left.split(SEP)) != sorted(right.split(SEP))
                    for left, right in zip(sequences, other_sequences)
                ):
                    raise ValueError("CKD conditions are not paired by labels and event multisets")
                other_data = TextDataset(other_sequences, other_labels, tokenizer, "ckd", recipe["max_length"])
                lengths = np.maximum(lengths, other_data.lengths)
                protected.append(int(np.argmax(other_data.lengths)))
                del other_data, other_sequences
            indices = smoke_indices(labels, lengths, 64 if split == "train" else 32,
                                    recipe["seed"], protected_indices=protected)
        else:
            indices = split_indices(labels, split, recipe)
            if split == "train":
                verify_reference_indices(indices, recipe)
        selected[split] = indices
        datasets[split] = dataset.subset(indices)
        np.save(output / f"{split}_indices.npy", indices, allow_pickle=False)
        source = plan["datasets"][job["task"]][job["condition"]][split]
        provenance[split] = {"rows": source["rows"], "positive_rows": int(labels.sum()),
            **{key: {"path": source[key], "sha256": source[key + "_sha256"]} for key in ("features", "labels")}}
        del dataset, sequences
    rows = {split: len(data) for split, data in datasets.items()}
    indices_hash = {split: sha256(output / f"{split}_indices.npy") for split in selected}
    config = {"schema_version": 1, "arm": ARM, "model_name": MODEL_NAME,
        "task": job["task"], "data_condition": job["condition"], "condition": job["condition"],
        "data": provenance, "data_provenance": provenance, "model_source": model_provenance(model_path),
        "rows": rows, **recipe, "is_smoke": job["smoke"], "is_dry_run": args.dry_run,
        "plan_sha256": plan["plan_sha256"], "index_sha256": indices_hash,
        "precision": "bf16_autocast", "trainable_dtype": "float32", "gradient_checkpointing": True,
        "truncation": False, "padding_side": "right", "optimizer": "AdamW", "optimizer_offload": "none",
        "prompt": "Sequential events: <events>" + ("\nOutcome (0 or 1):" if job["task"] == "ckd" else ""),
        "loss": "unweighted_two_class_cross_entropy_float32_no_language_model_loss",
        "selection": "minimum_validation_loss", "threshold_selection": "validation_f1",
        "threshold_grid": {"start": .1, "stop_exclusive": .9, "step": .01},
        "test_informed_stopping": False, "test_evaluation": "once_after_model_selection",
        "fresh_pretrained_initialization": True,
        "checkpoint_policy": "selected_adapter_and_head_only; unfinished fit restarts",
        "sampling": "stratified_subset_including_both_conditions_longest_rows" if job["smoke"] else
                    "fixed_stratified_training_subset_full_validation_test",
        "row_identity": "zero_based_row_index_in_fixed_csv_split"}
    manifest = {"config": config, "job": {**job, "output_dir": str(output)}, "recipe": recipe,
        "plan_sha256": plan["plan_sha256"], "index_sha256": indices_hash,
        "created_at_unix": time.time(), "tokenization": {split: data.audit for split, data in datasets.items()},
        "token_lengths": {split: data.audit for split, data in datasets.items()},
        "source_token_lengths": source_tokens, "source_sha256": plan["source_sha256"]}
    atomic_json(manifest, output / "manifest.json")
    if args.dry_run:
        audit = {"status": "completed", "job": args.job, "tokenization": manifest["tokenization"],
                 "source_token_lengths": source_tokens, "rows": rows, "weights_loaded": False,
                 "gpu_used": False, "truncation": False, "index_sha256": indices_hash}
        atomic_json(audit, output / "audit.json")
        progress("completed", weights_loaded=False, gpu_used=False)
        return audit

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --dry-run performs the CPU token audit")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    seed_all(recipe["seed"])
    progress("loading_model")
    from src.experiments.kip_efficiency_models import load_classifier
    model, loaded_tokenizer, metadata = load_classifier(ARM, model_path, device=device, gradient_checkpointing=True)
    if loaded_tokenizer.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("model and data padding token IDs differ")
    manifest["model"] = metadata
    manifest["device"] = str(device)
    atomic_json(manifest, output / "manifest.json")
    generator = torch.Generator().manual_seed(recipe["seed"])
    loaders = {split: DataLoader(data,
        batch_size=recipe["batch_size"] if split == "train" else recipe["eval_batch_size"],
        shuffle=split == "train", generator=generator if split == "train" else None,
        collate_fn=DynamicPadding(tokenizer.pad_token_id, recipe["max_length"]),
        num_workers=0, pin_memory=device.type == "cuda") for split, data in datasets.items()}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("nonempty FP32 trainable parameters required")
    optimizer = torch.optim.AdamW(parameters, lr=recipe["lr"], weight_decay=.01, foreach=False)
    accumulation = recipe["gradient_accumulation_steps"]
    total_steps = recipe["epochs"] * math.ceil(len(loaders["train"]) / accumulation)
    warmup = int(.06 * total_steps)

    def schedule(step):
        return step / max(1, warmup) if step < warmup else max(0., (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stopper, best_state, history, optimizer_steps = EarlyStopping(recipe["patience"]), None, [], 0
    stop_reason = "max_epochs"
    training_started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    progress("ready", rows=rows, trainable_parameters=sum(parameter.numel() for parameter in parameters),
             planned_optimizer_steps=total_steps)
    for epoch in range(1, recipe["epochs"] + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = torch.zeros((), device=device, dtype=torch.float64), 0
        progress("training", epoch=epoch, batch=0, batches=len(loaders["train"]))
        for index, batch in enumerate(loaders["train"]):
            stop, group_samples = accumulation_samples(index, len(loaders["train"]), recipe["batch_size"],
                                                       rows["train"], accumulation)
            loss, _ = classifier_loss(model, batch, device)
            count = len(batch["labels"])
            (loss * count / group_samples).backward()
            loss_sum += loss.detach().double() * count
            seen += count
            if index + 1 == stop:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if (index + 1) % 20 == 0 or index + 1 == len(loaders["train"]):
                average = loss_sum.item() / seen
                if not math.isfinite(average):
                    raise ValueError("non-finite training loss")
                progress("training", epoch=epoch, epochs=recipe["epochs"], batch=index + 1,
                         batches=len(loaders["train"]), train_loss=average, optimizer_steps=optimizer_steps,
                         examples_per_second=seen / max(1e-9, time.monotonic() - epoch_started))
        progress("validating", epoch=epoch)
        val_loss, probabilities, labels = evaluate(model, loaders["val"], device)
        improved = stopper.update(val_loss, epoch)
        if improved:
            best_state = save_trainable_checkpoint(model, output, epoch)
        history_record = {"epoch": epoch, "train_loss": loss_sum.item() / seen, "val_loss": val_loss,
            "val_auc": classification_metrics(labels, probabilities, .5)["auc"],
            "val_ap": float(average_precision_score(labels, probabilities)),
            "improved": improved, "best_epoch": stopper.best_epoch, "bad_epochs": stopper.bad_epochs,
            "optimizer_steps": optimizer_steps, "learning_rate": scheduler.get_last_lr()[0],
            "epoch_seconds": time.monotonic() - epoch_started}
        history.append(history_record)
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(history_record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        progress("epoch_complete", **history_record)
        if epoch >= recipe["min_epochs"] and stopper.stopped:
            stop_reason = "validation_patience"
            break
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
    training_seconds = time.monotonic() - training_started
    restore_trainable(model, best_state)
    del best_state, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    progress("evaluating_selected", best_epoch=stopper.best_epoch)
    val_loss, val_p, val_y = evaluate(model, loaders["val"], device)
    threshold = validation_threshold(val_y, val_p)
    metrics = {"status": "completed", "task": job["task"], "condition": job["condition"], "arm": ARM,
        "model": ARM, "is_smoke": job["smoke"], "rows": rows, "plan_sha256": plan["plan_sha256"],
        "best_epoch": stopper.best_epoch, "epochs_completed": len(history), "optimizer_steps": optimizer_steps,
        "early_stopped": stop_reason == "validation_patience", "stop_reason": stop_reason,
        "selected_by": "validation_loss", "threshold_selected_on": "validation", "test_informed_stopping": False,
        "tokenization": manifest["tokenization"], "training_seconds": training_seconds,
        "selected_weights_saved": True, "checkpoint_policy": config["checkpoint_policy"],
        "selected_checkpoint_sha256": sha256(output / "best_trainable.pt"),
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}
    for split in ("val", "test"):
        progress("evaluating", split=split)
        loss, probabilities, labels = (val_loss, val_p, val_y) if split == "val" else evaluate(model, loaders[split], device)
        name = "validation" if split == "val" else "test"
        metrics[name] = {"loss": loss, **classification_metrics(labels, probabilities, threshold)}
        metrics[name + "_at_0_5"] = classification_metrics(labels, probabilities, .5)
        pd.DataFrame({"row_index": selected[split], "label": labels, "probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
            "prediction_at_0_5": (probabilities >= .5).astype(int)}).to_csv(output / f"{split}_predictions.csv", index=False)
    metrics["wall_seconds"] = time.monotonic() - started
    metrics["artifact_sha256"] = {path.name: sha256(path) for path in output.iterdir()
        if path.is_file() and path.name not in {"status.json", "metrics.json"}}
    atomic_json(metrics, output / "metrics.json")
    progress("completed", test_auc=metrics["test"]["auc"], best_epoch=stopper.best_epoch)
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("batch_size", "eval_batch_size"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error("--" + name.replace("_", "-") + " must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except Exception as error:
        output = args.output_dir
        if output is None:
            try:
                plan = json.loads((args.run_root / "launch_plan.json").read_text())
                output = Path(next(job for job in plan["jobs"] if job["key"] == args.job)["output_dir"])
            except (OSError, ValueError, KeyError, StopIteration):
                output = None
        if output is not None and output.is_dir() and not isinstance(error, FileExistsError):
            atomic_json({"status": "failed", "pid": os.getpid(), "updated_at_unix": time.time(),
                         "error_type": type(error).__name__, "error": str(error)}, output / "status.json")
        raise


if __name__ == "__main__":
    main()
