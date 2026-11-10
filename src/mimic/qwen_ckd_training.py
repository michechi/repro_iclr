"""Qwen3.8-27B QLoRA outcome classifiers for the CKD-to-ESRD splits."""
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
from sklearn.metrics import average_precision_score
import torch
from torch.utils.data import DataLoader, Dataset

from src.experiments.kip_encoder_training import (
    SEP, DynamicPadding, EarlyStopping, atomic_checkpoint, atomic_json,
    metric_values, restore_trainable, seed_all, sha256, subset_indices,
    trainable_state, validation_threshold,
)
from src.experiments.kip_llama_training import accumulation_samples, classifier_loss, evaluate
from src.experiments.paper_protocol import CONFIG
from src.experiments.kip_data_efficiency import model_provenance

ROOT = Path(__file__).resolve().parents[2]
ARM = "qwen27b_qlora"
PROMPT = "Sequential events: {events}\nOutcome (0 or 1):"


def sequence_prompt(sequence):
    events = sequence.split(SEP)
    if not events or any(re.fullmatch(r"CCS_[0-9]+|OTHER", event) is None for event in events):
        raise ValueError("expected nonempty CCS_<number>/OTHER events separated by U+001F")
    return PROMPT.format(events=" ".join(events))


def read_split(directory, split, condition):
    x_path = directory / f"X_{split}_{condition}.csv"
    y_path = directory / f"y_{split}_{condition}.csv"
    features = pd.read_csv(x_path, dtype=str, keep_default_na=False)
    outcomes = pd.read_csv(y_path)
    if "Sequences" not in features or "Outcome" not in outcomes:
        raise ValueError(f"{split}/{condition}: expected Sequences and Outcome columns")
    if len(features) != len(outcomes) or len(features) < 2:
        raise ValueError(f"{split}/{condition}: matching nonempty feature/label rows required")
    if not outcomes["Outcome"].isin([0, 1]).all():
        raise ValueError(f"{split}/{condition}: labels must be binary integers")
    labels = outcomes["Outcome"].to_numpy(dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{split}/{condition}: both label classes required")
    sequences = features["Sequences"].tolist()
    for row, sequence in enumerate(sequences):
        try:
            sequence_prompt(sequence)
        except ValueError as error:
            raise ValueError(f"{split}/{condition}, row {row}: {error}") from error
    return sequences, labels, {
        "rows": len(labels), "positive_rows": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "features": {"path": str(x_path.resolve()), "sha256": sha256(x_path)},
        "labels": {"path": str(y_path.resolve()), "sha256": sha256(y_path)},
    }


def read_paired_data(directory, *, smoke=False):
    """Load and verify paired ordered and shuffled clinical splits."""
    data, provenance = {}, {}
    for split in (("train", "val") if smoke else ("train", "val", "test")):
        ordered, labels, ordered_meta = read_split(directory, split, "ordered")
        shuffled, shuffled_labels, shuffled_meta = read_split(directory, split, "shuffled")
        if not np.array_equal(labels, shuffled_labels):
            raise ValueError(f"{split}: ordered/shuffled labels or row counts differ")
        if any(sorted(left.split(SEP)) != sorted(right.split(SEP))
               for left, right in zip(ordered, shuffled)):
            raise ValueError(f"{split}: ordered/shuffled event multisets differ")
        data[split] = {"ordered": ordered, "shuffled": shuffled, "labels": labels}
        provenance[split] = {"ordered": ordered_meta, "shuffled": shuffled_meta}
    return data, provenance


class ClinicalDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        if not sequences or len(sequences) != len(labels):
            raise ValueError("tokenization requires matching nonempty sequences and labels")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_ids = []
        for start in range(0, len(sequences), 128):
            texts = [sequence_prompt(sequence) for sequence in sequences[start:start + 128]]
            encoded = tokenizer(texts, padding=False, truncation=False, add_special_tokens=True,
                                return_attention_mask=False, return_token_type_ids=False)
            self.input_ids.extend(encoded["input_ids"])
        if len(self.input_ids) != len(self.labels):
            raise ValueError("tokenizer changed the number of rows")
        lengths = np.array([len(ids) for ids in self.input_ids])
        if lengths.min() < 1 or lengths.max() > max_length:
            raise ValueError(f"token lengths {lengths.min()}..{lengths.max()} exceed allowed "
                             f"1..{max_length}; truncation is forbidden")
        self.lengths = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], int(self.labels[index])

    def subset(self, indices):
        result = object.__new__(ClinicalDataset)
        result.labels = self.labels[indices]
        result.input_ids = [self.input_ids[index] for index in indices]
        result.lengths = self.lengths[indices]
        return result

    def length_summary(self):
        return {"min": int(self.lengths.min()), "max": int(self.lengths.max()),
                "median": float(np.median(self.lengths)),
                "p95": float(np.percentile(self.lengths, 95)),
                "tokens": int(self.lengths.sum())}


def smoke_indices(dataset, limit, seed, longest_count=1):
    """Keep stratified class counts while including the longest rows in each split."""
    chosen = set(map(int, subset_indices(dataset.labels, min(limit, len(dataset)), seed)))
    longest = list(map(int, np.argsort(-dataset.lengths, kind="stable")[:longest_count]))
    protected = set(longest)
    for index in longest:
        if index in chosen:
            continue
        candidates = [other for other in sorted(chosen)
                      if other not in protected and dataset.labels[other] == dataset.labels[index]]
        if not candidates:
            raise ValueError("smoke sample cannot include longest rows while preserving class counts")
        chosen.remove(candidates[0])
        chosen.add(index)
    return np.array(sorted(chosen), dtype=np.int64)


def clinical_metrics(labels, probabilities, threshold):
    result = metric_values(labels, probabilities, threshold)
    average_precision = float(average_precision_score(labels, probabilities))
    return {**result, "auroc": result["auc"], "average_precision": average_precision,
            "auprc": average_precision, "auprc_definition": "average_precision",
            "rows": len(labels), "positive_rows": int(labels.sum()),
            "prevalence": float(labels.mean())}


def save_trainable_checkpoint(model, output, epoch):
    """Save adapter and classification head tensors."""
    state = trainable_state(model)
    needed = math.ceil(sum(value.numel() * value.element_size() for value in state.values()) * 1.05)
    needed += 64 * 1024 ** 2
    if shutil.disk_usage(output).free < needed:
        raise OSError(f"selected trainable checkpoint requires {needed} free bytes")
    atomic_checkpoint({"arm": ARM, "epoch": epoch, "state_dict": state}, output / "best_trainable.pt")
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/mimic_training")
    parser.add_argument("--data", choices=("ordered", "shuffled"), default="ordered")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    for name, default in (("batch-size", 2), ("gradient-accumulation-steps", 16),
                          ("eval-batch-size", 2), ("epochs", 30), ("patience", 5),
                          ("min-epochs", 5), ("max-length", 16384), ("threads", 4),
                          ("progress-steps", 20)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("batch_size", "gradient_accumulation_steps", "eval_batch_size", "epochs",
                 "patience", "min_epochs", "max_length", "threads", "progress_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.smoke:
        args.epochs = args.min_epochs = 1
    if args.min_epochs > args.epochs:
        parser.error("--min-epochs cannot exceed --epochs")
    if args.smoke and args.dry_run:
        parser.error("--smoke and --dry-run are separate modes")
    return args


def run(args):
    from transformers import AutoTokenizer

    started = time.monotonic()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    base = {"arm": ARM, "task": "ckd_to_esrd", "data": args.data, "seed": args.seed,
            "is_smoke": args.smoke, "is_dry_run": args.dry_run}

    def progress(status, **details):
        record = {**base, "status": status, "pid": os.getpid(), "updated_at_unix": time.time(),
                  "wall_seconds": time.monotonic() - started, **details}
        atomic_json(record, output / "status.json")
        print(json.dumps(record, allow_nan=False), flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(args.threads)
    progress("loading_data")
    paired, provenance = read_paired_data(args.data_dir, smoke=args.smoke)
    progress("tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True,
                                              trust_remote_code=False, padding_side="right")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer needs an EOS token for padding")
    tokenizer.pad_token = tokenizer.eos_token
    datasets, selected, token_summary, full_token_summary = {}, {}, {}, {}
    for split, source in paired.items():
        conditions = ("ordered", "shuffled") if args.dry_run else (args.data,)
        for condition in conditions:
            dataset = ClinicalDataset(source[condition], source["labels"], tokenizer, args.max_length)
            full_token_summary[f"{split}_{condition}"] = dataset.length_summary()
            if args.dry_run:
                continue
            if args.smoke:
                limit = min(64, 4 * args.batch_size * args.gradient_accumulation_steps) if split == "train" else 32
                indices = smoke_indices(dataset, limit, args.seed,
                                        longest_count=min(args.batch_size, 2) if split == "train" else 1)
            else:
                indices = np.arange(len(dataset), dtype=np.int64)
            datasets[split] = dataset.subset(indices)
            selected[split] = indices
            token_summary[split] = datasets[split].length_summary()
            np.save(output / f"{split}_indices.npy", indices, allow_pickle=False)
    if args.dry_run:
        audit = {**base, "status": "completed", "data_provenance": provenance,
                 "token_lengths": full_token_summary, "max_length": args.max_length,
                 "truncation": False, "paired_labels_and_event_multisets_verified": True,
                 "weights_loaded": False, "gpu_used": False}
        atomic_json(audit, output / "audit.json")
        progress("completed", audit="audit.json", weights_loaded=False)
        return audit
    del paired
    rows = {split: len(dataset) for split, dataset in datasets.items()}
    sources = [Path(__file__)] + [ROOT / "src/experiments" / name for name in (
        "kip_efficiency_models.py", "kip_encoder_training.py", "kip_llama_training.py",
        "kip_data_efficiency.py")]
    config = {**base, "schema_version": 1, "model_source": model_provenance(args.model_path),
              "data_provenance": provenance, "rows": rows, "batch_size": args.batch_size,
              "gradient_accumulation_steps": args.gradient_accumulation_steps,
              "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
              "eval_batch_size": args.eval_batch_size, "epochs": args.epochs,
              "patience": args.patience, "min_epochs": args.min_epochs, "lr": args.lr,
              "weight_decay": 0.01, "warmup_ratio": 0.06, "optimizer": "AdamW",
              "precision": "bf16_autocast", "trainable_dtype": "float32",
              "gradient_checkpointing": True, "max_length": args.max_length, "truncation": False,
              "prompt": PROMPT, "padding_side": "right",
              "loss": CONFIG.decoder_loss,
              "selection": "minimum_validation_loss", "threshold_selection": "validation_f1",
              "threshold_grid": {"start": 0.1, "stop_exclusive": 0.9, "step": 0.01},
              "test_evaluation": "once_after_model_selection",
              "row_identity": "zero_based_row_index_in_fixed_csv_split",
              "sampling": "all_rows" if not args.smoke else "stratified_subset_including_longest_rows",
              "index_sha256": {split: sha256(output / f"{split}_indices.npy") for split in selected}}
    manifest = {"config": config, "created_at_unix": time.time(), "token_lengths": token_summary,
                "source_token_lengths": full_token_summary,
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources}}
    atomic_json(manifest, output / "manifest.json")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --dry-run for the CPU data check")
    torch.set_float32_matmul_precision("high")
    seed_all(args.seed)
    device = torch.device("cuda")
    progress("loading_model")
    from src.experiments.kip_efficiency_models import load_classifier
    model, loaded_tokenizer, model_metadata = load_classifier(ARM, args.model_path, gradient_checkpointing=True, protocol="paper")
    if loaded_tokenizer.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("model/tokenization padding IDs differ")
    manifest["model"] = model_metadata
    atomic_json(manifest, output / "manifest.json")
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size if split == "train" else args.eval_batch_size,
                                 shuffle=split == "train", generator=generator if split == "train" else None,
                                 collate_fn=DynamicPadding(tokenizer.pad_token_id, args.max_length),
                                 num_workers=0, pin_memory=True)
               for split, dataset in datasets.items()}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("trainable parameters must be nonempty and FP32")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01, foreach=False)
    total_steps = args.epochs * math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps)
    warmup_steps = int(0.06 * total_steps)

    def schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stopper = EarlyStopping(args.patience)
    best_state, history, optimizer_steps = None, [], 0
    stop_reason = "max_epochs"
    training_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    progress("ready", rows=rows, trainable_parameters=sum(p.numel() for p in parameters),
             token_lengths=token_summary, planned_optimizer_steps=total_steps)
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, seen = torch.zeros((), device=device, dtype=torch.float64), 0
        progress("training", epoch=epoch, batch=0, batches=len(loaders["train"]))
        for index, batch in enumerate(loaders["train"]):
            stop, group_samples = accumulation_samples(index, len(loaders["train"]), args.batch_size,
                                                       rows["train"], args.gradient_accumulation_steps)
            loss, _ = classifier_loss(model, batch, device, "bf16")
            count = len(batch["labels"])
            (loss * count / group_samples).backward()
            loss_sum += loss.detach().double() * count
            seen += count
            if index + 1 == stop:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if (index + 1) % args.progress_steps == 0 or index + 1 == len(loaders["train"]):
                average = loss_sum.item() / seen
                if not math.isfinite(average):
                    raise ValueError("non-finite training loss")
                progress("training", epoch=epoch, batch=index + 1, batches=len(loaders["train"]),
                         train_loss=average, optimizer_steps=optimizer_steps,
                         examples_per_second=seen / (time.monotonic() - epoch_started))
        progress("validating", epoch=epoch)
        val_loss, probabilities, labels = evaluate(model, loaders["val"], device)
        improved = stopper.update(val_loss, epoch)
        if improved:
            best_state = save_trainable_checkpoint(model, output, epoch)
        record = {"epoch": epoch, "train_loss": loss_sum.item() / seen, "val_loss": val_loss,
                  "val_auc": clinical_metrics(labels, probabilities, 0.5)["auc"],
                  "val_auprc": float(average_precision_score(labels, probabilities)),
                  "improved": improved, "best_epoch": stopper.best_epoch, "bad_epochs": stopper.bad_epochs,
                  "optimizer_steps": optimizer_steps, "learning_rate": scheduler.get_last_lr()[0],
                  "epoch_seconds": time.monotonic() - epoch_started}
        history.append(record)
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        progress("epoch_complete", **record)
        if epoch >= args.min_epochs and stopper.stopped:
            stop_reason = "validation_patience"
            break
    if best_state is None:
        raise RuntimeError("no selected model")
    training_seconds = time.monotonic() - training_started
    restore_trainable(model, best_state)
    del best_state, optimizer, scheduler
    torch.cuda.empty_cache()
    progress("evaluating_selected", best_epoch=stopper.best_epoch)
    val_loss, val_probabilities, val_labels = evaluate(model, loaders["val"], device)
    threshold = validation_threshold(val_labels, val_probabilities)
    metrics = {**base, "status": "completed", "schema_version": 1, "rows": rows,
               "best_epoch": stopper.best_epoch, "epochs_completed": history[-1]["epoch"],
               "optimizer_steps": optimizer_steps, "early_stopped": stop_reason == "validation_patience",
               "stop_reason": stop_reason, "selected_by": "validation_loss",
               "threshold_selected_on": "validation", "training_seconds": training_seconds,
               "validation": {"loss": val_loss, **clinical_metrics(val_labels, val_probabilities, threshold)},
               "validation_at_0_5": clinical_metrics(val_labels, val_probabilities, 0.5),
               "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
               "selected_weights_saved": True}
    evaluations = [("val", val_probabilities, val_labels)]
    if not args.smoke:
        progress("evaluating_test", best_epoch=stopper.best_epoch)
        test_loss, test_probabilities, test_labels = evaluate(model, loaders["test"], device)
        metrics["test"] = {"loss": test_loss, **clinical_metrics(test_labels, test_probabilities, threshold)}
        metrics["test_at_0_5"] = clinical_metrics(test_labels, test_probabilities, 0.5)
        evaluations.append(("test", test_probabilities, test_labels))
    for split, probabilities, labels in evaluations:
        pd.DataFrame({"row_index": selected[split], "label": labels, "probability": probabilities,
                      "prediction": (probabilities >= threshold).astype(int),
                      "prediction_at_0_5": (probabilities >= 0.5).astype(int)}).to_csv(
                          output / f"{split}_predictions.csv", index=False)
    metrics["wall_seconds"] = time.monotonic() - started
    metrics["artifact_sha256"] = {path.name: sha256(path) for path in output.iterdir()
                                  if path.is_file() and path.name not in {"status.json", "metrics.json"}}
    atomic_json(metrics, output / "metrics.json")
    progress("completed", best_epoch=stopper.best_epoch,
             **({"test_auc": metrics["test"]["auc"], "test_auprc": metrics["test"]["auprc"]}
                if "test" in metrics else {"validation_auc": metrics["validation"]["auc"]}))
    return metrics


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except Exception as error:
        if args.output_dir.is_dir() and not isinstance(error, FileExistsError):
            atomic_json({"status": "failed", "pid": os.getpid(), "updated_at_unix": time.time(),
                         "error_type": type(error).__name__, "error": str(error)}, args.output_dir / "status.json")
        raise


if __name__ == "__main__":
    main()
