"""Pretrained BERT/RoBERTa classification on KIP splits."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.experiments.paper_protocol import CONFIG, PROTOCOL_ID, SEEDS, sequence_prompt


SCHEMA_VERSION = 1
SEP = "\x1f"


def atomic_json(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def atomic_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_split(directory: Path, tag: str, split: str) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    x_path = directory / f"X_{split}_{tag}.csv"
    y_path = directory / f"y_{split}_{tag}.csv"
    x = pd.read_csv(x_path, dtype=str, keep_default_na=False)
    y = pd.read_csv(y_path)
    if "Sequences" not in x or "Outcome" not in y:
        raise ValueError(f"{split}: expected Sequences and Outcome columns")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError(f"{split}: nonempty matching feature/label rows required")
    if y["Outcome"].isna().any() or not y["Outcome"].isin([0, 1]).all():
        raise ValueError(f"{split}: labels must be binary integers")
    labels = y["Outcome"].to_numpy(dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{split}: both label classes are required")
    sequences = x["Sequences"].tolist()
    seq_len = len(sequences[0].split(SEP))
    for row, sequence in enumerate(sequences):
        try:
            sequence_prompt(sequence, sequence_length=seq_len)
        except ValueError as error:
            raise ValueError(f"{x_path}, row {row}: {error}") from error
    provenance = {
        "rows": len(x), "positive_rows": int(labels.sum()),
        "sequence_length": seq_len,
        "features": {"path": str(x_path.resolve()), "sha256": sha256(x_path)},
        "labels": {"path": str(y_path.resolve()), "sha256": sha256(y_path)},
    }
    return sequences, labels, provenance


def subset_indices(labels: np.ndarray, limit: int | None, seed: int, *,
                   stratify_by: np.ndarray | None = None) -> np.ndarray:
    """Select a stratified subset. Remark A.4: pass latent Y* as ``stratify_by`` when available."""
    indices = np.arange(len(labels))
    if limit is None or limit >= len(labels):
        return indices
    if limit < 2:
        raise ValueError("row limits must permit both label classes")
    strata = labels if stratify_by is None else np.asarray(stratify_by)
    if len(strata) != len(labels):
        raise ValueError("stratify_by must align with labels")
    selected, _ = train_test_split(indices, train_size=limit, stratify=strata, random_state=seed)
    return np.sort(selected)


class TokenizedDataset(Dataset):
    def __init__(self, sequences: list[str], labels: np.ndarray, tokenizer: Any, max_length: int,
                 *, decoder: bool = False, sequence_length: int | None = None):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_ids: list[list[int]] = []
        if len(sequences) != len(labels) or not sequences:
            raise ValueError("tokenization requires nonempty matching sequences and labels")
        seq_len = len(sequences[0].split(SEP)) if sequence_length is None else sequence_length
        for start in range(0, len(sequences), 2048):
            texts = [sequence_prompt(sequence, decoder=decoder, sequence_length=seq_len)
                     for sequence in sequences[start:start + 2048]]
            encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_length, add_special_tokens=True,
                                return_attention_mask=False, return_token_type_ids=False)
            self.input_ids.extend(encoded["input_ids"])
        self.max_tokens = max(map(len, self.input_ids))
        self.min_tokens = min(map(len, self.input_ids))
        if self.min_tokens < 1 or self.max_tokens > max_length:
            raise ValueError(
                f"token lengths {self.min_tokens}..{self.max_tokens} exceed allowed "
                f"1..{max_length}; truncation is forbidden"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.input_ids[index], int(self.labels[index])


@dataclass
class DynamicPadding:
    pad_token_id: int
    max_length: int
    multiple: int = 8

    def __call__(self, rows: list[tuple[list[int], int]]) -> dict[str, torch.Tensor]:
        longest = max(len(ids) for ids, _ in rows)
        if longest > self.max_length:
            raise ValueError("batch would require truncation")
        width = min(self.max_length, math.ceil(longest / self.multiple) * self.multiple)
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, (tokens, _) in enumerate(rows):
            ids[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[index, :len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": torch.tensor([label for _, label in rows], dtype=torch.long)}


@dataclass
class EarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    bad_epochs: int = 0

    def update(self, loss: float, epoch: int) -> bool:
        if not math.isfinite(loss):
            raise ValueError("validation loss is non-finite")
        if loss < self.best_loss:
            self.best_loss, self.best_epoch, self.bad_epochs = loss, epoch, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def stopped(self) -> bool:
        return self.bad_epochs >= self.patience


def metric_values(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    if not np.isfinite(probabilities).all():
        raise ValueError("non-finite model probabilities")
    predictions = probabilities >= threshold
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
    }


def validation_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.arange(10, 90, dtype=np.int64) / 100
    scores = [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(generator: torch.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError("checkpoint trainable parameter names do not match the model")
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint tensors: {incompatible.unexpected_keys}")


def autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             precision: str) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses, count = 0.0, 0
    probabilities, labels = [], []
    for batch in loader:
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast(device, precision):
            output = model(**moved)
        batch_size = len(batch["labels"])
        losses += float(output.loss.item()) * batch_size
        count += batch_size
        probabilities.append(output.logits.float().softmax(-1)[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy())
    loss = losses / count
    probabilities_array = np.concatenate(probabilities)
    if not math.isfinite(loss) or not np.isfinite(probabilities_array).all():
        raise ValueError("non-finite validation/test output")
    return loss, probabilities_array, np.concatenate(labels)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path, help="Local pretrained snapshot")
    parser.add_argument("--revision", help="Hub revision when downloading a model by name")
    parser.add_argument("--tag", choices=["kip_m4", "kip_m6"], required=True)
    parser.add_argument("--mode", choices=["ordered", "shuffled_train"], required=True)
    parser.add_argument("--tuning", choices=["lora", "full"], required=True)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--ordered-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=CONFIG.effective_batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CONFIG.weight_decay)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=CONFIG.patience)
    parser.add_argument("--warmup-ratio", type=float, default=CONFIG.warmup_ratio)
    parser.add_argument("--max-length", type=int, default=CONFIG.max_length)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--progress-steps", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = 1 if args.smoke else CONFIG.max_epochs
    if args.smoke:
        for split, default in [("train", 256), ("val", 128), ("test", 128)]:
            name = f"max_{split}_rows"
            if getattr(args, name) is None:
                setattr(args, name, default)
    for name in ["batch_size", "eval_batch_size", "gradient_accumulation_steps", "epochs",
                 "patience", "max_length", "threads", "progress_steps"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup_ratio < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("learning rate must be positive and warmup ratio must be in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    return args


def train(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    started = time.monotonic()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.type == "cuda" and args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("requested BF16 training is unsupported on this GPU")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    status_path = output_dir / "status.json"
    last_path = output_dir / "last.pt"
    history_path = output_dir / "history.jsonl"
    previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous_manifest is not None and not args.resume:
        raise FileExistsError(f"{manifest_path} exists; use --resume to continue this run")
    if last_path.exists() and previous_manifest is None:
        raise ValueError("checkpoint exists without its manifest")

    def progress(state: str, **details: Any) -> None:
        atomic_json({"status": "completed" if state == "complete" else state,
                     "state": state, "pid": os.getpid(), "updated_at_unix": time.time(),
                     "wall_seconds_this_process": time.monotonic() - started,
                     "model_name": args.model_name, "tag": args.tag, "mode": args.mode,
                     "tuning": args.tuning, "seed": args.seed, **details}, status_path)

    progress("loading_data")
    selected_data: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    provenance: dict[str, Any] = {}
    selected_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        ordered, ordered_labels, ordered_meta = read_split(args.ordered_dir, args.tag, split)
        shuffled, shuffled_labels, shuffled_meta = read_split(args.shuffled_dir, args.tag, split)
        if not np.array_equal(ordered_labels, shuffled_labels):
            raise ValueError(f"{split}: ordered/shuffled labels or row counts differ")
        if any(sorted(left.split(SEP)) != sorted(right.split(SEP))
               for left, right in zip(ordered, shuffled)):
            raise ValueError(f"{split}: shuffled rows do not preserve corresponding event multisets")
        indices = subset_indices(ordered_labels, getattr(args, f"max_{split}_rows"), args.seed)
        source = ordered if args.mode == "ordered" else shuffled
        selected_data[split] = ([source[index] for index in indices], ordered_labels[indices], indices)
        if split == "test" and args.mode == "ordered":
            selected_data["shuffled_test"] = ([shuffled[index] for index in indices],
                                              shuffled_labels[indices], indices)
        provenance[split] = {"ordered": ordered_meta, "shuffled": shuffled_meta}
        selected_counts[split] = len(indices)
    source = str(args.model_path.resolve()) if args.model_path else args.model_name
    if args.model_path is not None and not args.model_path.is_dir():
        raise FileNotFoundError(f"pretrained snapshot does not exist: {args.model_path}")
    source_files: dict[str, str] = {}
    if args.model_path is not None:
        for path in sorted(args.model_path.iterdir()):
            if path.is_file() and path.suffix in {".json", ".txt", ".safetensors", ".bin"}:
                source_files[path.name] = sha256(path)
    is_smoke = args.smoke or any(selected_counts[split] < provenance[split]["ordered"]["rows"]
                                 for split in ("val", "test"))
    seq_len = provenance["test"]["ordered"]["sequence_length"]
    config = {
        "schema_version": SCHEMA_VERSION, "model_name": args.model_name, "model_source": source,
        "revision": args.revision, "source_file_sha256": source_files,
        "tag": args.tag, "mode": args.mode, "tuning": args.tuning, "seed": args.seed,
        "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
        "patience": args.patience, "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length, "precision": args.precision,
        "prompt": "Sequential events: " + " ".join(["<event>"] * seq_len),
        "lora": {"rank": CONFIG.lora_r, "alpha": CONFIG.lora_alpha, "dropout": CONFIG.lora_dropout,
                 "targets": ["query", "key", "value"]} if args.tuning == "lora" else None,
        "is_smoke": is_smoke, "selected_rows": selected_counts, "data": provenance,
        "selection": "minimum_validation_loss", "threshold_selection": "validation_f1",
        "stopping_rule": "validation_loss_patience",
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
    }
    if previous_manifest is not None and previous_manifest["config"] != config:
        differing = [key for key in set(config) | set(previous_manifest["config"])
                     if config.get(key) != previous_manifest["config"].get(key)]
        raise ValueError(f"resume configuration differs: {sorted(differing)}")
    if previous_manifest is not None and (output_dir / "metrics.json").exists():
        metrics = json.loads((output_dir / "metrics.json").read_text())
        required = [output_dir / "best_model" / ("adapter_model.safetensors"
                    if args.tuning == "lora" else "model.safetensors"),
                    output_dir / "test_predictions.csv"]
        if args.mode == "ordered":
            required.append(output_dir / "shuffled_test_predictions.csv")
        if any(not path.is_file() for path in required):
            raise ValueError("completed metrics exist without selected model weights/predictions")
        progress("complete", best_epoch=metrics["best_epoch"], resumed_completed_run=True)
        return metrics
    load_kwargs: dict[str, Any] = {"cache_dir": str(args.cache_dir), "local_files_only": args.model_path is not None}
    if args.revision is not None and args.model_path is None:
        load_kwargs["revision"] = args.revision
    progress("loading_model")
    tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
    if tokenizer.pad_token_id is None:
        raise ValueError("encoder tokenizer must define a padding token")
    if args.max_length > tokenizer.model_max_length:
        raise ValueError("max length exceeds the tokenizer's supported context")
    model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2, **load_kwargs)
    if model.config.model_type not in {"bert", "roberta"}:
        raise ValueError(f"expected BERT/RoBERTa, got {model.config.model_type}")
    if model.config.model_type == "roberta":
        # Main tables use RoBERTa-large; Appendix G KIP/OC attention uses RoBERTa-base.
        shape = (model.config.hidden_size, model.config.num_hidden_layers)
        if shape not in {(1024, 24), (768, 12)}:
            raise ValueError("RoBERTa fits require roberta-large (1024/24) or Appendix G roberta-base (768/12)")
    paper_settings = (args.max_length == CONFIG.max_length
                      and args.batch_size * args.gradient_accumulation_steps == CONFIG.effective_batch_size
                      and args.lr == CONFIG.learning_rate
                      and args.weight_decay == CONFIG.weight_decay
                      and args.warmup_ratio == CONFIG.warmup_ratio
                      and args.epochs == CONFIG.max_epochs
                      and args.patience == CONFIG.patience and args.seed in SEEDS)
    if not is_smoke and not paper_settings:
        raise ValueError("non-smoke encoder fits must use Table 1 settings: length 200, effective batch 16, "
                         "lr 2e-5, weight decay 0.01, warmup 0.06, 20 epochs, patience 3, and seed 9550, 9551, or 9552")
    if args.tuning == "lora":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=CONFIG.lora_r, lora_alpha=CONFIG.lora_alpha,
                               lora_dropout=CONFIG.lora_dropout,
                               target_modules=["query", "key", "value"], bias="none", task_type="SEQ_CLS"))
    model.to(device)
    progress("tokenizing")
    datasets = {split: TokenizedDataset(sequences, labels, tokenizer, args.max_length, sequence_length=seq_len)
                for split, (sequences, labels, _) in selected_data.items()}
    token_lengths = {split: {"min": data.min_tokens, "max": data.max_tokens}
                     for split, data in datasets.items()}
    manifest = {"config": config, "created_at_unix": time.time(),
                "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                                "device": str(device), "gpu": torch.cuda.get_device_name(device)
                                if device.type == "cuda" else None},
                "pretrained_commit": getattr(model.config, "_commit_hash", None),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()
                                            if parameter.requires_grad),
                "token_lengths": token_lengths}
    if previous_manifest is None:
        atomic_json(manifest, manifest_path)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.config.save_pretrained(output_dir / "model_config")
    generator = torch.Generator().manual_seed(args.seed)
    collator = DynamicPadding(tokenizer.pad_token_id, args.max_length)
    loaders = {split: DataLoader(data, batch_size=args.batch_size if split == "train" else args.eval_batch_size,
                                 shuffle=split == "train", generator=generator if split == "train" else None,
                                 collate_fn=collator, num_workers=0, pin_memory=device.type == "cuda")
               for split, data in datasets.items()}
    train_loader = loaders["train"]
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_ratio * total_steps)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stopper = EarlyStopping(args.patience)
    start_epoch, optimizer_steps, prior_seconds = 1, 0, 0.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config:
            raise ValueError("checkpoint config differs from this run")
        restore_trainable(model, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        stopper = EarlyStopping(**checkpoint["early_stopping"])
        best_state = checkpoint["best_model"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        optimizer_steps = checkpoint["optimizer_steps"]
        prior_seconds = checkpoint["training_seconds"]
        restore_rng(checkpoint["rng"], generator)
        del checkpoint
    with history_path.open("w") as handle:
        for record in history:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    training_started = time.monotonic()
    print(json.dumps({"event": "ready", "config": {key: value for key, value in config.items()
                                                if key not in {"data", "source_file_sha256"}},
                      "token_lengths": token_lengths, "trainable_parameters": manifest["trainable_parameters"]}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        if stopper.stopped:
            break
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, seen = torch.zeros((), device=device, dtype=torch.float64), 0
        progress("training", epoch=epoch, epochs=args.epochs, batch=0, batches=len(train_loader))
        for index, batch in enumerate(train_loader):
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            group_start = (index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_stop = min(group_start + args.gradient_accumulation_steps, len(train_loader))
            group_samples = min(group_stop * args.batch_size, len(datasets["train"])) - group_start * args.batch_size
            batch_size = len(batch["labels"])
            with autocast(device, args.precision):
                output = model(**moved)
                loss = output.loss * batch_size / group_samples
            loss.backward()
            total_loss += output.loss.detach().double() * batch_size
            seen += batch_size
            if index + 1 == group_stop:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if (index + 1) % args.progress_steps == 0 or index + 1 == len(train_loader):
                average_loss = float(total_loss.item()) / seen
                if not math.isfinite(average_loss):
                    raise ValueError(f"non-finite training loss at epoch={epoch}, batch={index + 1}")
                progress("training", epoch=epoch, epochs=args.epochs, batch=index + 1,
                         batches=len(train_loader), train_loss=average_loss,
                         examples_per_second=seen / max(time.monotonic() - epoch_started, 1e-9),
                         optimizer_steps=optimizer_steps, learning_rate=scheduler.get_last_lr()[0],
                         best_epoch=stopper.best_epoch)
        progress("validating", epoch=epoch, epochs=args.epochs)
        val_loss, val_probabilities, val_labels = evaluate(model, loaders["val"], device, args.precision)
        improved = stopper.update(val_loss, epoch)
        if improved:
            best_state = trainable_state(model)
        record = {"epoch": epoch, "train_loss": float(total_loss.item()) / seen, "val_loss": val_loss,
                  "val_auc": float(roc_auc_score(val_labels, val_probabilities)),
                  "val_f1_at_0_5": float(f1_score(val_labels, val_probabilities >= 0.5, zero_division=0)),
                  "best_epoch": stopper.best_epoch, "improved": improved,
                  "bad_epochs": stopper.bad_epochs, "epoch_seconds": time.monotonic() - epoch_started,
                  "optimizer_steps": optimizer_steps, "learning_rate": scheduler.get_last_lr()[0]}
        history.append(record)
        progress("checkpointing", epoch=epoch, best_epoch=stopper.best_epoch)
        atomic_checkpoint({"config": config, "epoch": epoch, "model": trainable_state(model),
                           "best_model": best_state, "optimizer": optimizer.state_dict(),
                           "scheduler": scheduler.state_dict(), "early_stopping": vars(stopper),
                           "rng": rng_state(generator), "optimizer_steps": optimizer_steps,
                           "history": history,
                           "training_seconds": prior_seconds + time.monotonic() - training_started}, last_path)
        with history_path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
    if best_state is None:
        raise RuntimeError("no completed validation-selected epoch exists")
    progress("saving_best_model", best_epoch=stopper.best_epoch)
    restore_trainable(model, best_state)
    best_dir = output_dir / "best_model"
    model.save_pretrained(best_dir, safe_serialization=True)
    tokenizer.save_pretrained(best_dir)
    model.config.save_pretrained(output_dir / "model_config")
    progress("evaluating", best_epoch=stopper.best_epoch)
    val_loss, val_probabilities, val_labels = evaluate(model, loaders["val"], device, args.precision)
    threshold = validation_threshold(val_labels, val_probabilities)
    metrics: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "model_name": args.model_name,
        "tag": args.tag, "mode": args.mode, "tuning": args.tuning, "seed": args.seed,
        "is_smoke": is_smoke, "rows": selected_counts, "best_epoch": stopper.best_epoch,
        "epochs_completed": history[-1]["epoch"], "early_stopped": stopper.stopped,
        "selected_by": "validation_loss", "threshold_selected_on": "validation",
        "stopping_rule": "validation_loss_patience",
        "max_length": args.max_length,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        **({"protocol_id": PROTOCOL_ID, "protocol": PROTOCOL_ID,
            "paper_recipe": f"{'roberta' if model.config.model_type == 'roberta' else 'bert'}_{args.tuning}"}
           if not is_smoke and paper_settings else {}),
        "validation": {"loss": val_loss, **metric_values(val_labels, val_probabilities, threshold)},
        "validation_at_0_5": metric_values(val_labels, val_probabilities, 0.5),
        "best_model": str(best_dir), "manifest": str(manifest_path),
        "training_seconds": prior_seconds + time.monotonic() - training_started,
    }
    for split in ["test"] + (["shuffled_test"] if args.mode == "ordered" else []):
        loss, probabilities, labels = evaluate(model, loaders[split], device, args.precision)
        metrics[split] = {"loss": loss, **metric_values(labels, probabilities, threshold)}
        metrics[f"{split}_at_0_5"] = metric_values(labels, probabilities, 0.5)
        pd.DataFrame({"row_index": selected_data[split][2], "label": labels,
                      "probability": probabilities, "prediction": (probabilities >= threshold).astype(int),
                      "prediction_at_0_5": (probabilities >= 0.5).astype(int)}).to_csv(
                          output_dir / f"{split}_predictions.csv", index=False)
    metrics["wall_seconds_this_process"] = time.monotonic() - started
    atomic_json(metrics, output_dir / "metrics.json")
    progress("complete", best_epoch=stopper.best_epoch, epochs_completed=history[-1]["epoch"],
             test_auc=metrics["test"]["auc"], is_smoke=is_smoke)
    print(json.dumps({"event": "complete", "metrics": metrics}, allow_nan=False), flush=True)
    return metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        train(args)
    except Exception as error:
        if args.output_dir.is_dir() and not isinstance(error, FileExistsError):
            atomic_json({"status": "failed", "state": "failed", "pid": os.getpid(), "updated_at_unix": time.time(),
                         "error_type": type(error).__name__, "error": str(error)}, args.output_dir / "status.json")
        raise


if __name__ == "__main__":
    main()
