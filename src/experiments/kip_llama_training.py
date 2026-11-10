"""Loss, evaluation, and gradient-accumulation utilities for decoder and encoder classifiers."""
from __future__ import annotations

import math
import shutil

import numpy as np
import torch
from torch.nn import functional as F

from src.experiments.kip_encoder_training import autocast

CHECKPOINT_POLICY = "selected_model_only; interrupted unfinished run must restart"


def classifier_loss(model, batch, device, precision="bf16"):
    """Return FP32 head cross-entropy, or the decoder wrapper's prompt-LM plus head cross-entropy."""
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    labels = moved.pop("labels")
    with autocast(device, precision):
        output = model(**moved, labels=labels) if getattr(model, "paper_objective", False) else model(**moved)
        logits = output.logits
    loss = output.loss if getattr(model, "paper_objective", False) else F.cross_entropy(logits.float(), labels)
    return loss, logits.float()


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
        raise FileExistsError("selected-model destination/staging exists")
    parameter_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    needed = math.ceil(parameter_bytes * 1.02) + 64 * 1024 ** 2
    available = shutil.disk_usage(output_dir).free
    if available < needed:
        raise OSError(f"selected-model export requires {needed} free bytes, available {available}")
    model.save_pretrained(staging, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(staging)
    staging.rename(destination)
    return destination
