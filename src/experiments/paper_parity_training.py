"""Training and curriculum transfer for Appendix E.1-E.5 parity experiments."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.experiments.paper_parity_protocol import (
    PROTOCOL_ID, stable_seed, draw_sequences, generate_split, validate_parent_job,
)


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class ParityTransformer(nn.Module):
    def __init__(self, ell, n, *, cls_readout=False, dropout=.3):
        super().__init__()
        self.embedding = nn.Embedding(ell + 1, 64, padding_idx=0)
        self.positions = nn.Parameter(torch.randn(1, n + int(cls_readout), 64))
        self.cls = nn.Parameter(torch.randn(1, 1, 64)) if cls_readout else None
        layer = nn.TransformerEncoderLayer(64, 8, dim_feedforward=512, dropout=dropout,
                                          activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        for block in self.encoder.layers:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()
            block.self_attn._reset_parameters()
        self.norm = nn.LayerNorm(64)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        mask = x != 0
        hidden = self.embedding(x)
        if self.cls is not None:
            hidden = torch.cat((self.cls.expand(len(x), -1, -1), hidden), dim=1)
            mask = torch.cat((torch.ones(len(x), 1, device=x.device, dtype=torch.bool), mask), dim=1)
        if hidden.shape[1] > self.positions.shape[1]:
            raise ValueError("sequence exceeds configured positional length")
        hidden = self.encoder(hidden + self.positions[:, :hidden.shape[1]], src_key_padding_mask=~mask)
        pooled = hidden[:, 0] if self.cls is not None else ((hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1))
        return self.head(self.dropout(self.norm(pooled)))


class ParityLSTM(nn.Module):
    def __init__(self, ell):
        super().__init__()
        self.embedding = nn.Embedding(ell + 1, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(.3)
        self.head = nn.Linear(64, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(self.embedding(x))
        return self.head(self.dropout(hidden[-1]))


def transfer_curriculum_state(model, checkpoint, job):
    """Copy Stage 1 weights into Stage 2, extending Transformer positional embeddings."""
    validate_parent_job(checkpoint["job"], job, checkpoint.get("protocol_id"))
    current = model.state_dict()
    previous = checkpoint["state_dict"]
    if current.keys() != previous.keys():
        raise ValueError("curriculum checkpoint keys differ from current architecture")
    merged = {}
    for name, value in previous.items():
        if current[name].shape == value.shape:
            merged[name] = value
        elif name == "positions" and job["model"] == "transformer" and tuple(value.shape) == (1, 10, 64) and tuple(current[name].shape) == (1, 20, 64):
            combined = current[name].clone()
            combined[:, :10] = value
            merged[name] = combined
        else:
            raise ValueError(f"unexpected curriculum shape mismatch: {name}")
    model.load_state_dict(merged, strict=True)


def bert_inputs(x, tokenizer):
    prompts = ["Sequential events: " + " ".join(chr(64 + int(i)) for i in row) for row in x]
    return tokenizer(prompts, padding=True, truncation=True, max_length=200, return_tensors="pt")


def run_job(job, run_root, *, bert_path=None, device="cuda"):
    from sklearn.metrics import f1_score, roc_auc_score
    from src.experiments.kip_encoder_training import validation_threshold, metric_values
    from src.experiments.paper_pretrained_training import load_paper_model
    from transformers import get_linear_schedule_with_warmup

    root = Path(run_root).resolve()
    out = root / job["id"]
    if out.exists():
        raise FileExistsError("output directory already exists")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if job["model"] == "bert" and bert_path is None:
        raise ValueError("BERT runs require an explicit local --bert-path")
    out.mkdir(parents=True)
    torch.manual_seed(job["seed"])
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    tokenizer, model_metadata = None, None
    if job["model"] == "transformer":
        model = ParityTransformer(job["alphabet_size"], job["sequence_length"], cls_readout=job["section"] == "E1")
    elif job["model"] == "lstm":
        model = ParityLSTM(job["alphabet_size"])
    else:
        model, tokenizer, model_metadata = load_paper_model("bert", bert_path, tuning="lora", device=device)
        from src.experiments.paper_protocol import CONFIG
        model_metadata = {k: v for k, v in model_metadata.items() if k not in CONFIG.to_dict()}
        model_metadata["training_configuration_source"] = "manifest.job.config"
    model.to(device)
    parent = None
    if job["depends_on"]:
        parent = root / job["depends_on"] / "selected.pt"
        parent_result = json.loads(parent.with_name("metrics.json").read_text())
        if parent_result.get("status") != "completed":
            raise ValueError("stage one must have completed before stage two")
        validate_parent_job(parent_result["job"], job, parent_result.get("protocol_id"))
        if hashlib.sha256(parent.read_bytes()).hexdigest() != parent_result.get("checkpoint_sha256"):
            raise ValueError("stage-one checkpoint does not match its completed result hash")
        checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
        transfer_curriculum_state(model, checkpoint, job)
    data = {s: generate_split(job, s) for s in ("train", "val", "test") if job["rows"][s]
            and not (s == "train" and job["section"] == "E1" and job["regime"] == "streaming")}
    provenance = {s: {"rows": len(x), "x_sha256": array_sha(x), "y_sha256": array_sha(y),
                      "seed": stable_seed(job["data_seed"], s)} for s, (x, y) in data.items()}
    manifest = {"protocol_id": PROTOCOL_ID, "job": job, "status": "running", "data": provenance,
                "parent_checkpoint_sha256": hashlib.sha256(parent.read_bytes()).hexdigest() if parent else None,
                "model_metadata": model_metadata, "parameters": sum(p.numel() for p in model.parameters()),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                  [Path(__file__)] + [Path(__file__).with_name(name) for name in
                                   ("paper_parity_protocol.py", "paper_pretrained_training.py", "paper_protocol.py",
                                    "paper_data.py", "kip_encoder_training.py", "kip_llama_training.py")]},
                "environment": {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn", "transformers")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cfg = job["config"]
    opt_class = torch.optim.Adam if cfg["optimizer"] == "Adam" else torch.optim.AdamW
    optimizer = opt_class([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batch_size = cfg["batch_size"]

    def loss_scores(x, y):
        labels = torch.as_tensor(y, dtype=torch.long, device=device)
        if tokenizer is None:
            logits = model(torch.as_tensor(x, dtype=torch.long, device=device))
        else:
            inputs = {k: v.to(device) for k, v in bert_inputs(x, tokenizer).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**inputs).logits
        loss = F.cross_entropy(logits.float(), labels)
        return loss, logits.float().softmax(-1)[:, 1]

    def evaluate(split):
        model.eval()
        x, y = data[split]
        losses, scores = 0., []
        with torch.no_grad():
            for start in range(0, len(y), batch_size):
                loss, p = loss_scores(x[start:start + batch_size], y[start:start + batch_size])
                losses += float(loss) * len(p)
                scores.extend(p.cpu().tolist())
        return losses / len(y), np.asarray(scores), y

    started = time.monotonic()
    history, best, best_stat, bad, best_epoch = [], None, None, 0, None
    if job["section"] == "E1":
        rng = np.random.default_rng(stable_seed(job["data_seed"], "training_stream"))
        model.train()
        for step in range(1, cfg["steps"] + 1):
            if job["regime"] == "streaming":
                x, y = draw_sequences(batch_size, job["sequence_length"], job["alphabet_size"], job["keys"], rng)
            else:
                indices = rng.integers(0, len(data["train"][1]), size=batch_size)
                x, y = (a[indices] for a in data["train"])
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_scores(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            loss.backward()
            optimizer.step()
        best_epoch = cfg["steps"]
        selected_by = "fixed_1000_steps"
        threshold = .5
    else:
        x, y = data["train"]
        steps_per_epoch = (len(y) + batch_size - 1) // batch_size
        scheduler = None
        if cfg["schedule"] == "linear_6_percent_warmup":
            steps = steps_per_epoch * cfg["max_epochs"]
            scheduler = get_linear_schedule_with_warmup(optimizer, int(.06 * steps), steps)
        generator = np.random.default_rng(stable_seed(job["seed"], "minibatches"))
        for epoch in range(1, cfg["max_epochs"] + 1):
            model.train()
            order = generator.permutation(len(y))
            for start in range(0, len(y), batch_size):
                indices = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_scores(x[indices], y[indices])
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite loss")
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            val_loss, val_p, val_y = evaluate("val")
            val_f1 = float(f1_score(val_y, val_p >= .5, zero_division=0))
            statistic = val_loss if cfg["stopping"] == "validation_loss" else -val_f1
            if best_stat is None or statistic < best_stat:
                best_stat, bad, best_epoch = statistic, 0, epoch
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            history.append({"epoch": epoch, "validation_loss": val_loss, "validation_f1_at_half": val_f1})
            (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            if bad >= cfg["patience"]:
                break
        if best is None:
            raise RuntimeError("no finite validation-selected checkpoint")
        model.load_state_dict(best)
        _, val_p, val_y = evaluate("val")
        threshold = validation_threshold(val_y, val_p)
        selected_by = cfg["stopping"]
    torch.save({"protocol_id": PROTOCOL_ID, "job": job,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, out / "selected.pt")
    test_loss, test_p, test_y = evaluate("test")
    np.savez(out / "test_predictions.npz", labels=test_y, probabilities=test_p)
    result = {"protocol_id": PROTOCOL_ID, "status": "completed", "job": job,
              "selected_by": selected_by, "best_epoch_or_step": best_epoch,
              "epochs_completed": len(history) if history else None,
              "test": {"loss": test_loss, **metric_values(test_y, test_p, threshold)},
              "success_auc_gt_0_9": bool(roc_auc_score(test_y, test_p) > .9),
              "elapsed_seconds": time.monotonic() - started,
              "checkpoint_sha256": hashlib.sha256((out / "selected.pt").read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
              "predictions_sha256": hashlib.sha256((out / "test_predictions.npz").read_bytes()).hexdigest()}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
