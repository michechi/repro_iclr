"""Fresh BERT/RoBERTa classifiers on the existing OC and CKD splits."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.experiments.kip_encoder_training import (
    DynamicPadding, EarlyStopping, evaluate, metric_values, seed_all, sha256,
    subset_indices, validation_threshold,
)
from src.analysis.kip_checkpoint_attention import read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


def prompt(sequence, task):
    text = "Sequential events: " + sequence.replace("\x1f", " ")
    return text + ("\nOutcome (0 or 1):" if task == "ckd" else "")


def read_data(plan, task, condition, split):
    source = plan["datasets"][task][condition][split]
    for key in ("features", "labels"):
        if sha256(Path(source[key])) != source[key + "_sha256"]:
            raise ValueError("frozen input changed: " + source[key])
    sequences = pd.read_csv(source["features"], dtype=str, keep_default_na=False)["Sequences"].tolist()
    labels = pd.read_csv(source["labels"])["Outcome"].to_numpy()
    if (len(sequences) != source["rows"] or len(labels) != len(sequences)
            or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2):
        raise ValueError("invalid split rows or labels")
    return sequences, labels.astype(np.int64)


def split_indices(labels, split, recipe, smoke=False):
    limit = (64 if split == "train" else 32) if smoke else recipe.get("train_rows") if split == "train" else None
    if not smoke and limit is not None and not 0 < limit <= len(labels):
        raise ValueError("requested training rows exceed the available split or are nonpositive")
    return subset_indices(labels, limit, recipe["seed"])


class EncoderDataset:
    def __init__(self, sequences, labels, tokenizer, task):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_ids, self.full_lengths = [], []
        self.max_length = 512 if task == "ckd" else 64
        for start in range(0, len(sequences), 512):
            texts = [prompt(s, task) for s in sequences[start:start + 512]]
            full = tokenizer(texts, truncation=False, return_attention_mask=False,
                             return_token_type_ids=False)["input_ids"]
            self.full_lengths.extend(map(len, full))
            if task == "ckd":
                ids = tokenizer(texts, truncation=True, max_length=512,
                                return_attention_mask=False, return_token_type_ids=False)["input_ids"]
            else:
                ids = full
                if any(len(x) > 64 for x in ids):
                    raise ValueError("OC would require truncation")
            self.input_ids.extend(ids)
        if len(self.input_ids) != len(self.labels) or not len(self.labels):
            raise ValueError("nonempty matching rows required")
        lengths = np.array(self.full_lengths)
        self.audit = dict(rows=len(lengths), min_full_tokens=int(lengths.min()),
                          max_full_tokens=int(lengths.max()), max_input_tokens=max(map(len, self.input_ids)),
                          truncated_rows=int((lengths > self.max_length).sum()),
                          truncated_fraction=float((lengths > self.max_length).mean()),
                          context_policy="native 512-token right truncation" if task == "ckd" else "no truncation")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], int(self.labels[index])


def classification_metrics(y, probability, threshold):
    values = metric_values(y, probability, threshold)
    return {**values, "auroc": float(values["auc"]),
            "average_precision": float(average_precision_score(y, probability)),
            "positive_rows": int(np.sum(y)), "rows": len(y)}


def train(args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    plan = read_json(args.run_root / "launch_plan.json")
    job = next(j for j in plan["jobs"] if j["key"] == args.job)
    if job["kind"] != "fit":
        raise ValueError("not a fit job")
    output = Path(args.output_dir) if getattr(args, "output_dir", None) else Path(job["output_dir"])
    job = {**job, "output_dir": str(output)}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("fresh fit requires an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    recipe = dict(plan["recipes"][job["task"]])
    if job["smoke"]:
        recipe.update(epochs=1, min_epochs=1)
    batch_size = args.batch_size or recipe["batch_size"]
    if recipe["effective_batch_size"] % batch_size:
        raise ValueError("batch size must divide the fixed effective batch size")
    accumulation = recipe["effective_batch_size"] // batch_size
    eval_batch = min(recipe["eval_batch_size"], batch_size) if args.batch_size else recipe["eval_batch_size"]
    seed_all(recipe["seed"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    def progress(status, **detail):
        write_json(output / "status.json", dict(status=status, job=args.job, pid=os.getpid(),
            updated_at_unix=time.time(), wall_seconds=time.monotonic() - started, **detail))

    progress("loading_data")
    model_record = plan["models"][job["model"]]
    for filename, expected in model_record["files_sha256"].items():
        if sha256(Path(model_record["path"]) / filename) != expected:
            raise ValueError("pretrained model changed")
    tokenizer = AutoTokenizer.from_pretrained(model_record["path"], local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.truncation_side = "right"
    datasets, selected = {}, {}
    for split in ("train", "val", "test"):
        x, y = read_data(plan, job["task"], job["condition"], split)
        indices = split_indices(y, split, recipe, smoke=job["smoke"])
        selected[split] = indices
        np.save(output / f"{split}_indices.npy", indices, allow_pickle=False)
        datasets[split] = EncoderDataset([x[i] for i in indices], y[indices], tokenizer, job["task"])
    progress("loading_model")
    model, loading = AutoModelForSequenceClassification.from_pretrained(
        model_record["path"], num_labels=2, dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False, output_loading_info=True)
    if model.config.model_type != job["model"]:
        raise ValueError("wrong encoder architecture")
    if loading.get("error_msgs") or loading.get("mismatched_keys"):
        raise ValueError("pretrained encoder loading failed")
    if any(not (name.startswith("classifier.") or name.startswith("pooler."))
           for name in loading.get("missing_keys", [])):
        raise ValueError("pretrained encoder backbone has missing weights")
    model.to(device)
    maximum = 512 if job["task"] == "ckd" else 64
    generator = torch.Generator().manual_seed(recipe["seed"])
    loaders = {split: DataLoader(data, batch_size=batch_size if split == "train" else eval_batch,
        shuffle=split == "train", generator=generator if split == "train" else None,
        collate_fn=DynamicPadding(tokenizer.pad_token_id, maximum), num_workers=0,
        pin_memory=device.type == "cuda") for split, data in datasets.items()}
    manifest = dict(plan_sha256=plan["plan_sha256"], job=job, model_source=model_record,
        recipe={**recipe, "batch_size": batch_size, "eval_batch_size": eval_batch,
                "gradient_accumulation_steps": accumulation},
        prompt=prompt("<events>", job["task"]), truncation_side="right", loading_info=loading,
        tokenization={k:v.audit for k,v in datasets.items()},
        index_sha256={s:sha256(output / f"{s}_indices.npy") for s in selected},
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        environment={"torch":torch.__version__,"transformers":importlib.metadata.version("transformers"),
                     "gpu":torch.cuda.get_device_name() if device.type == "cuda" else None},
        source_sha256=plan["source_sha256"], selection="minimum_validation_loss",
        test_informed_stopping=False, checkpoint_policy="best full model retained; unfinished fits restart")
    write_json(output / "manifest.json", manifest)
    tokenizer.save_pretrained(output / "tokenizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=.01, foreach=False)
    total_steps = recipe["epochs"] * math.ceil(len(loaders["train"]) / accumulation)
    warmup = int(total_steps * .06)

    def schedule(step):
        return step / max(1, warmup) if step < warmup else max(0., (total_steps-step) / max(1,total_steps-warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    stopper, best, steps, history = EarlyStopping(recipe["patience"]), None, 0, []
    for epoch in range(1, recipe["epochs"] + 1):
        epoch_started = time.monotonic()
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_loss, seen = 0., 0
        for index, batch in enumerate(loaders["train"]):
            moved = {k:v.to(device) for k,v in batch.items()}
            group_start = index // accumulation * accumulation
            group_stop = min(group_start + accumulation, len(loaders["train"]))
            group_samples = min(group_stop*batch_size,len(datasets["train"])) - group_start*batch_size
            count = len(batch["labels"])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                result = model(**moved)
                loss = result.loss * count / group_samples
            loss.backward()
            total_loss += float(result.loss.detach()) * count; seen += count
            if index+1 == group_stop:
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); steps += 1
            if index % 20 == 0 or index+1 == len(loaders["train"]):
                if not math.isfinite(total_loss):
                    raise ValueError("non-finite training loss")
                progress("training", epoch=epoch, epochs=recipe["epochs"], batch=index+1,
                    batches=len(loaders["train"]), train_loss=total_loss/seen, optimizer_steps=steps,
                    examples_per_second=seen/max(1e-9,time.monotonic()-epoch_started))
        progress("validating", epoch=epoch)
        val_loss, val_p, val_y = evaluate(model, loaders["val"], device, "bf16")
        improved = stopper.update(val_loss,epoch)
        if improved:
            best = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        record = dict(epoch=epoch, train_loss=total_loss/seen, val_loss=val_loss,
            val_auc=classification_metrics(val_y,val_p,.5)["auc"], val_ap=float(average_precision_score(val_y,val_p)),
            improved=improved, best_epoch=stopper.best_epoch, bad_epochs=stopper.bad_epochs,
            optimizer_steps=steps, epoch_seconds=time.monotonic()-epoch_started)
        history.append(record)
        with (output / "history.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n"); f.flush(); os.fsync(f.fileno())
        if stopper.stopped and epoch >= recipe["min_epochs"]:
            break
    if best is None:
        raise RuntimeError("no selected checkpoint")
    model.load_state_dict(best,strict=True)
    del best, optimizer
    progress("saving_selected_model", best_epoch=stopper.best_epoch)
    model.save_pretrained(output / "best_model", safe_serialization=True)
    tokenizer.save_pretrained(output / "best_model")
    val_loss, val_p, val_y = evaluate(model,loaders["val"],device,"bf16")
    threshold = validation_threshold(val_y,val_p)
    metrics = dict(status="completed", task=job["task"], model=job["model"], condition=job["condition"],
        is_smoke=job["smoke"], best_epoch=stopper.best_epoch, epochs_completed=len(history),
        stop_reason="validation_patience" if stopper.stopped else "max_epochs",
        selected_by="validation_loss", threshold_selected_on="validation", test_informed_stopping=False,
        tokenization=manifest["tokenization"], plan_sha256=plan["plan_sha256"])
    for split in ("val", "test"):
        progress("evaluating", split=split)
        loss,p,y = (val_loss,val_p,val_y) if split == "val" else evaluate(model,loaders[split],device,"bf16")
        metrics["validation" if split == "val" else "test"] = {"loss":loss,**classification_metrics(y,p,threshold)}
        pd.DataFrame(dict(row_index=selected[split],label=y,probability=p)).to_csv(output/f"{split}_predictions.csv",index=False)
    metrics.update(wall_seconds=time.monotonic()-started,
        selected_checkpoint_sha256=sha256(output / "best_model/model.safetensors"),
        artifact_sha256={f.name:sha256(f) for f in output.iterdir() if f.is_file() and f.name not in {"status.json","metrics.json"}})
    write_json(output / "metrics.json",metrics)
    progress("completed", test_auc=metrics["test"]["auc"], best_epoch=stopper.best_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--job",required=True)
    parser.add_argument("--batch-size",type=int)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    train(parser.parse_args())
