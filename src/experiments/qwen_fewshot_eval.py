"""Auditable, classification-only in-context evaluation of a local Qwen base model.

No training, generation, chat template, remote code, or truncation is used. Each
probability is the softmax restricted to the verified one-token labels 0 and 1.
An interrupted cell restarts; completed cells are skipped only after identity
and artifact-hash validation. --dry-run validates data/sampling without weights.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.metrics import recall_score, roc_auc_score


SCHEMA_VERSION = 1
TASKS = ("oc_det", "oc_noise", "gpc", "kip4", "kip6", "oc_det_legacy", "oc_noise_legacy")
DEFAULT_TASKS = TASKS[:5]
SEP = "\x1f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def ensure_json(path: Path, value: Any) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"configuration differs from existing artifact: {path}")
    else:
        atomic_json(path, value)


def normalize_sequence(sequence: str) -> str:
    """Return compact events, accepting the existing unit-separated CSV format."""
    if SEP in sequence:
        events = sequence.split(SEP)
    elif any(c.isspace() for c in sequence):
        events = sequence.split()
    else:
        events = list(sequence)
    if len(events) != 20 or any(len(c) != 1 or not "A" <= c <= "Z" for c in events):
        raise ValueError("expected exactly 20 uppercase A-Z events")
    return "".join(events)


def render_example(sequence: str, label: int | None) -> str:
    body = " ".join(normalize_sequence(sequence))
    head = f"Sequential events: {body}\nOutcome (0 or 1): "
    if label is None:
        return head
    if label not in (0, 1):
        raise ValueError("demonstration labels must be 0 or 1")
    return head + str(int(label))


def build_prompt(demonstrations: list[tuple[str, int]], query: str) -> str:
    parts = [render_example(sequence, label) for sequence, label in demonstrations]
    parts.append(render_example(query, None))
    return "\n\n".join(parts)


def read_split(directory: Path, tag: str, split: str):
    xp, yp = directory / f"X_{split}_{tag}.csv", directory / f"y_{split}_{tag}.csv"
    x = pd.read_csv(xp, dtype=str, keep_default_na=False)
    y = pd.read_csv(yp)
    if "Sequences" not in x or "Outcome" not in y or len(x) != len(y) or len(x) < 2:
        raise ValueError(f"{directory}/{tag}/{split}: matching Sequences/Outcome rows required")
    if y["Outcome"].isna().any() or not y["Outcome"].isin([0, 1]).all():
        raise ValueError(f"{yp}: labels must be binary 0/1 (0.0/1.0 accepted)")
    labels = y["Outcome"].to_numpy(dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{yp}: both label classes are required")
    sequences = []
    for index, sequence in enumerate(x["Sequences"]):
        try:
            sequences.append(normalize_sequence(sequence))
        except ValueError as error:
            raise ValueError(f"{xp}, row {index}: {error}") from error
    sources = {"features": {"path": str(xp.resolve()), "sha256": sha256(xp)},
               "labels": {"path": str(yp.resolve()), "sha256": sha256(yp)},
               "rows": len(labels), "positive_rows": int(labels.sum())}
    return sequences, labels, sources


def stratified_test_indices(labels: np.ndarray, n_test: int, seed: int) -> np.ndarray:
    """Fixed, approximately proportional stratification; always include both classes."""
    labels = np.asarray(labels)
    if set(labels.tolist()) != {0, 1} or not 2 <= n_test <= len(labels):
        raise ValueError("test sample requires both classes and 2 <= n_test <= available rows")
    if n_test == len(labels):
        return np.arange(len(labels), dtype=np.int64)
    positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    n_positive = math.floor(n_test * len(positives) / len(labels) + .5)
    n_positive = min(max(n_positive, 1, n_test - len(negatives)), len(positives), n_test - 1)
    rng = np.random.default_rng(seed)
    selected = np.concatenate([rng.choice(positives, n_positive, replace=False),
                               rng.choice(negatives, n_test - n_positive, replace=False)])
    return np.sort(selected)


def sample_demo_indices(labels: np.ndarray, shots: int, seed: int,
                        eligible: np.ndarray | None = None) -> np.ndarray:
    """Balanced floor/ceil counts; odd shot counts have one extra negative label."""
    labels = np.asarray(labels)
    if shots < 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("nonnegative shots and binary labels required")
    if eligible is None:
        eligible = np.ones(len(labels), dtype=bool)
    if eligible.shape != labels.shape or eligible.dtype != bool:
        raise ValueError("eligible must be a boolean mask aligned with training labels")
    rng = np.random.default_rng(seed)
    parts = []
    for label, count in ((0, shots - shots // 2), (1, shots // 2)):
        candidates = np.flatnonzero((labels == label) & eligible)
        if len(candidates) < count:
            raise ValueError(f"insufficient eligible training examples for {shots} balanced shots")
        parts.append(rng.choice(candidates, count, replace=False))
    chosen = np.concatenate(parts)
    rng.shuffle(chosen)
    return chosen


def assert_no_overlap(demonstrations: list[tuple[str, int]], queries: list[str]) -> None:
    query_set = {normalize_sequence(x) for x in queries}
    if any(normalize_sequence(x) in query_set for x, _ in demonstrations):
        raise ValueError("demonstration sequence overlaps sampled test sequence")


def build_cells(tasks: list[str], shots: list[int], demo_seeds: list[int]) -> list[dict]:
    cells = []
    for task in dict.fromkeys(tasks):
        for count in sorted(set(shots)):
            for seed in ([None] if count == 0 else sorted(set(demo_seeds))):
                key = f"{task}/shots{count}" + (f"/demo_seed{seed}" if seed is not None else "")
                cells.append({"task": task, "shots": count, "demo_seed": seed, "key": key})
    return cells


def tokenize_prompts(tokenizer, prompts: list[str], max_length: int):
    """Verify each fully rendered label is exactly one token after this prompt.

    In particular we never tokenize ' 0' in isolation and then discard its
    whitespace token: the conditioning prompt already contains that whitespace.
    Unsupported retokenizing/multi-token continuations fail rather than mis-score.
    """
    token_ids, label_ids = [], []
    for row, prompt in enumerate(prompts):
        if not prompt.endswith(" "):
            raise ValueError("query prompt must include the space before its label")
        prefix = list(tokenizer.encode(prompt, add_special_tokens=False))
        if not prefix or len(prefix) > max_length:
            raise ValueError(f"row {row}: {len(prefix)} prompt tokens exceed 1..{max_length}; truncation forbidden")
        continuations = []
        for label in ("0", "1"):
            full = list(tokenizer.encode(prompt + label, add_special_tokens=False))
            if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
                raise ValueError(f"row {row}: unsupported label tokenization for {label}; "
                                 "requires exact prompt prefix plus one continuation token")
            continuations.append(int(full[-1]))
        if continuations[0] == continuations[1]:
            raise ValueError("labels 0 and 1 must have distinct continuation tokens")
        token_ids.append(prefix)
        label_ids.append(continuations)
    return token_ids, label_ids


def score_batch(model, token_ids, label_ids, pad_token_id: int, device: str):
    import torch
    width = max(map(len, token_ids))
    inputs = torch.full((len(token_ids), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(inputs)
    for i, tokens in enumerate(token_ids):
        inputs[i, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention[i, -len(tokens):] = 1
    positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
    with torch.inference_mode():
        output = model(input_ids=inputs, attention_mask=attention, position_ids=positions,
                       use_cache=False, logits_to_keep=1)
        last = output.logits[:, -1, :].float()
        label_tensor = torch.tensor(label_ids, dtype=torch.long, device=device)
        logits = last.gather(1, label_tensor)
        log_odds = logits[:, 1] - logits[:, 0]
        probabilities = torch.sigmoid(log_odds)
        if not torch.isfinite(logits).all() or not torch.isfinite(log_odds).all() or not torch.isfinite(probabilities).all():
            raise ValueError("non-finite label logits or probabilities")
        return logits.cpu().numpy(), log_odds.cpu().numpy(), probabilities.cpu().numpy()


def metrics(labels, probabilities, ranking_scores=None) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape != probabilities.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError("metrics require aligned arrays with both true label classes")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    ranking = probabilities if ranking_scores is None else np.asarray(ranking_scores)
    if ranking.shape != labels.shape or not np.isfinite(ranking).all():
        raise ValueError("ranking scores must be finite and aligned with labels")
    predicted = probabilities >= .5
    return {"auc": float(roc_auc_score(labels, ranking)),
            "auc_from_probabilities": float(roc_auc_score(labels, probabilities)),
            "accuracy": float(accuracy_score(labels, predicted)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "threshold": .5, "rows": len(labels), "positive_labels": int(labels.sum()),
            "positive_predictions": int(predicted.sum()),
            "confusion_matrix_labels_0_1": confusion_matrix(labels, predicted, labels=[0, 1]).tolist()}


def completed_cell(directory: Path, cell_identity: str) -> dict | None:
    path = directory / "metrics.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result.get("cell_identity") != cell_identity:
        raise ValueError(f"completed cell configuration differs: {directory}")
    if result.get("status") != "completed":
        raise ValueError(f"invalid completed cell status: {directory}")
    artifacts = result.get("artifact_sha256", {})
    if set(artifacts) != {"predictions.csv", "manifest.json", "demonstrations.json"}:
        raise ValueError(f"completed cell artifact inventory invalid: {directory}")
    for name, expected in artifacts.items():
        artifact = directory / name
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"completed artifact missing or changed: {artifact}")
    sample = json.loads((directory / "manifest.json").read_text()).get("test_sample", {})
    sample_path = Path(sample.get("path", ""))
    if not sample_path.is_file() or sha256(sample_path) != sample.get("sha256"):
        raise ValueError(f"completed test sample missing or changed: {sample_path}")
    return result


def task_source(args, task: str) -> tuple[Path, str]:
    if task in ("kip4", "kip6"):
        return args.kip_data_dir, {"kip4": "kip_m4", "kip6": "kip_m6"}[task]
    if task in ("oc_det", "oc_noise"):
        return args.oc_data_dir or args.data_dir, args.oc_det_tag if task == "oc_det" else args.oc_noise_tag
    return args.data_dir, {"gpc": "test_just_pair", "oc_det_legacy": "6", "oc_noise_legacy": "9"}[task]


def prepare_data(args):
    data, provenance = {}, {}
    for task in args.tasks:
        directory, tag = task_source(args, task)
        train_x, train_y, train_source = read_split(directory, tag, "train")
        test_x, test_y, test_source = read_split(directory, tag, "test")
        selected = stratified_test_indices(test_y, args.n_test, args.test_seed)
        queries = [test_x[int(i)] for i in selected]
        selected_set = set(queries)
        eligible = np.array([x not in selected_set for x in train_x], dtype=bool)
        data[task] = {"train_x": train_x, "train_y": train_y, "queries": queries,
                      "labels": test_y[selected], "test_indices": selected.tolist(), "eligible": eligible}
        provenance[task] = {"tag": tag, "train": train_source, "test": test_source,
                            "test_indices": selected.tolist(),
                            "sampled_test_positives": int(test_y[selected].sum()),
                            "excluded_training_rows_matching_sampled_test": int((~eligible).sum())}
    return data, provenance


def demonstration_payload(data, cell):
    chosen = sample_demo_indices(data["train_y"], cell["shots"], cell["demo_seed"] or 0,
                                 data["eligible"])
    examples = [(data["train_x"][int(i)], int(data["train_y"][i])) for i in chosen]
    assert_no_overlap(examples, data["queries"])
    return {"train_indices": chosen.tolist(), "positive_labels": sum(y for _, y in examples),
            "negative_labels": sum(y == 0 for _, y in examples),
            "examples": [{"train_index": int(i), "sequence": sequence, "label": label,
                          "text": render_example(sequence, label)}
                         for i, (sequence, label) in zip(chosen, examples)],
            "text_prefix": "\n\n".join(render_example(x, y) for x, y in examples)}


def model_source(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"local model snapshot not found: {path}")
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "vocab.json", "merges.txt", "model.safetensors.index.json", "generation_config.json")
    small = {name: sha256(path / name) for name in names if (path / name).is_file()}
    weights = sorted(path.glob("*.safetensors"))
    if not weights:
        raise FileNotFoundError(f"no local safetensors model weights found: {path}")
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        required = set(json.loads(index_path.read_text())["weight_map"].values())
        if any(Path(name).name != name or not (path / name).is_file() for name in required):
            raise ValueError("safetensors index references an absent or invalid shard")
    return {"path": str(path.resolve()),
            "snapshot_revision": path.name if re.fullmatch(r"[a-f0-9]{40}", path.name) else None,
            "metadata_sha256": small,
            "weight_files": {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                             for p in weights},
            "weight_identity_note": "weight size/mtime and index hash; weights are not individually SHA256-hashed"}


def versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn", "accelerate", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen evaluation requires an available BF16-capable CUDA device")
    torch.cuda.set_device(torch.device(args.device))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Qwen evaluation requires an available BF16-capable CUDA device")
    torch.manual_seed(args.test_seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True,
                                               trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer requires an EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(args.model_path), local_files_only=True,
              trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
              device_map={"": args.device}, attn_implementation="sdpa")
    if model.config.model_type != "qwen2":
        raise ValueError(f"expected Qwen2.5 (qwen2 architecture), found {model.config.model_type}")
    model.requires_grad_(False)
    model.eval()
    model.config.use_cache = False
    context = int(model.config.max_position_embeddings)
    if args.max_length > context:
        raise ValueError(f"max-length {args.max_length} exceeds model context {context}")
    runtime = {"versions": versions(), "device": args.device,
               "gpu": torch.cuda.get_device_name(torch.device(args.device)),
               "cuda_build": torch.version.cuda, "parameter_dtype": "bfloat16",
               "score_dtype": "float32", "parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": 0, "model_type": model.config.model_type,
               "model_revision": getattr(model.config, "_commit_hash", None) or
                   (args.model_path.name if re.fullmatch(r"[a-f0-9]{40}", args.model_path.name) else None),
               "tokenizer_class": type(tokenizer).__name__, "tokenizer_name": tokenizer.name_or_path,
               "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
               "add_special_tokens": False, "chat_template_applied": False,
               "max_position_embeddings": context}
    return model, tokenizer, runtime


def run(args) -> dict:
    started = time.monotonic()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another evaluator holds this output-directory lock") from error
        return _run_locked(args, started)


def _run_locked(args, started) -> dict:
    output = args.output_dir
    data, provenance = prepare_data(args)
    cells = build_cells(args.tasks, args.shots, args.demo_seeds)
    demos = {c["key"]: demonstration_payload(data[c["task"]], c) for c in cells}
    config = {"schema_version": SCHEMA_VERSION, "model_name": args.model_name,
              "model_path": str(args.model_path.resolve()), "tasks": args.tasks,
              "shots": args.shots, "demo_seeds": args.demo_seeds, "test_seed": args.test_seed,
              "n_test": args.n_test, "batch_size": args.batch_size, "max_length": args.max_length,
              "device": args.device, "data": provenance, "script_sha256": sha256(Path(__file__)),
              "sampling": "fixed proportional stratified test; balanced train demos; odd shots extra negative",
              "prompt": "Sequential events: <20 space-separated letters>\nOutcome (0 or 1): ",
              "scoring": "sigmoid(float32(logit1)-float32(logit0)); exact one-token continuation verified",
              "auc_score": "raw_float32_log_odds_1_minus_0",
              "threshold": .5, "fine_tuning": False, "cells": cells}
    if args.dry_run:
        result = {"status": "data_validated", "config": config, "demonstrations": demos,
                  "tokenization_checked": False, "weights_loaded": False,
                  "note": "Tokenization/context/label-continuation checks occur before each evaluated cell.",
                  "seconds": time.monotonic() - started}
        atomic_json(output / "dry_run.json", result)
        print(json.dumps({"status": "data_validated", "cells": len(cells), "output": str(output)}), flush=True)
        return result
    config["model_source"] = model_source(args.model_path)
    config["versions"] = versions()
    config_path = output / "config.json"
    if not config_path.exists() and any(p.name not in {".run.lock", "dry_run.json"} for p in output.iterdir()):
        raise FileExistsError("nonempty output directory has no matching configuration")
    ensure_json(config_path, config)
    run_identity = identity(config)
    statuses = []
    pending = []
    for cell in cells:
        cell_id = identity({"run_identity": run_identity, "cell": cell, "demonstrations": demos[cell["key"]]})
        existing = completed_cell(output / "cells" / cell["key"], cell_id)
        if existing:
            statuses.append(existing)
        else:
            pending.append((cell, cell_id))

    def progress(state, **values):
        status = {"status": state, "pid": os.getpid(), "run_identity": run_identity,
                  "updated_at_unix": time.time(), "total_cells": len(cells),
                  "completed_cells": len(statuses), "wall_seconds": time.monotonic() - started, **values}
        atomic_json(output / "status.json", status)
        print(json.dumps(status), flush=True)

    try:
        if pending:
            progress("loading_model")
            loading_started = time.monotonic()
            model, tokenizer, runtime = load_model(args)
            runtime["model_load_seconds"] = time.monotonic() - loading_started
            atomic_json(output / "runtime.json", runtime)
            for cell, cell_id in pending:
                cell_started = time.monotonic()
                directory = output / "cells" / cell["key"]
                directory.mkdir(parents=True, exist_ok=True)
                demo = demos[cell["key"]]
                ensure_json(directory / "demonstrations.json", demo)
                task_data = data[cell["task"]]
                sample = {"test_seed": args.test_seed, "source": provenance[cell["task"]]["test"],
                          "indices": task_data["test_indices"], "sequences": task_data["queries"],
                          "labels": task_data["labels"].tolist()}
                sample_path = output / "samples" / f"{cell['task']}.json"
                ensure_json(sample_path, sample)
                examples = [(r["sequence"], r["label"]) for r in demo["examples"]]
                prompts = [build_prompt(examples, query) for query in task_data["queries"]]
                progress("tokenizing", cell=cell)
                tokenizing_started = time.monotonic()
                token_ids, labels_ids = tokenize_prompts(tokenizer, prompts, args.max_length)
                token_seconds = time.monotonic() - tokenizing_started
                lengths = list(map(len, token_ids))
                manifest = {"schema_version": SCHEMA_VERSION, "cell_identity": cell_id,
                            "run_identity": run_identity, "cell": cell, "runtime": runtime,
                            "test_sample": {"path": str(sample_path.resolve()), "sha256": sha256(sample_path)},
                            "demo_source": provenance[cell["task"]]["train"],
                            "prompt_sha256": identity(prompts), "input_ids_sha256": identity(token_ids),
                            "token_lengths": {"min": min(lengths), "max": max(lengths),
                                              "mean": float(np.mean(lengths)), "all": lengths},
                            "label_token_ids_by_row": labels_ids,
                            "tokenization_seconds": token_seconds}
                atomic_json(directory / "manifest.json", manifest)
                inference_started = time.monotonic()
                probabilities = []
                ranking_scores = []
                temporary = directory / "predictions.partial.csv"
                with temporary.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["test_row_index", "label", "probability_1", "prediction_at_0_5",
                                     "logit_0", "logit_1", "log_odds_1_minus_0", "prompt_tokens"])
                    for start in range(0, len(prompts), args.batch_size):
                        stop = min(start + args.batch_size, len(prompts))
                        logits, odds, batch_prob = score_batch(model, token_ids[start:stop], labels_ids[start:stop],
                                                               tokenizer.pad_token_id, args.device)
                        probabilities.extend(float(p) for p in batch_prob)
                        ranking_scores.extend(float(score) for score in odds)
                        for offset, p in enumerate(batch_prob):
                            row = start + offset
                            writer.writerow([task_data["test_indices"][row], int(task_data["labels"][row]),
                                             float(p), int(p >= .5), float(logits[offset, 0]),
                                             float(logits[offset, 1]), float(odds[offset]), lengths[row]])
                        handle.flush()
                        progress("evaluating", cell=cell, examples_completed=stop, examples_total=len(prompts),
                                 inference_seconds=time.monotonic() - inference_started)
                    os.fsync(handle.fileno())
                inference_seconds = time.monotonic() - inference_started
                temporary.replace(directory / "predictions.csv")
                result = {"status": "completed", "cell": cell, "cell_identity": cell_id,
                          "run_identity": run_identity,
                          "metrics": metrics(task_data["labels"], probabilities, ranking_scores),
                          "tokenization_seconds": token_seconds, "inference_seconds": inference_seconds,
                          "examples_per_second": len(prompts) / max(inference_seconds, 1e-9),
                          "wall_seconds": time.monotonic() - cell_started,
                          "artifact_sha256": {name: sha256(directory / name) for name in
                                              ("predictions.csv", "manifest.json", "demonstrations.json")}}
                atomic_json(directory / "metrics.json", result)
                statuses.append(result)
                atomic_json(output / "summary.json", {"run_identity": run_identity, "cells": statuses})
                progress("cell_completed", cell=cell, metrics=result["metrics"])
        summary = {"run_identity": run_identity, "status": "completed", "cells": statuses}
        atomic_json(output / "summary.json", summary)
        progress("completed", skipped_cells=len(cells) - len(pending))
        return summary
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("./data/simulation/tested"))
    parser.add_argument("--kip-data-dir", type=Path, default=Path("./data/simulation/kip_ordered_recovered"))
    parser.add_argument("--oc-data-dir", type=Path)
    parser.add_argument("--oc-det-tag", default="6")
    parser.add_argument("--oc-noise-tag", default="9")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--shots", default="0,5,20")
    parser.add_argument("--demo-seeds", default="0,1,2")
    parser.add_argument("--test-seed", type=int, default=9950)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.tasks = list(dict.fromkeys(x.strip() for x in args.tasks.split(",") if x.strip()))
    try:
        args.shots = sorted(set(int(x) for x in args.shots.split(",")))
        args.demo_seeds = sorted(set(int(x) for x in args.demo_seeds.split(",")))
    except ValueError:
        parser.error("shots and demo-seeds must be comma-separated integers")
    if not args.tasks or set(args.tasks) - set(TASKS):
        parser.error(f"tasks must be selected from {','.join(TASKS)}")
    if any(x < 0 for x in args.shots + args.demo_seeds) or args.test_seed < 0:
        parser.error("shots and seeds must be nonnegative")
    if args.n_test < 2 or args.batch_size < 1 or args.max_length < 1:
        parser.error("n-test >= 2, batch-size >= 1 and max-length >= 1 are required")
    if not re.fullmatch(r"cuda:\d+", args.device):
        parser.error("device must identify one CUDA device, e.g. cuda:0")
    for tag in (args.oc_det_tag, args.oc_noise_tag):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            parser.error("OC tags must contain only letters, digits, underscores or hyphens")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
