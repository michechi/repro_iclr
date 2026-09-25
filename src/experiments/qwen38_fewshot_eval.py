"""Qwen3.8 nonthinking-chat label scoring, without changing the Qwen2.5 protocol."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from src.experiments import qwen_fewshot_eval as core


INSTRUCTION = ("Predict the binary outcome for the final sequence. Use the labeled examples when provided. "
               "Reply with exactly one digit, 0 or 1, and no other text.")
MODEL_NAME = "Qwen/Qwen3.8-27B"


def build_chat_prompt(tokenizer, demonstrations, query):
    content = INSTRUCTION + "\n\n" + core.build_prompt(demonstrations, query)
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)


def tokenize_prompts(tokenizer, prompts, max_length):
    ids, label_ids = [], []
    for row, prompt in enumerate(prompts):
        prefix = list(tokenizer.encode(prompt, add_special_tokens=False))
        if not prefix or len(prefix) > max_length:
            raise ValueError(f"row {row}: {len(prefix)} tokens exceed 1..{max_length}; truncation forbidden")
        labels = []
        for digit in ("0", "1"):
            full = list(tokenizer.encode(prompt + digit, add_special_tokens=False))
            if full[:len(prefix)] != prefix or len(full) != len(prefix) + 1:
                raise ValueError(f"row {row}: label {digit} requires exact prompt prefix plus one token")
            labels.append(int(full[-1]))
        if labels[0] == labels[1]:
            raise ValueError("label continuation tokens must be distinct")
        ids.append(prefix)
        label_ids.append(labels)
    return ids, label_ids


def model_source(path: Path, hash_manifest: Path | None = None):
    """Identify content without mtimes or RAM-cache location in the run identity.

    HF cache blob names supply expected SHA256 identities, not proof that this
    process hashed the payload. An optional external audit has format
    {"files": {"model-....safetensors": {"sha256": "...", "bytes": 123}}}.
    Non-content-addressed local files are hashed here unless externally audited.
    """
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    required = sorted(set(index["weight_map"].values()))
    audit = json.loads(hash_manifest.read_text())["files"] if hash_manifest else {}
    weights, verification = {}, {}
    for name in required:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("invalid shard name in model index")
        shard = path / name
        if not shard.is_file():
            raise FileNotFoundError(f"missing model shard: {shard}")
        blob = shard.resolve().name
        expected = blob if re.fullmatch(r"[a-f0-9]{64}", blob) else None
        size = shard.stat().st_size
        if hash_manifest:
            record = audit.get(name, {})
            digest = record.get("sha256", "")
            if not re.fullmatch(r"[a-f0-9]{64}", digest) or record.get("bytes") != size:
                raise ValueError(f"invalid external hash record for {name}")
            if expected and digest != expected:
                raise ValueError(f"external SHA256 disagrees with HF blob identity: {name}")
            evidence = "external_sha256_audit; evaluator checked blob identity and byte count"
        elif expected:
            digest, evidence = expected, "HF_blob_name_expected_sha256; payload_not_hashed_by_evaluator"
        else:
            digest, evidence = core.sha256(shard), "payload_sha256_computed_by_evaluator"
        weights[name] = {"sha256": digest, "bytes": size}
        verification[name] = evidence
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "special_tokens_map.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
             "generation_config.json")
    return {"snapshot_revision": path.name if re.fullmatch(r"[a-f0-9]{40}", path.name) else None,
            "metadata_sha256": {n: core.sha256(path / n) for n in names if (path / n).is_file()},
            "weight_files": weights, "weight_verification": verification,
            "external_hash_manifest_sha256": core.sha256(hash_manifest) if hash_manifest else None,
            "ignored_auxiliary_checkpoint_keys": sorted(k for k in index["weight_map"] if k.startswith("mtp."))}


def validate_loading_info(info):
    for key in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        if info.get(key):
            raise ValueError(f"model loading failed: {key}={info[key]}")
    unexpected = sorted(info.get("unexpected_keys", []))
    if any(not key.startswith("mtp.") for key in unexpected):
        raise ValueError(f"unexpected non-MTP checkpoint keys: {unexpected}")
    return {"missing_keys": [], "mismatched_keys": [], "error_msgs": [], "unexpected_keys": unexpected}


def versions():
    result = core.versions()
    for name in ("fla-core", "einops", "triton", "causal-conv1d"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def kernel_metadata(model, architecture):
    """Record actual layer bindings separately from the all-or-nothing fast flag."""
    def binding(value):
        if value is None:
            return None
        return f"{value.__module__}.{getattr(value, '__qualname__', type(value).__qualname__)}"

    layers = [m for m in model.modules() if isinstance(m, architecture.Qwen3_5GatedDeltaNet)]
    fused_norm = architecture.FusedRMSNormGated
    result = {"linear_attention_layers": len(layers),
              "all_optional_fast_path_available": bool(architecture.is_fast_path_available),
              "causal_conv1d_available": architecture.causal_conv1d_fn is not None,
              "flash_linear_attention_available": architecture.chunk_gated_delta_rule is not None,
              "fused_gated_norm_available": fused_norm is not None,
              "fused_gated_norm_layers": sum(fused_norm is not None and isinstance(m.norm, fused_norm) for m in layers),
              "torch_causal_conv1d_prefill_layers": sum(m.causal_conv1d_fn is None for m in layers),
              "gated_norm_bindings": sorted({binding(type(m.norm)) for m in layers})}
    for attribute in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
        result[attribute + "_bindings"] = sorted({binding(getattr(m, attribute)) or "torch.nn.Conv1d (prefill)"
                                                  for m in layers})
    return result


def load_tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True,
                                               trust_remote_code=False, padding_side="left")
    if not tokenizer.chat_template or not isinstance(tokenizer.chat_template, str):
        raise ValueError("one explicit official tokenizer chat template is required")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs an EOS or pad token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(args, tokenizer):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5 as architecture
    if not torch.cuda.is_available():
        raise RuntimeError("a BF16-capable CUDA device is required")
    torch.cuda.set_device(torch.device(args.device))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected GPU does not support BF16")
    torch.manual_seed(args.test_seed)
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.bfloat16, device_map={"": args.device}, attn_implementation="sdpa",
        output_loading_info=True)
    loading = validate_loading_info(info)
    if model.config.model_type != "qwen3_5":
        raise ValueError("expected qwen3_5 checkpoint architecture")
    if any(p.is_meta for p in model.parameters()):
        raise ValueError("model contains unloaded meta parameters")
    model.eval().requires_grad_(False)
    model.config.text_config.use_cache = False
    context = model.config.text_config.max_position_embeddings
    if args.max_length > context:
        raise ValueError("requested max-length exceeds model context")
    return model, {"versions": versions(), "device": args.device,
                   "gpu": torch.cuda.get_device_name(torch.device(args.device)),
                   "cuda_build": torch.version.cuda, "parameter_dtype": "bfloat16", "score_dtype": "float32",
                   "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": 0,
                   "model_type": model.config.model_type, "model_class": type(model).__name__,
                   "model_path": str(args.model_path.resolve()), "model_revision": args.model_path.name,
                   "tokenizer_class": type(tokenizer).__name__, "pad_token_id": tokenizer.pad_token_id,
                   "eos_token_id": tokenizer.eos_token_id, "max_position_embeddings": context,
                   "loading_info": loading, "deltanet_fast_path": bool(architecture.is_fast_path_available),
                   "causal_conv1d": architecture.causal_conv1d_fn is not None,
                   "flash_linear_attention": architecture.chunk_gated_delta_rule is not None,
                   "fused_gated_norm_available": architecture.FusedRMSNormGated is not None,
                   "kernel_bindings": kernel_metadata(model, architecture)}


def check_reference(reference, data, provenance, cells, demos):
    if reference is None:
        return {"enabled": False}
    ref_config_path = reference / "config.json"
    ref_config = json.loads(ref_config_path.read_text())
    matched, absent = [], []
    for task, current in data.items():
        sample_path = reference / "samples" / f"{task}.json"
        sample = json.loads(sample_path.read_text())
        for field, expected in (("indices", current["test_indices"]), ("labels", current["labels"].tolist()),
                                ("sequences", current["queries"])):
            if sample[field] != expected:
                raise ValueError(f"reference test {field} differ for {task}")
        for split in ("train", "test"):
            for kind in ("features", "labels"):
                if ref_config["data"][task][split][kind]["sha256"] != provenance[task][split][kind]["sha256"]:
                    raise ValueError(f"reference data content differs: {task}/{split}/{kind}")
    for cell in cells:
        directory = reference / "cells" / cell["key"]
        metric_path = directory / "metrics.json"
        if not metric_path.is_file():
            absent.append(cell["key"])
            continue
        recorded = json.loads(metric_path.read_text())
        validated = core.completed_cell(directory, recorded["cell_identity"])
        if validated["run_identity"] != core.identity(ref_config) or validated["cell"] != cell:
            raise ValueError(f"reference cell/run identity mismatch: {cell['key']}")
        if json.loads((directory / "demonstrations.json").read_text()) != demos[cell["key"]]:
            raise ValueError(f"reference demonstration indices, labels, text or order differ: {cell['key']}")
        matched.append({"cell": cell["key"], "metrics_sha256": core.sha256(metric_path)})
    if not matched:
        raise ValueError("reference run has no matching completed cells")
    return {"enabled": True, "path": str(reference.resolve()), "config_sha256": core.sha256(ref_config_path),
            "matched_completed_cells": matched, "reference_cells_not_completed": absent}


def evaluate_cell(args, cell, cell_id, run_id, task_data, provenance, demo, model, tokenizer,
                  runtime, protocol, progress):
    started = time.monotonic()
    directory = args.output_dir / "cells" / cell["key"]
    directory.mkdir(parents=True, exist_ok=True)
    core.ensure_json(directory / "demonstrations.json", demo)
    sample_path = args.output_dir / "samples" / f"{cell['task']}.json"
    core.ensure_json(sample_path, {"test_seed": args.test_seed, "source": provenance["test"],
                     "indices": task_data["test_indices"], "sequences": task_data["queries"],
                     "labels": task_data["labels"].tolist()})
    examples = [(x["sequence"], x["label"]) for x in demo["examples"]]
    progress("tokenizing", cell=cell)
    token_start = time.monotonic()
    prompts = [build_chat_prompt(tokenizer, examples, query) for query in task_data["queries"]]
    token_ids, label_ids = tokenize_prompts(tokenizer, prompts, args.max_length)
    token_seconds = time.monotonic() - token_start
    lengths = list(map(len, token_ids))
    manifest = {"schema_version": 1, "cell_identity": cell_id, "run_identity": run_id, "cell": cell,
                "runtime": runtime, "protocol": protocol,
                "test_sample": {"path": str(sample_path.resolve()), "sha256": core.sha256(sample_path)},
                "demo_source": provenance["train"], "prompt_sha256": core.identity(prompts),
                "input_ids_sha256": core.identity(token_ids), "sample_rendered_prompt": prompts[0],
                "token_lengths": {"min": min(lengths), "max": max(lengths), "mean": float(np.mean(lengths)), "all": lengths},
                "label_token_ids_by_row": label_ids, "tokenization_seconds": token_seconds}
    core.atomic_json(directory / "manifest.json", manifest)
    inference_start = time.monotonic()
    probabilities, scores = [], []
    partial = directory / "predictions.partial.csv"
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_row_index", "label", "probability_1", "prediction_at_0_5", "logit_0",
                         "logit_1", "log_odds_1_minus_0", "prompt_tokens"])
        for start in range(0, len(prompts), args.batch_size):
            stop = min(start + args.batch_size, len(prompts))
            logits, odds, probability = core.score_batch(model, token_ids[start:stop], label_ids[start:stop],
                                                         tokenizer.pad_token_id, args.device)
            probabilities.extend(float(x) for x in probability)
            scores.extend(float(x) for x in odds)
            for j, p in enumerate(probability):
                row = start + j
                writer.writerow([task_data["test_indices"][row], int(task_data["labels"][row]), float(p),
                                 int(p >= .5), float(logits[j, 0]), float(logits[j, 1]), float(odds[j]), lengths[row]])
            handle.flush()
            progress("evaluating", cell=cell, examples_completed=stop, examples_total=len(prompts),
                     inference_seconds=time.monotonic() - inference_start)
        os.fsync(handle.fileno())
    seconds = time.monotonic() - inference_start
    partial.replace(directory / "predictions.csv")
    result = {"status": "completed", "cell": cell, "cell_identity": cell_id, "run_identity": run_id,
              "metrics": core.metrics(task_data["labels"], probabilities, scores), "tokenization_seconds": token_seconds,
              "inference_seconds": seconds, "examples_per_second": len(prompts) / max(seconds, 1e-9),
              "wall_seconds": time.monotonic() - started,
              "artifact_sha256": {name: core.sha256(directory / name) for name in
                                  ("predictions.csv", "manifest.json", "demonstrations.json")}}
    core.atomic_json(directory / "metrics.json", result)
    return result


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another evaluator holds this output lock") from error
        return _run_locked(args)


def _run_locked(args):
    started = time.monotonic()
    data, provenance = core.prepare_data(args)
    cells = core.build_cells(args.tasks, args.shots, args.demo_seeds)
    demos = {c["key"]: core.demonstration_payload(data[c["task"]], c) for c in cells}
    reference = check_reference(args.reference_run, data, provenance, cells, demos)
    protocol = {"prompt_mode": "official_chat_nonthinking", "instruction": INSTRUCTION,
                "enable_thinking": False, "chat_template_applied": True, "generated_reasoning_tokens": 0,
                "add_special_tokens": False, "add_generation_prompt": True, "fine_tuning": False,
                "model_training_stage": "posttrained", "input_modality": "text_only",
                "source_code_sha256": {"qwen38_fewshot_eval.py": core.sha256(Path(__file__)),
                                       "qwen_fewshot_eval.py": core.sha256(Path(core.__file__))}}
    config = {"schema_version": 1, "model_name": args.model_name, "tasks": args.tasks, "shots": args.shots,
              "demo_seeds": args.demo_seeds, "test_seed": args.test_seed, "n_test": args.n_test,
              "batch_size": args.batch_size, "max_length": args.max_length, "device": args.device,
              "data": provenance, "cells": cells, "protocol": protocol, "reference_check": reference,
              "auc_score": "raw_float32_log_odds_1_minus_0", "threshold": .5}
    if args.dry_run:
        result = {"status": "data_validated", "config": config, "demonstrations": demos,
                  "tokenization_checked": False, "weights_loaded": False,
                  "note": "Only data, sampling and reference correspondence checked; model files may be incomplete."}
        core.atomic_json(args.output_dir / "dry_run.json", result)
        print(json.dumps({"status": "data_validated", "cells": len(cells), "reference": reference}), flush=True)
        return result
    tokenizer = load_tokenizer(args)
    protocol["chat_template"] = tokenizer.chat_template
    protocol["chat_template_sha256"] = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    first = cells[0]
    protocol["sample_rendered_prompt"] = build_chat_prompt(tokenizer,
        [(e["sequence"], e["label"]) for e in demos[first["key"]]["examples"]], data[first["task"]]["queries"][0])
    tokenize_prompts(tokenizer, [protocol["sample_rendered_prompt"]], args.max_length)
    config.update(model_source=model_source(args.model_path, args.weight_hash_manifest), versions=versions())
    config_path = args.output_dir / "config.json"
    if not config_path.exists() and any(p.name not in {".run.lock", "dry_run.json"} for p in args.output_dir.iterdir()):
        raise FileExistsError("nonempty output directory has no matching configuration")
    core.ensure_json(config_path, config)
    run_id, completed, pending = core.identity(config), [], []
    for cell in cells:
        cell_id = core.identity({"run_identity": run_id, "cell": cell, "demonstrations": demos[cell["key"]]})
        existing = core.completed_cell(args.output_dir / "cells" / cell["key"], cell_id)
        if existing:
            completed.append(existing)
        else:
            pending.append((cell, cell_id))

    def progress(status, **values):
        payload = {"status": status, "pid": os.getpid(), "run_identity": run_id, "updated_at_unix": time.time(),
                   "completed_cells": len(completed), "total_cells": len(cells),
                   "wall_seconds": time.monotonic() - started, **values}
        core.atomic_json(args.output_dir / "status.json", payload)
        print(json.dumps(payload), flush=True)

    try:
        if pending:
            progress("loading_model")
            model_start = time.monotonic()
            model, runtime = load_model(args, tokenizer)
            runtime["model_load_seconds"] = time.monotonic() - model_start
            core.atomic_json(args.output_dir / "runtime.json", runtime)
            for cell, cell_id in pending:
                result = evaluate_cell(args, cell, cell_id, run_id, data[cell["task"]], provenance[cell["task"]],
                                       demos[cell["key"]], model, tokenizer, runtime, protocol, progress)
                completed.append(result)
                core.atomic_json(args.output_dir / "summary.json", {"run_identity": run_id, "cells": completed})
                progress("cell_completed", cell=cell, metrics=result["metrics"])
        summary = {"status": "completed", "run_identity": run_id, "cells": completed}
        core.atomic_json(args.output_dir / "summary.json", summary)
        progress("completed", skipped_cells=len(cells) - len(pending))
        return summary
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


def parse_args(argv=None):
    # Reuse the unchanged core CLI validation, consuming only this protocol's additions.
    argv = list(sys.argv[1:] if argv is None else argv)
    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument("--reference-run", type=Path)
    extension.add_argument("--weight-hash-manifest", type=Path)
    extra, remaining = extension.parse_known_args(argv)
    if not any(x == "--model-name" or x.startswith("--model-name=") for x in remaining):
        remaining += ["--model-name", MODEL_NAME]
    args = core.parse_args(remaining)
    args.reference_run, args.weight_hash_manifest = extra.reference_run, extra.weight_hash_manifest
    return args


if __name__ == "__main__":
    run(parse_args())
