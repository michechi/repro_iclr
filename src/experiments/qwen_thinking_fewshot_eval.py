"""Budgeted thinking followed by binary-label scoring on fixed Qwen test samples."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from src.experiments import qwen_fewshot_eval as core
from src.experiments import qwen38_fewshot_eval as native
from src.experiments.qwen_thinking_generation import generate_reasoning_batch


def build_prompt(tokenizer, examples, query, effort):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": native.INSTRUCTION + "\n\n" + core.build_prompt(examples, query)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort=effort)


def prepare_reference_data(args):
    reference_config = json.loads((args.reference_run / "config.json").read_text())
    if reference_config["model_name"] != args.model_name:
        raise ValueError("reference must use the same checkpoint name")
    if args.n_test > reference_config["n_test"]:
        raise ValueError("requested sample exceeds fixed reference test sample")
    full_args = copy.copy(args)
    full_args.n_test = reference_config["n_test"]
    data, provenance = core.prepare_data(full_args)
    evidence = {"path": str(args.reference_run.resolve()),
                "config_sha256": core.sha256(args.reference_run / "config.json"), "tasks": {}}
    for task in args.tasks:
        sample_path = args.reference_run / "samples" / f"{task}.json"
        sample = json.loads(sample_path.read_text())
        current = data[task]
        if (sample["indices"] != current["test_indices"] or sample["labels"] != current["labels"].tolist()
                or sample["sequences"] != current["queries"]):
            raise ValueError(f"full reference test sample differs: {task}")
        for split in ("train", "test"):
            for kind in ("features", "labels"):
                if provenance[task][split][kind]["sha256"] != reference_config["data"][task][split][kind]["sha256"]:
                    raise ValueError(f"reference input file differs: {task}/{split}/{kind}")
        positions = (list(range(len(current["labels"]))) if args.n_test == len(current["labels"])
                     else core.stratified_test_indices(current["labels"], args.n_test, args.test_seed).tolist())
        current["labels"] = current["labels"][positions]
        current["queries"] = [current["queries"][i] for i in positions]
        current["test_indices"] = [current["test_indices"][i] for i in positions]
        provenance[task].update(test_indices=current["test_indices"],
                                sampled_test_positives=int(current["labels"].sum()))
        evidence["tasks"][task] = {"sample_sha256": core.sha256(sample_path),
                                    "reference_positions": positions,
                                    "full_reference_rows": full_args.n_test,
                                    "demonstration_exclusion_pool": "entire reference test sample"}
    return data, provenance, evidence


def batch_seed(seed, cell_key, start):
    return int(core.identity({"generation_seed": seed, "cell": cell_key, "batch_start": start})[:8], 16)


def check_batch(path, cell_id, expected_indices):
    payload = json.loads(path.read_text())
    if payload.get("cell_identity") != cell_id or [r["test_row_index"] for r in payload["rows"]] != expected_indices:
        raise ValueError(f"saved batch identity differs: {path}")
    if payload.get("rows_sha256") != core.identity(payload["rows"]):
        raise ValueError(f"saved batch payload hash differs: {path}")
    return payload


def completed_cell(directory, cell_id):
    result = core.completed_cell(directory, cell_id)
    if result is None:
        return None
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, digest in manifest["reasoning_batch_sha256"].items():
        if core.sha256(directory / name) != digest:
            raise ValueError(f"completed reasoning batch changed: {directory / name}")
    return result


def evaluate_cell(args, cell, cell_id, run_id, task, provenance, demo, model, tokenizer, runtime, protocol, progress):
    started = time.monotonic()
    directory = args.output_dir / "cells" / cell["key"]
    (directory / "batches").mkdir(parents=True, exist_ok=True)
    core.ensure_json(directory / "demonstrations.json", demo)
    sample_path = args.output_dir / "samples" / f"{cell['task']}.json"
    core.ensure_json(sample_path, {"test_seed": args.test_seed, "source": provenance["test"],
        "indices": task["test_indices"], "sequences": task["queries"], "labels": task["labels"].tolist()})
    examples = [(e["sequence"], e["label"]) for e in demo["examples"]]
    prompts = [build_prompt(tokenizer, examples, query, args.reasoning_effort) for query in task["queries"]]
    ids = [list(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    if any(not p or len(p) + args.max_thinking_tokens + 3 > args.max_length for p in ids):
        raise ValueError("prompt plus full thinking budget and answer reserve exceeds max-length; truncation forbidden")
    if any(not p.endswith("<think>\n") for p in prompts):
        raise ValueError("thinking prompt must have an open thinking block")
    manifest = {"schema_version": 2, "cell_identity": cell_id, "run_identity": run_id, "cell": cell,
        "runtime": runtime, "protocol": protocol, "demo_source": provenance["train"],
        "test_sample": {"path": str(sample_path.resolve()), "sha256": core.sha256(sample_path)},
        "prompt_sha256": core.identity(prompts), "input_ids_sha256": core.identity(ids),
        "sample_rendered_prompt": prompts[0], "token_lengths": {"min": min(map(len, ids)),
            "max": max(map(len, ids)), "all": list(map(len, ids))}, "reasoning_batch_sha256": {}}
    rows = []
    for start in range(0, len(ids), args.batch_size):
        stop = min(start + args.batch_size, len(ids))
        batch_path = directory / "batches" / f"batch_{start:06d}.json"
        seed = batch_seed(args.generation_seed, cell["key"], start)
        if batch_path.exists():
            payload = check_batch(batch_path, cell_id, task["test_indices"][start:stop])
            if payload["generation_seed"] != seed or payload["input_ids_sha256"] != core.identity(ids[start:stop]):
                raise ValueError("saved batch seed/input mismatch")
            if ([r["label"] for r in payload["rows"]] != task["labels"][start:stop].tolist()
                    or [r["thinking"]["prompt_ids"] for r in payload["rows"]] != ids[start:stop]):
                raise ValueError("saved batch labels/prompt tokens differ")
        else:
            progress("thinking", cell=cell, examples_completed=start, examples_total=len(ids),
                     batch_start=start, batch_size=stop-start)
            batch_start = time.monotonic()
            def heartbeat(*values, **details):
                progress("thinking", cell=cell, examples_completed=start, examples_total=len(ids),
                         batch_start=start, batch_size=stop-start, generation_progress=details or list(values),
                         batch_seconds=time.monotonic()-batch_start)
            pending_reasoning = directory / f"pending_reasoning_{start:06d}.json"
            if pending_reasoning.exists():
                draft = json.loads(pending_reasoning.read_text())
                if (draft["cell_identity"] != cell_id or draft["generation_seed"] != seed
                        or draft["input_ids_sha256"] != core.identity(ids[start:stop])
                        or draft["records_sha256"] != core.identity(draft["records"])
                        or [r["prompt_ids"] for r in draft["records"]] != ids[start:stop]):
                    raise ValueError("pending reasoning checkpoint differs from this batch")
                reasoning, generation_seconds = draft["records"], draft["generation_seconds"]
            else:
                reasoning = generate_reasoning_batch(model, tokenizer, ids[start:stop], args.device,
                                                    args.max_thinking_tokens, seed, progress=heartbeat)
                generation_seconds = time.monotonic() - batch_start
                core.atomic_json(pending_reasoning, {"cell_identity": cell_id, "generation_seed": seed,
                    "input_ids_sha256": core.identity(ids[start:stop]), "records": reasoning,
                    "records_sha256": core.identity(reasoning), "generation_seconds": generation_seconds})
            scoring_ids = [r["scoring_prefix_ids"] for r in reasoning]
            if any(len(p) >= args.max_length for p in scoring_ids):
                raise ValueError("actual reasoning scoring prefix exceeds reserved context")
            progress("scoring", cell=cell, examples_completed=start, examples_total=len(ids), batch_start=start)
            score_start = time.monotonic()
            pieces = [core.score_batch(model, scoring_ids[k:k+args.score_batch_size],
                        [[15, 16]] * len(scoring_ids[k:k+args.score_batch_size]), tokenizer.pad_token_id, args.device)
                      for k in range(0, len(scoring_ids), args.score_batch_size)]
            logits, odds, probs = [np.concatenate([piece[k] for piece in pieces]) for k in range(3)]
            batch_rows = []
            for i, record in enumerate(reasoning):
                j = start + i
                batch_rows.append({"test_row_index": task["test_indices"][j], "label": int(task["labels"][j]),
                    "probability_1": float(probs[i]), "prediction_at_0_5": int(probs[i] >= .5),
                    "logit_0": float(logits[i, 0]), "logit_1": float(logits[i, 1]),
                    "log_odds_1_minus_0": float(odds[i]), "prompt_tokens": len(ids[j]),
                    "scoring_tokens": len(scoring_ids[i]), "thinking": record})
            payload = {"schema_version": 1, "cell_identity": cell_id, "generation_seed": seed,
                       "input_ids_sha256": core.identity(ids[start:stop]), "rows": batch_rows,
                       "rows_sha256": core.identity(batch_rows), "generation_seconds": generation_seconds,
                       "scoring_seconds": time.monotonic()-score_start, "batch_start": start}
            core.atomic_json(batch_path, payload)
            pending_reasoning.unlink()  # Its exact records are now in the durable completed batch.
        rows.extend(payload["rows"])
        manifest["reasoning_batch_sha256"][str(batch_path.relative_to(directory))] = core.sha256(batch_path)
        progress("batch_completed", cell=cell, examples_completed=len(rows), examples_total=len(ids),
                 thinking_completed=sum(r["thinking"]["thinking_completed"] for r in rows),
                 forced_close=sum(r["thinking"]["forced_close"] for r in rows))
    columns = ["test_row_index", "label", "probability_1", "prediction_at_0_5", "logit_0", "logit_1",
               "log_odds_1_minus_0", "prompt_tokens", "scoring_tokens", "reasoning_token_count",
               "thinking_completed", "forced_close", "stop_reason"]
    partial = directory / "predictions.partial.csv"
    with partial.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in columns if k in row}
            flat.update({k: row["thinking"][k] for k in columns if k in row["thinking"]})
            writer.writerow(flat)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(directory / "predictions.csv")
    core.atomic_json(directory / "manifest.json", manifest)
    counts = {reason: sum(r["thinking"]["stop_reason"] == reason for r in rows)
              for reason in sorted({r["thinking"]["stop_reason"] for r in rows})}
    result = {"status": "completed", "cell": cell, "cell_identity": cell_id, "run_identity": run_id,
        "metrics": core.metrics(task["labels"], [r["probability_1"] for r in rows], [r["log_odds_1_minus_0"] for r in rows]),
        "thinking": {"stop_counts": counts, "forced_close": sum(r["thinking"]["forced_close"] for r in rows),
            "natural_completion": sum(r["thinking"]["thinking_completed"] for r in rows),
            "reasoning_tokens_mean": float(np.mean([r["thinking"]["reasoning_token_count"] for r in rows]))},
        "wall_seconds": time.monotonic()-started,
        "artifact_sha256": {name: core.sha256(directory / name) for name in
            ("predictions.csv", "manifest.json", "demonstrations.json")}}
    core.atomic_json(directory / "metrics.json", result)
    return result


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args)


def run_locked(args):
    started = time.monotonic()
    data, provenance, reference = prepare_reference_data(args)
    cells = core.build_cells(args.tasks, args.shots, args.demo_seeds)
    demos = {c["key"]: core.demonstration_payload(data[c["task"]], c) for c in cells}
    source_files = [Path(__file__), Path(core.__file__), Path(native.__file__),
                    Path(__file__).with_name("qwen_thinking_generation.py")]
    protocol = {"prompt_mode": "official_chat_budgeted_thinking", "instruction": native.INSTRUCTION,
        "enable_thinking": True, "reasoning_effort": args.reasoning_effort,
        "max_thinking_tokens": args.max_thinking_tokens, "fine_tuning": False,
        "temperature": 1.0, "top_p": .95, "top_k": 20, "min_p": 0., "repetition_penalty": 1.,
        "generation_seed": args.generation_seed, "rng_seeding": "SHA256(generation_seed,cell_key,batch_start), first32bits",
        "answer_scoring": "raw_unwarped_label_logits_after_generated_reasoning_and_canonical_close_newlines",
        "probability_scope": "conditional_on_one_sampled_reasoning_trace_and_two_label_tokens",
        "cap_policy": "force closing think token, retain all queries, report forced completion explicitly",
        "early_eos_policy": "record early EOS, remove terminal EOS, force closing think, retain all queries",
        "source_code_sha256": {p.name: core.sha256(p) for p in source_files}}
    config = {"schema_version": 2, "model_name": args.model_name, "tasks": args.tasks, "shots": args.shots,
        "demo_seeds": args.demo_seeds, "test_seed": args.test_seed, "n_test": args.n_test,
        "batch_size": args.batch_size, "score_batch_size": args.score_batch_size,
        "max_length": args.max_length, "device": args.device,
        "cells": cells, "data": provenance, "reference_check": reference, "protocol": protocol,
        "auc_score": "raw_float32_log_odds_1_minus_0", "threshold": .5, "versions": native.versions()}
    if args.dry_run:
        core.atomic_json(args.output_dir / "dry_run.json", {"status": "data_validated", "config": config,
                         "demonstrations": demos, "weights_loaded": False, "tokenization_checked": False})
        print(json.dumps({"status": "data_validated", "cells": len(cells), "n_test": args.n_test}), flush=True)
        return
    tokenizer = native.load_tokenizer(args)
    config["model_source"] = native.model_source(args.model_path, args.weight_hash_manifest)
    reference_config = json.loads((args.reference_run / "config.json").read_text())
    if config["model_source"] != reference_config["model_source"]:
        raise ValueError("checkpoint content identity differs from reference")
    protocol["chat_template_sha256"] = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    protocol["chat_template"] = tokenizer.chat_template
    core.ensure_json(args.output_dir / "config.json", config)
    run_id = core.identity(config)
    completed, pending = [], []
    for cell in cells:
        cell_id = core.identity({"run_identity": run_id, "cell": cell, "demonstrations": demos[cell["key"]]})
        existing = completed_cell(args.output_dir / "cells" / cell["key"], cell_id)
        if existing:
            completed.append(existing)
        else:
            pending.append((cell, cell_id))
    def progress(status, **details):
        record = {"status": status, "pid": os.getpid(), "run_identity": run_id,
                  "updated_at_unix": time.time(), "wall_seconds": time.monotonic()-started,
                  "completed_cells": len(completed), "total_cells": len(cells), **details}
        core.atomic_json(args.output_dir / "status.json", record)
        print(json.dumps(record), flush=True)
    try:
        if pending:
            progress("loading_model")
            model_start = time.monotonic()
            model, runtime = native.load_model(args, tokenizer)
            runtime.update(model_load_seconds=time.monotonic()-model_start, generation_use_cache=True,
                           generation_batch_size=args.batch_size, reasoning_effort=args.reasoning_effort)
            core.atomic_json(args.output_dir / "runtime.json", runtime)
            for cell, cell_id in pending:
                result = evaluate_cell(args, cell, cell_id, run_id, data[cell["task"]], provenance[cell["task"]],
                    demos[cell["key"]], model, tokenizer, runtime, protocol, progress)
                completed.append(result)
                core.atomic_json(args.output_dir / "summary.json", {"run_identity": run_id, "cells": completed})
                progress("cell_completed", cell=cell, metrics=result["metrics"], thinking=result["thinking"])
        core.atomic_json(args.output_dir / "summary.json", {"status": "completed", "run_identity": run_id, "cells": completed})
        progress("completed")
    except BaseException as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


def parse_args(argv=None):
    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument("--reference-run", type=Path, required=True)
    extension.add_argument("--weight-hash-manifest", type=Path, required=True)
    extension.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    extension.add_argument("--max-thinking-tokens", type=int, default=4096)
    extension.add_argument("--generation-seed", type=int, default=9951)
    extension.add_argument("--score-batch-size", type=int, default=4)
    extra, remaining = extension.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not any(v == "--model-name" or v.startswith("--model-name=") for v in remaining):
        remaining += ["--model-name", native.MODEL_NAME]
    args = core.parse_args(remaining)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    if args.max_thinking_tokens < 1 or args.generation_seed < 0 or args.score_batch_size < 1 or not args.shots:
        extension.error("positive thinking budget, nonnegative generation seed and nonempty shots required")
    return args


if __name__ == "__main__":
    run(parse_args())
