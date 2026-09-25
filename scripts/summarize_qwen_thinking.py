#!/usr/bin/env python3
"""Independently audit a partial or complete 40-shot budgeted-thinking run.

The prior nonthinking analysis is reverified, and its predictions are restricted
to the exact new test subset before comparisons. No model inference is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

if __package__:
    from . import compare_qwen_fewshot as comparison
    from . import summarize_qwen_fewshot as audit
else:
    import compare_qwen_fewshot as comparison
    import summarize_qwen_fewshot as audit

STOPS = {248044, 248046, 248069}
PREDICTION_COLUMNS = ["test_row_index", "label", "probability_1", "prediction_at_0_5", "logit_0", "logit_1",
                      "log_odds_1_minus_0", "prompt_tokens", "scoring_tokens", "reasoning_token_count",
                      "thinking_completed", "forced_close", "stop_reason"]


def validate_config(config):
    tasks, seeds = config.get("tasks", []), config.get("demo_seeds", [])
    audit.require(tasks and len(tasks) == len(set(tasks)) and set(tasks) <= set(audit.TASKS), "invalid task subset")
    audit.require(seeds and len(seeds) == len(set(seeds)) and all(type(s) is int and s >= 0 for s in seeds), "invalid demo seeds")
    audit.require(config.get("shots") == [40], "expected only 40-shot thinking cells")
    expected = [{"task": t, "shots": 40, "demo_seed": s, "key": f"{t}/shots40/demo_seed{s}"}
                for t in tasks for s in seeds]
    audit.require(config.get("cells") == expected, "thinking grid differs from configuration")
    p = config.get("protocol", {})
    audit.require(config.get("model_name") == "Qwen/Qwen3.8-27B" and p.get("fine_tuning") is False,
                  "expected inference-only Qwen3.8 checkpoint")
    audit.require(p.get("prompt_mode") == "official_chat_budgeted_thinking" and p.get("enable_thinking") is True,
                  "expected budgeted thinking protocol")
    audit.require(p.get("reasoning_effort") in {"low", "medium", "xhigh"}, "invalid reasoning effort")
    audit.require(p.get("answer_scoring") == "raw_unwarped_label_logits_after_generated_reasoning_and_canonical_close_newlines" and
                  p.get("probability_scope") == "conditional_on_one_sampled_reasoning_trace_and_two_label_tokens",
                  "unsupported thinking scoring protocol")
    audit.require(config.get("auc_score") == "raw_float32_log_odds_1_minus_0" and config.get("threshold") == .5,
                  "expected raw-log-odds AUROC and fixed threshold")
    audit.require(all(type(config.get(k)) is int and config[k] > 0 for k in ("n_test", "batch_size", "max_length")) and
                  config["n_test"] >= 2 and type(p.get("max_thinking_tokens")) is int and p["max_thinking_tokens"] > 0,
                  "invalid sample/context/budget sizes")
    audit.require(all(p.get(k) == value for k, value in {"temperature": 1., "top_p": .95, "top_k": 20,
                  "min_p": 0., "repetition_penalty": 1.}.items()), "unexpected reasoning sampling parameters")
    return expected


def expected_seed(config, cell_key, start):
    value = {"generation_seed": config["protocol"]["generation_seed"], "cell": cell_key, "batch_start": start}
    return int(audit.identity(value)[:8], 16)


def verify_trace(record, config, seed):
    """Check the archived token sequence, including forced-close cases, without decoding it."""
    for field in ("prompt_ids", "generated_ids", "reasoning_ids", "scoring_prefix_ids"):
        ids = record.get(field)
        audit.require(isinstance(ids, list) and all(type(i) is int and i >= 0 for i in ids), f"invalid {field}")
    prompt, generated, reasoning = (record[k] for k in ("prompt_ids", "generated_ids", "reasoning_ids"))
    budget = config["protocol"]["max_thinking_tokens"]
    audit.require(prompt and generated and len(generated) <= budget and len(prompt) + budget + 3 <= config["max_length"],
                  "trace violates prompt/budget/context limits")
    audit.require(record["max_new_tokens"] == budget and record["batch_seed"] == seed, "trace budget/seed differs")
    audit.require(record["generated_token_count"] == len(generated) and record["reasoning_token_count"] == len(reasoning),
                  "trace token counts differ")
    reason = record.get("stop_reason")
    audit.require(reason in {"thinking_completed", "token_cap", "early_eos"}, "unknown stop reason")
    if reason == "token_cap":
        audit.require(len(generated) == budget and not (set(generated) & STOPS) and generated == reasoning and
                      record["stop_token_id"] is None, "invalid token-cap trace")
    else:
        allowed = {248069} if reason == "thinking_completed" else STOPS - {248069}
        audit.require(generated[-1] in allowed and not (set(generated[:-1]) & STOPS) and
                      reasoning == generated[:-1] and record["stop_token_id"] == generated[-1], "invalid first-stop trace")
    natural = reason == "thinking_completed"
    audit.require(record["thinking_completed"] is natural and record["forced_close"] is (not natural) and
                  record["forced_close_reason"] == (None if natural else reason), "termination flags differ")
    audit.require(record["scoring_prefix_ids"] == prompt + reasoning + [248069, 271] and record["label_ids"] == [15, 16],
                  "scoring prefix does not preserve exact reasoning IDs and label boundary")
    audit.require(len(record["scoring_prefix_ids"]) < config["max_length"], "scoring context exceeded")


def verify_batches(directory, cell, config, base_row):
    manifest = audit.read_json(directory / "manifest.json")
    saved = audit.read_json(directory / "metrics.json")
    sample = audit.read_json(Path(manifest["test_sample"]["path"]))
    audit.require(manifest.get("protocol") == config["protocol"], "cell protocol differs from run")
    starts = list(range(0, config["n_test"], config["batch_size"]))
    names = [f"batches/batch_{start:06d}.json" for start in starts]
    inventory = manifest.get("reasoning_batch_sha256", {})
    audit.require(set(inventory) == set(names), "reasoning batch inventory differs")
    audit.require({str(p.relative_to(directory)) for p in (directory / "batches").glob("*.json")} == set(names),
                  "unexpected reasoning batch files")
    rows, prompts, batch_evidence, generation_seconds, scoring_seconds = [], [], [], 0., 0.
    for start, name in zip(starts, names):
        path = directory / name
        digest = audit.digest_file(path)
        audit.require(digest == inventory[name], f"changed reasoning batch: {path}")
        payload = audit.read_json(path)
        seed = expected_seed(config, cell["key"], start)
        audit.require(payload["cell_identity"] == saved["cell_identity"] and payload["batch_start"] == start and
                      payload["generation_seed"] == seed, "batch identity/start/seed differs")
        batch_rows = payload["rows"]
        stop = min(start + config["batch_size"], config["n_test"])
        audit.require([r["test_row_index"] for r in batch_rows] == sample["indices"][start:stop] and
                      [r["label"] for r in batch_rows] == sample["labels"][start:stop], "batch rows/labels differ from fixed sample")
        audit.require(payload["rows_sha256"] == audit.identity(batch_rows), "batch row hash differs")
        batch_prompts = []
        for row in batch_rows:
            trace = row["thinking"]
            verify_trace(trace, config, seed)
            audit.require(row["prompt_tokens"] == len(trace["prompt_ids"]) and
                          row["scoring_tokens"] == len(trace["scoring_prefix_ids"]), "saved prompt/scoring length differs")
            batch_prompts.append(trace["prompt_ids"])
        audit.require(payload["input_ids_sha256"] == audit.identity(batch_prompts), "batch prompt IDs hash differs")
        prompts.extend(batch_prompts)
        rows.extend(batch_rows)
        for key in ("generation_seconds", "scoring_seconds"):
            audit.require(isinstance(payload[key], (float, int)) and math.isfinite(payload[key]) and payload[key] >= 0,
                          "invalid batch timing")
        generation_seconds += payload["generation_seconds"]
        scoring_seconds += payload["scoring_seconds"]
        batch_evidence.append({"path": str(path.resolve()), "sha256": digest, "rows_sha256": payload["rows_sha256"]})
    lengths = list(map(len, prompts))
    audit.require(manifest["input_ids_sha256"] == audit.identity(prompts) and manifest["token_lengths"]["all"] == lengths and
                  manifest["token_lengths"]["min"] == min(lengths) and manifest["token_lengths"]["max"] == max(lengths),
                  "manifest prompt IDs/lengths differ from batches")
    flat = [{key: row[key] if key in row else str(row["thinking"][key]) if type(row["thinking"][key]) is bool
             else row["thinking"][key] for key in PREDICTION_COLUMNS} for row in rows]
    comparison.check_csv(directory / "predictions.csv", flat, ("test_row_index",))
    counts = {reason: sum(r["thinking"]["stop_reason"] == reason for r in rows)
              for reason in sorted({r["thinking"]["stop_reason"] for r in rows})}
    lengths = [r["thinking"]["reasoning_token_count"] for r in rows]
    natural, forced = counts.get("thinking_completed", 0), counts.get("token_cap", 0) + counts.get("early_eos", 0)
    expected = {"stop_counts": counts, "forced_close": forced, "natural_completion": natural,
                "reasoning_tokens_mean": statistics.mean(lengths)}
    audit.require(saved["thinking"] == expected, "cell reasoning summary differs from archived traces")
    return {**base_row, "condition": "thinking_40", "source_run": str(directory.parents[3]),
            "metric_scope": "new_fixed_test_sample", "natural_completion": natural, "token_cap": counts.get("token_cap", 0),
            "early_eos": counts.get("early_eos", 0), "forced_close": forced, "natural_completion_fraction": natural / len(rows),
            "reasoning_tokens_mean": statistics.mean(lengths), "reasoning_tokens_max": max(lengths),
            "reasoning_tokens_total": sum(lengths), "generation_seconds": generation_seconds, "scoring_seconds": scoring_seconds}, batch_evidence


def verify_reference(config, baseline):
    old = baseline["config"]
    ref = config["reference_check"]
    audit.require(Path(ref["path"]).resolve() == baseline["run"] and ref["config_sha256"] == baseline["evidence"]["config_sha256"],
                  "baseline analysis does not match the run reference")
    audit.require(config["model_name"] == old["model_name"] and config["model_source"] == old["model_source"],
                  "checkpoint content identity differs from baseline")
    audit.require(old["protocol"]["prompt_mode"] == "official_chat_nonthinking" and old["protocol"]["enable_thinking"] is False,
                  "expected prior nonthinking chat baseline")
    for key in ("chat_template", "chat_template_sha256", "instruction"):
        audit.require(config["protocol"][key] == old["protocol"][key], f"shared prompt component differs: {key}")
    for name in ("qwen_fewshot_eval.py", "qwen38_fewshot_eval.py"):
        audit.require(config["protocol"]["source_code_sha256"][name] == old["protocol"]["source_code_sha256"][name],
                      f"shared evaluator source differs: {name}")
    audit.require(config["test_seed"] == old["test_seed"] and set(config["demo_seeds"]) <= set(old["demo_seeds"]),
                  "test seed or demonstration seeds differ from baseline")
    selections = {}
    for task in config["tasks"]:
        full = baseline["samples"].get(task)
        audit.require(full is not None, f"baseline analysis has no verified sample for {task}")
        spec, source = ref["tasks"][task], config["data"][task]
        positions = spec["reference_positions"]
        audit.require(len(positions) == config["n_test"] and all(type(i) is int and 0 <= i < len(full["indices"]) for i in positions)
                      and len(set(positions)) == len(positions), "invalid reference subset positions")
        labels = np.asarray(full["labels"])
        if config["n_test"] == len(labels):
            expected_positions = list(range(len(labels)))
        else:
            positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
            count = math.floor(config["n_test"] * len(positives) / len(labels) + .5)
            count = min(max(count, 1, config["n_test"] - len(negatives)), len(positives), config["n_test"] - 1)
            rng = np.random.default_rng(config["test_seed"])
            expected_positions = np.sort(np.concatenate((rng.choice(positives, count, replace=False),
                rng.choice(negatives, config["n_test"] - count, replace=False)))).tolist()
        audit.require(positions == expected_positions, "reference positions differ from fixed stratified sampling")
        audit.require(spec["full_reference_rows"] == old["n_test"] and
                      spec["sample_sha256"] == audit.digest_file(baseline["run"] / "samples" / f"{task}.json"),
                      "reference sample hash/count differs")
        chosen = {k: [full[k][i] for i in positions] for k in ("indices", "labels", "sequences")}
        chosen["test_seed"] = config["test_seed"]
        audit.require(source["test_indices"] == chosen["indices"] and source["sampled_test_positives"] == sum(chosen["labels"]),
                      "planned subset differs from reference")
        for split in ("train", "test"):
            for kind in ("features", "labels"):
                audit.require(source[split][kind]["sha256"] == old["data"][task][split][kind]["sha256"],
                              f"source data differ: {task}/{split}/{kind}")
        selections[task] = chosen
    return selections


def baseline_rows(baseline, config):
    """Recompute old metrics on the same new query subset, including small pilots."""
    result = []
    for task in config["tasks"]:
        positions = config["reference_check"]["tasks"][task]["reference_positions"]
        for shots in (0, 5, 20):
            for seed in ([None] if shots == 0 else config["demo_seeds"]):
                key = f"{task}/shots{shots}" + ("" if seed is None else f"/demo_seed{seed}")
                if key not in baseline["rows"]:
                    continue
                row = dict(baseline["rows"][key])
                with Path(row["predictions_path"]).open(newline="") as handle:
                    predictions = list(csv.DictReader(handle))
                selected = [predictions[i] for i in positions]
                labels = np.array([int(r["label"]) for r in selected])
                odds = np.array([float(r["log_odds_1_minus_0"]) for r in selected])
                probs = np.array([float(r["probability_1"]) for r in selected])
                predicted = probs >= .5
                tn, fp, fn, tp = audit.confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
                row.update(auc=float(audit.roc_auc_score(labels, odds)), auc_from_probabilities=float(audit.roc_auc_score(labels, probs)),
                    accuracy=float(audit.accuracy_score(labels, predicted)), f1=float(audit.f1_score(labels, predicted, zero_division=0)),
                    precision=float(audit.precision_score(labels, predicted, zero_division=0)), recall=float(audit.recall_score(labels, predicted, zero_division=0)),
                    rows=len(labels), positive_labels=int(labels.sum()), negative_labels=int((labels == 0).sum()),
                    positive_predictions=int(predicted.sum()), negative_predictions=int((~predicted).sum()),
                    tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), probability_min=float(probs.min()), probability_max=float(probs.max()),
                    distinct_probabilities=len(np.unique(probs)), distinct_log_odds=len(np.unique(odds)),
                    probability_boundary_count=int(((probs == 0) | (probs == 1)).sum()), inference_seconds="",
                    condition="nonthinking_previous", source_run=str(baseline["run"]), metric_scope="recomputed_on_new_fixed_test_subset")
                result.append(row)
    return result


def aggregate(rows, config):
    groups, deltas = [], []
    for task in config["tasks"]:
        for shots in (0, 5, 20, 40):
            selected = [r for r in rows if r["task"] == task and r["shots"] == shots]
            n, expected = len(selected), 1 if shots == 0 else len(config["demo_seeds"])
            row = {"task": task, "shots": shots, "condition": "thinking_40" if shots == 40 else "nonthinking_previous",
                   "completed_cells": n, "expected_cells": expected, "status": "complete" if n == expected else "partial" if n else "pending",
                   "test_rows": config["n_test"], "completed_demo_seeds": ",".join(str(r["demo_seed"]) for r in selected) if shots else ""}
            for metric in audit.SCORES:
                values = [r[metric] for r in selected]
                row[metric + "_mean"] = statistics.mean(values) if values else ""
                row[metric + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
            for field in ("natural_completion", "token_cap", "early_eos", "forced_close", "reasoning_tokens_total"):
                row[field] = sum(r[field] for r in selected) if shots == 40 else ""
            row["scored_queries"] = n * config["n_test"]
            groups.append(row)
        old = {r["demo_seed"]: r for r in rows if r["task"] == task and r["shots"] == 20}
        new = {r["demo_seed"]: r for r in rows if r["task"] == task and r["shots"] == 40}
        seeds = sorted(old.keys() & new.keys())
        a, b = [old[s]["auc"] for s in seeds], [new[s]["auc"] for s in seeds]
        deltas.append({"task": task, "paired_cells": len(seeds), "expected_cells": len(config["demo_seeds"]),
                       "paired_demo_seeds": ",".join(map(str, seeds)), "test_rows": config["n_test"],
                       "nonthinking20_auc_mean": statistics.mean(a) if a else "", "thinking40_auc_mean": statistics.mean(b) if b else "",
                       "auc_delta_thinking40_minus_nonthinking20": statistics.mean(y-x for x, y in zip(a, b)) if a else "",
                       "interpretation": "both demonstration count/set and thinking protocol changed"})
    return groups, deltas


def plot_results(output, rows, config, pilot=False):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {0: "#777777", 5: "#377EB8", 20: "#4DAF4A", 40: "#E6550D"}
    fig, ax = plt.subplots(figsize=(8.5, max(2.8, len(config["tasks"])*.7 + 1.2)), constrained_layout=True)
    for t, task in enumerate(config["tasks"]):
        for s, shots in enumerate((0, 5, 20, 40)):
            selected = [r for r in rows if r["task"] == task and r["shots"] == shots]
            for i, row in enumerate(selected):
                ax.scatter(row["auc"], t + (s-1.5)*.16 + (i-(len(selected)-1)/2)*.032,
                           color=colors[shots], marker="D" if shots == 40 else "o", s=30)
    ax.axvline(.5, color=".6", linestyle="--", linewidth=.8)
    completed = sum(r["condition"] == "thinking_40" for r in rows)
    expected = len(config["tasks"]) * len(config["demo_seeds"])
    ax.set(xlim=(0, 1), yticks=range(len(config["tasks"])), yticklabels=[audit.TASK_LABELS[t] for t in config["tasks"]],
           xlabel="Test AUROC from raw label log-odds", title=("Pilot diagnostic: " if pilot else "") +
           "Qwen3.8 prior nonthinking and 40-shot thinking\n" +
           f"{config['n_test']:,} matched test rows/task; {completed}/{expected} thinking cells completed")
    ax.invert_yaxis()
    for shots, color in colors.items():
        ax.scatter([], [], color=color, marker="D" if shots == 40 else "o",
                   label=f"{shots}-shot " + ("thinking" if shots == 40 else "nonthinking"))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    paths = [output / f"qwen_thinking_auroc.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths


def summarize(run, baseline_dir, output, make_plot=True):
    run, output = run.resolve(), output.resolve()
    baseline = comparison.load_analysis(baseline_dir)
    audit.require(all(output != p and not output.is_relative_to(p) for p in (run, baseline["run"], baseline["directory"])),
                  "output must be outside evaluator runs and the prior analysis")
    config_path = run / "config.json"
    config = audit.read_json(config_path)
    cells = validate_config(config)
    run_id = audit.identity(config)
    selections = verify_reference(config, baseline)
    expected_paths = {run / "cells" / c["key"] / "metrics.json" for c in cells}
    audit.require(set((run / "cells").rglob("metrics.json")) <= expected_paths, "unexpected completed thinking cells")
    fixed, sample_hashes, thinking, included = {}, {}, [], []
    for cell in cells:
        directory = run / "cells" / cell["key"]
        row = audit.verify_cell(directory, cell, config, run_id, fixed, sample_hashes)
        if row is None:
            continue
        sample = audit.read_json(Path(audit.read_json(directory / "manifest.json")["test_sample"]["path"]))
        audit.require({k: sample[k] for k in selections[cell["task"]]} == selections[cell["task"]], "actual subset differs from reference")
        demos = audit.read_json(directory / "demonstrations.json")
        audit.require(not ({r["sequence"] for r in demos["examples"]} & set(baseline["samples"][cell["task"]]["sequences"])),
                      "demonstration overlaps the full reference test pool")
        checked, batches = verify_batches(directory, cell, config, row)
        thinking.append(checked)
        path = directory / "metrics.json"
        included.append({"path": str(path), "sha256": audit.digest_file(path), "cell": cell, "reasoning_batches": batches})
    rows = baseline_rows(baseline, config) + thinking
    groups, deltas = aggregate(rows, config)
    n, expected = len(thinking), len(cells)
    pilot = run.name == "pilot" or run.name.startswith("pilot_")
    timestamp = datetime.now(timezone.utc).isoformat()
    verification = {"created_at_utc": timestamp, "status": "completed" if n == expected else "partial",
        "run_kind": "pilot_diagnostic" if pilot else "main_evaluation", "run_path": str(run),
        "completed_cells": n, "expected_cells": expected, "config_path": str(config_path), "config_sha256": audit.digest_file(config_path),
        "run_identity": run_id, "baseline": baseline["evidence"], "metrics_files": included,
        "sample_sha256": {str(p): h for p, h in sample_hashes.items()}, "subset_identity": {t: audit.identity(s) for t, s in selections.items()},
        "script_sha256": audit.digest_file(Path(__file__)), "prediction_verifier_sha256": audit.digest_file(Path(audit.__file__)),
        "reasoning_probability_scope": "conditional on one sampled trace and on choosing label 0 or 1",
        "no_capped_queries_dropped": True, "no_confidence_intervals_or_significance_tests": True}
    output.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    audit.write_csv(output / "per_cell.csv", [{k: row.get(k, "") for k in columns} for row in rows], columns)
    audit.write_csv(output / "summary.csv", groups, list(groups[0]))
    audit.write_csv(output / "comparison_20_40.csv", deltas, list(deltas[0]))
    figures = plot_results(output, rows, config, pilot) if make_plot else []
    natural = sum(r["natural_completion"] for r in thinking)
    capped, eos = sum(r["token_cap"] for r in thinking), sum(r["early_eos"] for r in thinking)
    total = n * config["n_test"]
    p = config["protocol"]
    lines = [f"# {'Pilot diagnostic: ' if pilot else ''}{'complete' if n == expected else 'partial'} Qwen3.8 40-shot thinking results", "",
        f"Independently verified {n}/{expected} thinking cells ({total:,} scored query instances) as of {timestamp}. "
        f"Each cell uses {config['n_test']:,} fixed test rows; repeated cells reuse these rows. "
        "Pending cells are excluded from metrics and are not treated as zero scores. " +
        (f"This pilot covers {len(config['tasks'])} task(s) and is a diagnostic measurement; it must not be presented "
         "as completion of the five-task main benchmark." if pilot else ""), "",
        f"Qwen3.8-27B generated actual reasoning with the official chat template, thinking enabled, "
        f"reasoning effort `{p['reasoning_effort']}`, and a maximum of {p['max_thinking_tokens']:,} generated reasoning tokens "
        "per query (the natural closing token counts toward the budget). Sampling uses temperature 1.0, top-p 0.95 and top-k 20; "
        "batch seeds are fixed and recorded. Forty demonstrations contain 20 negatives and 20 positives, sampled from training "
        "with the existing balanced strategy. They need not contain the earlier 20-shot demonstration set.", "",
        f"Termination among scored query instances: **{natural:,} natural completions, {capped:,} token-cap cutoffs, "
        f"and {eos:,} early EOS stops**. Capped and early-EOS traces are explicitly force-closed before binary scoring; "
        "all remain in the reported metrics. Forced closure is not counted as natural completion.", "",
        "Scores use unwarped label logits after the exact generated token sequence and canonical closing delimiter. "
        "AUROC uses FP32 logit(1)−logit(0); sigmoid gives a probability conditional on this one sampled reasoning trace "
        "and the two answer tokens. It is not a probability averaged over possible reasoning traces or a calibrated "
        "full-vocabulary probability. Accuracy and F1 use threshold 0.5. Raw log-odds avoid sigmoid-induced ties but "
        "cannot remove ties already present in BF16 logits.", "",
        "Prior 0/5/20-shot cells used nonthinking chat. Their predictions below are recomputed on precisely the same "
        "test rows as this run, including nested pilot subsets, using only its requested task and demonstration seeds. "
        "The 20→40 comparison changes demonstration count, demonstration composition, and reasoning protocol together; "
        "it does not isolate the effect of more examples or thinking. Shared seed numbers do not imply nested demonstrations.", "",
        "| Task | Shots / mode | Cells | AUROC mean ± sample SD | Accuracy | F1 |",
        "|---|---|---:|---:|---:|---:|"]
    for row in groups:
        if row["completed_cells"]:
            value = f"{row['auc_mean']:.6f}" + (f" ± {row['auc_sample_sd']:.6f}" if row["auc_sample_sd"] != "" else "")
            accuracy, f1 = f"{row['accuracy_mean']:.6f}", f"{row['f1_mean']:.6f}"
        else:
            value = accuracy = f1 = "—"
        lines.append(f"| {audit.TASK_LABELS[row['task']]} | {row['shots']} / {'thinking' if row['shots']==40 else 'nonthinking'} | "
                     f"{row['completed_cells']}/{row['expected_cells']} | {value} | {accuracy} | {f1} |")
    lines += ["", "Means and sample SDs describe cells with different demonstration draws and sampled reasoning. "
        "They are not confidence intervals, independent training replications, or uncertainty over new test sets. "
        "Zero-shot has one cell and no SD. The 20/40 comparison CSV averages only the same available seed numbers on "
        "both sides. Small or weak differences do not establish a fundamental reasoning limitation.", "",
        "Validation checked checkpoint content identity, shared template/instruction and evaluator source hashes, "
        "source datasets, exact reference subsets, prediction metrics, artifact/batch/row hashes, per-batch seeds, "
        "token-prefix preservation, termination rules, and all CSV/archive correspondence. Reasoning token IDs are "
        "authoritative; decoded text is archival. This audit does not rerun model inference.", "",
        " / ".join(audit.link(output / name, output, name) for name in
                   ("per_cell.csv", "summary.csv", "comparison_20_40.csv", "verification.json")) + ".", "",
        "Sources: " + audit.link(config_path, output, "thinking configuration") + " / " +
        audit.link(baseline["directory"] / "verification.json", output, "prior analysis verification") + "."]
    if figures:
        lines += ["", "Figure: " + " / ".join(audit.link(path, output, path.suffix[1:].upper()) for path in figures) + ". "
            "Each point is one verified cell on the same test subset. Vertical offsets distinguish cells without "
            "changing AUROC. The dashed line marks 0.5; there are no error bars. Missing cells are not plotted."]
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paper = ["# Pilot diagnostic snapshot — not a main benchmark result" if pilot else "# Paper-ready results snapshot", ""]
    for i in (2, 4, 6, 8, 10):
        paper += [lines[i], ""]
    paper += ["Verified 40-shot AUROC by task:", ""]
    for row in groups:
        if row["shots"] == 40:
            value = f"{row['auc_mean']:.6f}" if row["completed_cells"] else "pending"
            spread = f" (sample SD {row['auc_sample_sd']:.6f})" if row["auc_sample_sd"] != "" else ""
            paper.append(f"- {audit.TASK_LABELS[row['task']]}: {value}{spread}; {row['completed_cells']}/{row['expected_cells']} cells.")
    paper += ["", "Differences from prior 20-shot nonthinking results combine changes in demonstration count, "
              "demonstration composition and thinking protocol. Sample SDs describe seed/trace variation, not confidence "
              "intervals. All capped queries remain included. See RESULTS.md and verification.json for scope and provenance."]
    (output / "PAPER_SUMMARY.md").write_text("\n".join(paper) + "\n", encoding="utf-8")
    verification["output_sha256"] = {path.name: audit.digest_file(path) for path in
        [output / name for name in ("per_cell.csv", "summary.csv", "comparison_20_40.csv", "RESULTS.md", "PAPER_SUMMARY.md")] + figures}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-analysis", type=Path, help="verified prior analysis; defaults to reference-run's sibling analysis/")
    parser.add_argument("--reference-run", type=Path, help="prior nonthinking evaluator run")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    if args.baseline_analysis is None and args.reference_run is None:
        parser.error("provide --reference-run or --baseline-analysis")
    if args.reference_run is not None:
        config = audit.read_json(args.run_dir / "config.json")
        audit.require(Path(config["reference_check"]["path"]).resolve() == args.reference_run.resolve(),
                      "requested reference run differs from saved configuration")
    baseline = args.baseline_analysis or args.reference_run.resolve().parent / "analysis"
    result = summarize(args.run_dir, baseline, args.output_dir, not args.no_plot)
    print(json.dumps({k: result[k] for k in ("status", "completed_cells", "expected_cells")}))


if __name__ == "__main__":
    main()
