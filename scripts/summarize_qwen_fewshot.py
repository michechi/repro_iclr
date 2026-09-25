#!/usr/bin/env python3
"""Validate and summarize the five-task Qwen few-shot evaluation without loading a model.

Example:
  python -B scripts/summarize_qwen_fewshot.py \
    --run-dir results/qwen25_14b_fewshot_20260918/run \
    --output-dir results/qwen25_14b_fewshot_20260918/report

Only atomically completed cells count. Partial runs produce explicitly partial
reports; corrupt or inconsistent completed results raise an error. Output must
be outside the evaluator run directory so its configuration guard is preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score


TASKS = ("oc_det", "oc_noise", "gpc", "kip4", "kip6")
TASK_LABELS = {"oc_det": "OC deterministic", "oc_noise": "OC noisy", "gpc": "GPC",
               "kip4": "KIP M4", "kip6": "KIP M6"}
SHOTS = (0, 5, 20)
DEMO_SEEDS = (0, 1, 2)
SCORES = ("auc", "auc_from_probabilities", "accuracy", "f1", "precision", "recall")


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def expected_cells() -> list[dict]:
    cells = []
    for task in TASKS:
        for shots in SHOTS:
            for seed in ((None,) if shots == 0 else DEMO_SEEDS):
                key = f"{task}/shots{shots}" + ("" if seed is None else f"/demo_seed{seed}")
                cells.append({"task": task, "shots": shots, "demo_seed": seed, "key": key})
    return cells


def validate_config(config: dict) -> list[dict]:
    cells = expected_cells()
    require(set(config.get("tasks", [])) == set(TASKS), "expected exactly the five production tasks")
    require(config.get("shots") == list(SHOTS), "expected shots [0, 5, 20]")
    require(config.get("demo_seeds") == list(DEMO_SEEDS), "expected demonstration seeds [0, 1, 2]")
    observed = config.get("cells", [])
    require(len(observed) == 35 and {identity(c) for c in observed} == {identity(c) for c in cells},
            "configuration must contain all 35 distinct planned cells")
    require(config.get("fine_tuning", config.get("protocol", {}).get("fine_tuning")) is False,
            "expected inference-only configuration")
    require(config.get("threshold") == .5, "expected fixed decision threshold 0.5")
    require(config.get("n_test", 0) >= 2, "invalid test sample size")
    return cells


def verify_cell(directory: Path, cell: dict, config: dict, run_id: str,
                fixed_samples: dict, checked_hashes: dict) -> dict | None:
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return None
    saved = read_json(metrics_path)
    require(saved.get("status") == "completed", f"invalid completion marker: {metrics_path}")
    require(saved.get("cell") == cell, f"cell description mismatch: {metrics_path}")
    require(saved.get("run_identity") == run_id, f"run identity mismatch: {metrics_path}")
    hashes = saved.get("artifact_sha256", {})
    require(set(hashes) == {"manifest.json", "demonstrations.json", "predictions.csv"},
            f"invalid artifact inventory: {metrics_path}")
    for name, expected in hashes.items():
        artifact = directory / name
        require(artifact.is_file() and digest_file(artifact) == expected,
                f"missing or changed completed artifact: {artifact}")
    manifest = read_json(directory / "manifest.json")
    demos = read_json(directory / "demonstrations.json")
    cell_id = identity({"run_identity": run_id, "cell": cell, "demonstrations": demos})
    require(saved.get("cell_identity") == cell_id == manifest.get("cell_identity"),
            f"cell identity mismatch: {directory}")
    require(manifest.get("run_identity") == run_id and manifest.get("cell") == cell,
            f"manifest configuration mismatch: {directory}")
    sample_spec = manifest.get("test_sample", {})
    sample_path = Path(sample_spec.get("path", ""))
    require(sample_path.is_file(), f"missing test sample: {sample_path}")
    if sample_path not in checked_hashes:
        checked_hashes[sample_path] = digest_file(sample_path)
    require(checked_hashes[sample_path] == sample_spec.get("sha256"),
            f"changed test sample: {sample_path}")
    sample = read_json(sample_path)
    task = cell["task"]
    source = config["data"][task]
    require(sample.get("source") == source["test"] and sample.get("test_seed") == config["test_seed"],
            f"test-sample provenance mismatch: {directory}")
    require(sample.get("indices") == source["test_indices"], f"test indices differ from plan: {directory}")
    require(len(sample.get("indices", [])) == len(sample.get("labels", [])) ==
            len(sample.get("sequences", [])) == config["n_test"], f"test-sample size mismatch: {directory}")
    require(manifest.get("demo_source") == source["train"], f"demo source mismatch: {directory}")
    examples = demos.get("examples", [])
    require(len(examples) == cell["shots"], f"demonstration count mismatch: {directory}")
    require(all(e.get("label") in (0, 1) for e in examples), f"invalid demonstration label: {directory}")
    require(sum(e["label"] for e in examples) == cell["shots"] // 2,
            f"demonstration class balance mismatch: {directory}")
    demo_indices = [e["train_index"] for e in examples]
    require(len(set(demo_indices)) == len(demo_indices) and demo_indices == demos.get("train_indices"),
            f"duplicate or inconsistent demonstration indices: {directory}")
    require(not ({e["sequence"] for e in examples} & set(sample["sequences"])),
            f"demonstration/test sequence overlap: {directory}")
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    require(len(predictions) == config["n_test"], f"prediction count mismatch: {directory}")
    indices = [int(p["test_row_index"]) for p in predictions]
    labels = np.array([int(p["label"]) for p in predictions], dtype=np.int64)
    probabilities = np.array([float(p["probability_1"]) for p in predictions])
    odds = np.array([float(p["log_odds_1_minus_0"]) for p in predictions])
    logits0 = np.array([float(p["logit_0"]) for p in predictions], dtype=np.float32)
    logits1 = np.array([float(p["logit_1"]) for p in predictions], dtype=np.float32)
    require(indices == sample["indices"] and labels.tolist() == sample["labels"],
            f"prediction rows/labels differ from fixed sample: {directory}")
    require(len(set(indices)) == len(indices) and set(labels.tolist()) == {0, 1},
            f"duplicate test indices or missing label class: {directory}")
    require(int(labels.sum()) == source["sampled_test_positives"], f"test prevalence mismatch: {directory}")
    signature = (indices, labels.tolist())
    if task in fixed_samples:
        require(signature == fixed_samples[task], f"test rows/labels change across {task} cells")
    else:
        fixed_samples[task] = signature
    require(np.isfinite(probabilities).all() and np.isfinite(odds).all() and
            np.isfinite(logits0).all() and np.isfinite(logits1).all(), f"nonfinite predictions: {directory}")
    require(((probabilities >= 0) & (probabilities <= 1)).all(), f"invalid probabilities: {directory}")
    require(np.allclose(odds, logits1 - logits0, atol=1e-7, rtol=1e-7),
            f"stored log-odds differ from label-logit subtraction: {directory}")
    # Stable FP64 sigmoid independently checks FP32 evaluator output without overflow.
    expected_probability = np.exp(-np.logaddexp(0.0, -odds))
    require(np.allclose(probabilities, expected_probability, atol=2e-7, rtol=2e-7),
            f"stored probabilities differ from sigmoid(log-odds): {directory}")
    predicted = probabilities >= .5
    require([int(p["prediction_at_0_5"]) for p in predictions] == predicted.astype(int).tolist(),
            f"saved class decisions differ from fixed threshold: {directory}")
    confusion = confusion_matrix(labels, predicted, labels=[0, 1])
    computed = {"auc": float(roc_auc_score(labels, odds)),
                "auc_from_probabilities": float(roc_auc_score(labels, probabilities)),
                "accuracy": float(accuracy_score(labels, predicted)),
                "f1": float(f1_score(labels, predicted, zero_division=0)),
                "precision": float(precision_score(labels, predicted, zero_division=0)),
                "recall": float(recall_score(labels, predicted, zero_division=0)),
                "threshold": .5, "rows": len(labels), "positive_labels": int(labels.sum()),
                "positive_predictions": int(predicted.sum()),
                "confusion_matrix_labels_0_1": confusion.tolist()}
    recorded_metrics = saved.get("metrics", {})
    for key, value in computed.items():
        other = recorded_metrics.get(key)
        if isinstance(value, float):
            require(isinstance(other, (int, float)) and math.isfinite(other) and
                    math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-10),
                    f"recomputed metric disagrees: {directory}, {key}: {value} != {other}")
        else:
            require(value == other, f"recomputed metric disagrees: {directory}, {key}")
    tn, fp, fn, tp = (int(v) for v in confusion.ravel())
    return {"task": task, "task_name": TASK_LABELS[task], "shots": cell["shots"],
            "demo_seed": "" if cell["demo_seed"] is None else cell["demo_seed"],
            "cell_key": cell["key"], "status": "completed", "rows": len(labels),
            "negative_labels": int((labels == 0).sum()), "positive_labels": int(labels.sum()),
            **{key: computed[key] for key in SCORES}, "threshold": .5,
            "negative_predictions": int((~predicted).sum()), "positive_predictions": int(predicted.sum()),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "probability_min": float(probabilities.min()), "probability_max": float(probabilities.max()),
            "distinct_probabilities": len(np.unique(probabilities)), "distinct_log_odds": len(np.unique(odds)),
            "probability_boundary_count": int(((probabilities == 0) | (probabilities == 1)).sum()),
            "inference_seconds": saved.get("inference_seconds", ""),
            "metrics_path": str(metrics_path.resolve()),
            "predictions_path": str((directory / "predictions.csv").resolve()),
            "manifest_path": str((directory / "manifest.json").resolve()),
            "demonstrations_path": str((directory / "demonstrations.json").resolve())}


def aggregate(per_cell: list[dict], config: dict) -> list[dict]:
    summaries = []
    for task in TASKS:
        for shots in SHOTS:
            rows = [r for r in per_cell if r["task"] == task and r["shots"] == shots]
            expected = 1 if shots == 0 else 3
            group = {"task": task, "task_name": TASK_LABELS[task], "shots": shots,
                     "completed_cells": len(rows), "expected_cells": expected,
                     "status": "complete" if len(rows) == expected else "partial" if rows else "pending",
                     "completed_demo_seeds": ",".join(str(r["demo_seed"]) for r in rows) if shots else "",
                     "test_rows": config["n_test"],
                     "positive_labels": config["data"][task]["sampled_test_positives"]}
            group["negative_labels"] = group["test_rows"] - group["positive_labels"]
            for metric in SCORES:
                values = [r[metric] for r in rows]
                group[metric + "_mean"] = statistics.mean(values) if values else ""
                group[metric + "_sample_sd"] = statistics.stdev(values) if len(values) >= 2 else ""
            summaries.append(group)
    return summaries


def link(path: Path, output: Path, text: str) -> str:
    return f"[{text}](<{os.path.relpath(path.resolve(), output.resolve())}>)"


def plot_summary(output: Path, per_cell: list[dict], model_name: str) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
    colors = {0: "#666666", 5: "#1565C0", 20: "#D55E00"}
    markers = {0: "D", 5: "o", 20: "s"}
    for ti, task in enumerate(TASKS):
        for si, shots in enumerate(SHOTS):
            rows = [r for r in per_cell if r["task"] == task and r["shots"] == shots]
            y = ti + (si - 1) * .22
            for ri, row in enumerate(rows):
                # Offset coincident seeds vertically only; all AUROC positions are exact.
                offset = (ri - (len(rows) - 1) / 2) * .045
                ax.scatter(row["auc"], y + offset, marker=markers[shots], color=colors[shots],
                           edgecolors="white", linewidths=.5, s=42, zorder=3)
    ax.axvline(.5, color="#888888", linestyle="--", linewidth=1)
    ax.set(yticks=range(len(TASKS)), yticklabels=[TASK_LABELS[t] for t in TASKS],
           xlabel="Test AUROC from label log-odds", xlim=(0, 1.02),
           title=f"{model_name.split('/')[-1]}: {len(per_cell)}/35 completed evaluation cells")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=colors[s], marker=markers[s], linestyle="None",
                             label=f"{s} shots") for s in SHOTS], loc="lower right", frameon=True)
    paths = []
    for extension in ("pdf", "svg", "png"):
        path = output / f"qwen_fewshot_auroc.{extension}"
        fig.savefig(path, dpi=200)
        paths.append(path)
    plt.close(fig)
    return paths


def write_report(output: Path, run: Path, config: dict, per_cell: list[dict],
                 summary: list[dict], figures: list[Path], audited_at: str) -> None:
    complete = len(per_cell) == 35
    chat = config.get("protocol", {}).get("prompt_mode") == "official_chat_nonthinking"
    prompt_description = (
        "- Prompt: the official chat template with `enable_thinking=False`, a neutral instruction to "
        "predict exactly 0 or 1, and labeled demonstrations followed by the unlabeled query. Each sequence "
        "contains 20 space-separated letters. No task identity or hidden rule is disclosed, no reasoning "
        "or label text is generated, and no prompt truncation is allowed. The assistant prefix contains "
        "the official empty thinking block. Zero-shot is an uninformed label-scoring control for the hidden rules."
        if chat else
        "- Prompt: `Sequential events: <20 space-separated letters>`, followed by a newline and "
        "`Outcome (0 or 1): `. Labeled demonstrations precede the unlabeled query. No hidden task rule "
        "or task identifier, chat template, label generation, or prompt truncation is used. Zero-shot "
        "therefore measures an uninformed label-scoring control for the hidden rules.")
    lines = [f"# {config['model_name'].split('/')[-1]} few-shot evaluation", "",
             f"Evidence checked at **{audited_at}**. **{'COMPLETE' if complete else 'PARTIAL'}: "
             f"{len(per_cell)}/35 planned cells completed and independently verified.**",
             "Incomplete cells are absent from per-cell results and never counted as failures or chance results.", "",
             "## Test AUROC by task and demonstration count", "",
             "Primary AUROC uses the saved label log-odds. Entries show mean ± sample SD over "
             "completed demonstration seeds and completed/expected cell counts. Zero-shot is evaluated "
             "once per task; its SD is undefined. SD is also blank whenever only one few-shot seed is complete.", "",
             "| Task | 0 shots | 5 shots | 20 shots |", "|---|---:|---:|---:|"]
    for task in TASKS:
        formatted = []
        for shots in SHOTS:
            row = next(r for r in summary if r["task"] == task and r["shots"] == shots)
            if not row["completed_cells"]:
                value = "pending"
            else:
                value = f"{row['auc_mean']:.6f}"
                if row["auc_sample_sd"] != "":
                    value += f" ± {row['auc_sample_sd']:.6f}"
            value += f" ({row['completed_cells']}/{row['expected_cells']})"
            formatted.append(value)
        lines.append(f"| {TASK_LABELS[task]} | " + " | ".join(formatted) + " |")
    lines += ["", "The same sampled test rows and labels are used across all completed cells of each task. "
              "Demo seeds vary the selected training demonstrations and their order; they are not independent "
              "test sets, optimization runs, or model training. Their sample SD is **not a confidence interval**.", "",
              "## Sample counts and trivial-prediction baselines", "",
              "Counts describe the fixed test sampling plan and are checked against every completed prediction file. "
              "F1 is for the positive class. Constant scores have AUROC 0.5.", "",
              "| Task | Test rows | Class 0 | Class 1 | Positive fraction | All-positive accuracy | All-positive F1 | All-negative accuracy |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        n = config["n_test"]
        pos = config["data"][task]["sampled_test_positives"]
        p = pos / n
        lines.append(f"| {TASK_LABELS[task]} | {n:,} | {n-pos:,} | {pos:,} | {p:.6f} | "
                     f"{p:.6f} | {2*p/(1+p):.6f} | {1-p:.6f} |")
    lines += ["", "All-negative precision, recall, and F1 are zero under the recorded `zero_division=0` convention. "
              "Always-positive recall is 1 and precision is the positive fraction. These baselines help "
              "distinguish discrimination from class imbalance or a dominant label preference.", "",
              "## Verified per-cell metrics", "",
              "| Task | Shots | Demo seed | AUROC (log-odds) | AUROC (probabilities) | Accuracy | F1 | Precision | Recall | Predicted positives |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_cell:
        lines.append(f"| {TASK_LABELS[row['task']]} | {row['shots']} | {row['demo_seed'] if row['demo_seed'] != '' else '—'} | "
                     + " | ".join(f"{row[k]:.6f}" for k in SCORES)
                     + f" | {row['positive_predictions']}/{row['rows']} |")
    lines += ["", "Confusion-matrix counts (TN, FP, FN, TP), both class counts, probability diagnostics, "
              "and source paths are retained in " + link(output / "per_cell.csv", output, "per_cell.csv") + ". "
              "The grouped metrics and sample SDs are in " + link(output / "summary.csv", output, "summary.csv") + ".", "",
              "## Protocol and interpretation", "",
              f"- Model: **{config['model_name']}**; local pinned snapshot "
              f"`{config['model_source'].get('snapshot_revision')}`. No fine-tuning or parameter updates.",
              f"- Test sampling: {config['n_test']:,} approximately proportionally stratified rows per task, "
              f"test seed {config['test_seed']}; fixed across demonstration seeds and shot counts.",
              "- Demonstrations come from the training split. Five-shot prompts contain 3 negative and 2 positive "
              "examples; twenty-shot prompts contain 10 of each. Demo seeds are 0, 1, and 2. Demonstration "
              "sets at different shot counts are not nested. Exact sequence overlap with sampled test queries is excluded.",
              prompt_description,
              "- Scores are `logit(1)-logit(0)` and `sigmoid(logit(1)-logit(0))`. The latter is a probability "
              "**conditional on choosing one of {0,1}**, not the full-vocabulary probability mass and not "
              "a demonstrated calibrated probability. Exact one-token label continuations are checked by the evaluator.",
              "- Primary AUROC uses log-odds to avoid artificial ranking ties from sigmoid saturation. "
              "It still retains ties already present in BF16 output logits; FP32 subtraction does not "
              "recover precision lost before that conversion. Distinct score counts are reported per cell. "
              "Secondary AUROC uses saved probabilities. Accuracy, precision, recall, F1, and confusion matrices "
              "use the fixed probability threshold 0.5; no test-selected or validation-selected threshold.",
              f"- Execution settings: BF16 model weights; label logits are promoted to FP32 before subtraction; "
              f"batch size {config['batch_size']}, maximum prompt length {config['max_length']}, "
              f"device `{config['device']}`. No model was loaded by this summarizer.",
              "- Comparisons across tasks concern their particular fixed datasets and rules. This evaluation "
              "uses only a fixed test sample and sampled demonstrations; it is not a matched comparison with "
              "models fine-tuned on full training sets. Do not attribute cross-protocol differences solely to "
              "model architecture, size, or fine-tuning. No uncertainty interval over new test sets or rules is provided.", "",
              "## Source files and validation", "",
              "The complete protocol, versions, model metadata, data hashes, test indices, evaluator source hash, "
              "and intended cell design are in " + link(run / "config.json", output, "run/config.json") + ". "
              "Every included cell passed artifact SHA256 checks, sample/identity consistency, and independent "
              "metric recomputation from saved predictions. Saved logits, log-odds, probabilities, class decisions, "
              "and test indices/labels were cross-checked. Weights are not rehashed by the summarizer.", "",
              "| Task | Train features | Train labels | Test features | Test labels |", "|---|---|---|---|---|"]
    for task in TASKS:
        source = config["data"][task]
        links = [link(Path(source[split][kind]["path"]), output, f"{split} {kind}")
                 for split in ("train", "test") for kind in ("features", "labels")]
        lines.append(f"| {TASK_LABELS[task]} (`{source['tag']}`) | " + " | ".join(links) + " |")
    for name in ("runtime.json", "status.json"):
        if (run / name).is_file():
            lines += ["", link(run / name, output, f"Evaluator {name}") +
                      " is a live runtime/status artifact; completion in this report is determined only by verified cell results."]
    if chat:
        reference = config.get("reference_check", {})
        lines += ["", "This is the post-trained Qwen3.8 checkpoint evaluated in nonthinking chat mode. "
                  "The prior Qwen2.5-14B experiment used base-model raw completion prompts. "
                  "Differences therefore combine checkpoint, training-stage, size, and prompt-protocol effects; "
                  "they do not isolate any one of those factors. Thinking-enabled reasoning was not evaluated."]
        if reference.get("enabled"):
            lines += ["", f"The evaluator checked exact test rows, labels, sequence text, and ordered demonstration "
                      f"payloads against {len(reference['matched_completed_cells'])} completed cells in "
                      + link(Path(reference["path"]), output, "the previous Qwen2.5 run") + "."]
    weight_hashes = run.parent / "evidence/qwen_model_weight_hashes.json"
    if not weight_hashes.is_file():
        weight_hashes = run.parent / "evidence/model_weight_hashes.json"
    if weight_hashes.is_file():
        lines += ["", "Independent checkpoint integrity evidence: " + link(weight_hashes, output, "weight SHA256 audit") + "."]
    lines += ["", "The verification manifest records exactly which metrics files this report checked: "
              + link(output / "verification.json", output, "verification.json") + "."]
    if figures:
        lines += ["", "## Standalone comparison figure", "", " / ".join(link(p, output, p.suffix[1:].upper()) for p in figures), "",
                  f"Caption: Test AUROC from saved label log-odds for each completed {config['model_name']} evaluation. "
                  "Marker/color identify 0, 5, or 20 demonstrations; each marker is one completed cell. "
                  "Small vertical offsets separate demonstration seeds without changing AUROC. The dashed "
                  "line denotes 0.5. There are no confidence intervals or error bars. Missing cells are "
                  "not plotted. All cells for a task use the same test sample."]
    lines += ["", "## Recorded environment", "", "```json", json.dumps(config.get("versions", {}), indent=2), "```", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(run: Path, output: Path, make_plot: bool = True) -> dict:
    run, output = run.resolve(), output.resolve()
    require(run != output and not output.is_relative_to(run), "output directory must be outside the evaluator run directory")
    config_path = run / "config.json"
    config = read_json(config_path)
    cells = validate_config(config)
    run_id = identity(config)
    expected_paths = {run / "cells" / cell["key"] / "metrics.json" for cell in cells}
    extras = set((run / "cells").rglob("metrics.json")) - expected_paths
    require(not extras, f"unexpected completed metrics files: {sorted(map(str, extras))}")
    fixed_samples, checked_hashes, results, included = {}, {}, [], []
    for cell in cells:
        directory = run / "cells" / cell["key"]
        row = verify_cell(directory, cell, config, run_id, fixed_samples, checked_hashes)
        if row is not None:
            results.append(row)
            path = directory / "metrics.json"
            included.append({"path": str(path), "sha256": digest_file(path), "cell": cell})
    summary = aggregate(results, config)
    timestamp = datetime.now(timezone.utc).isoformat()
    verification = {"created_at_utc": timestamp, "status": "completed" if len(results) == 35 else "partial",
                    "completed_cells": len(results), "expected_cells": 35,
                    "run_identity": run_id, "config_path": str(config_path),
                    "config_sha256": digest_file(config_path), "metrics_files": included,
                    "sample_sha256": {str(p): h for p, h in checked_hashes.items()},
                    "summarizer_path": str(Path(__file__).resolve()), "summarizer_sha256": digest_file(Path(__file__)),
                    "primary_auroc_score": "saved log_odds_1_minus_0",
                    "secondary_auroc_score": "saved probability_1", "threshold": .5,
                    "note": "Only completed cells were read; no model loading, inference, or new fitting."}
    # Validate every cell before emitting or overwriting any report artifacts.
    output.mkdir(parents=True, exist_ok=True)
    columns = ["task", "task_name", "shots", "demo_seed", "cell_key", "status", "rows",
               "negative_labels", "positive_labels", *SCORES, "threshold", "negative_predictions",
               "positive_predictions", "tn", "fp", "fn", "tp", "probability_min", "probability_max",
               "distinct_probabilities", "distinct_log_odds", "probability_boundary_count", "inference_seconds",
               "metrics_path", "predictions_path", "manifest_path", "demonstrations_path"]
    write_csv(output / "per_cell.csv", results, columns)
    write_csv(output / "summary.csv", summary, list(summary[0]))
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    figures = plot_summary(output, results, config["model_name"]) if make_plot else []
    write_report(output, run, config, results, summary, figures, timestamp)
    return verification


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true", help="skip standalone matplotlib figures")
    args = parser.parse_args(argv)
    result = summarize(args.run_dir, args.output_dir, make_plot=not args.no_plot)
    print(json.dumps({"status": result["status"], "completed_cells": result["completed_cells"],
                      "expected_cells": result["expected_cells"], "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
