#!/usr/bin/env python3
"""Compare verified Qwen2.5 raw/base and Qwen3.8 nonthinking-chat analyses.

Each analysis must first be produced by summarize_qwen_fewshot.py. The input
verification receipts define a fixed snapshot; later completed cells are ignored
until the analyses are refreshed. No model loading, inference, or fitting occurs.
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

if __package__:
    from . import summarize_qwen_fewshot as audit
else:
    import summarize_qwen_fewshot as audit


def check_csv(path: Path, expected: list[dict], key_fields: tuple[str, ...]) -> None:
    """Bind report CSVs to independently recomputed source metrics, not filenames."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, actual = reader.fieldnames, list(reader)
    if expected:
        audit.require(set(fields or []) == set(expected[0]), f"CSV columns differ: {path}")
    audit.require(len(actual) == len(expected), f"CSV row count differs: {path}")
    key = lambda row: tuple(str(row[field]) for field in key_fields)
    indexed = {key(row): row for row in actual}
    audit.require(len(indexed) == len(actual), f"duplicate CSV keys: {path}")
    audit.require(set(indexed) == {key(row) for row in expected}, f"CSV cell keys differ: {path}")
    for row in expected:
        for field, value in row.items():
            other = indexed[key(row)].get(field)
            if isinstance(value, (int, float)):
                try:
                    number = float(other)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"nonnumeric CSV field: {path}, {field}") from error
                audit.require(math.isfinite(number) and math.isclose(number, value, rel_tol=1e-12, abs_tol=1e-12),
                              f"CSV value differs from source: {path}, {key(row)}, {field}")
            else:
                audit.require(other == str(value), f"CSV value differs: {path}, {key(row)}, {field}")


def load_analysis(directory: Path) -> dict:
    directory = directory.resolve()
    input_files = {name: directory / name for name in ("verification.json", "per_cell.csv", "summary.csv")}
    input_hashes = {name: audit.digest_file(path) for name, path in input_files.items()}
    receipt = audit.read_json(input_files["verification.json"])
    config_path = Path(receipt["config_path"]).resolve()
    audit.require(config_path.name == "config.json", "verification must identify a run config.json")
    audit.require(audit.digest_file(config_path) == receipt["config_sha256"], "changed source configuration")
    config = audit.read_json(config_path)
    cells = audit.validate_config(config)
    run_id, run = audit.identity(config), config_path.parent
    audit.require(run_id == receipt["run_identity"], "verification/config identity mismatch")
    audit.require(config.get("auc_score") == "raw_float32_log_odds_1_minus_0", "expected raw-log-odds AUROC")
    audit.require(receipt.get("primary_auroc_score") == "saved log_odds_1_minus_0" and
                  receipt.get("secondary_auroc_score") == "saved probability_1" and receipt.get("threshold") == .5,
                  "verification score protocol differs")
    inventory = receipt.get("metrics_files", [])
    audit.require(receipt.get("expected_cells") == 35 and receipt.get("completed_cells") == len(inventory) <= 35,
                  "invalid verification cell counts")
    audit.require(receipt.get("status") == ("completed" if len(inventory) == 35 else "partial"),
                  "verification completion status differs")
    planned = {c["key"]: c for c in cells}
    rows, seen, fixed_samples, sample_hashes, samples, demonstrations = [], set(), {}, {}, {}, {}
    for entry in inventory:
        cell = entry["cell"]
        key = cell["key"]
        audit.require(key in planned and cell == planned[key] and key not in seen, "unexpected/duplicate receipted cell")
        seen.add(key)
        path = (run / "cells" / key / "metrics.json").resolve()
        audit.require(Path(entry["path"]).resolve() == path and audit.digest_file(path) == entry["sha256"],
                      f"changed or misplaced receipted metrics: {key}")
        row = audit.verify_cell(path.parent, cell, config, run_id, fixed_samples, sample_hashes)
        audit.require(row is not None, f"missing receipted completed cell: {key}")
        rows.append(row)
        manifest = audit.read_json(path.parent / "manifest.json")
        sample = audit.read_json(Path(manifest["test_sample"]["path"]))
        payload = {field: sample[field] for field in ("test_seed", "indices", "labels", "sequences")}
        if cell["task"] in samples:
            audit.require(samples[cell["task"]] == payload, "sample text changes across completed cells")
        samples[cell["task"]] = payload
        demonstrations[key] = audit.read_json(path.parent / "demonstrations.json")
    audit.require({str(p): h for p, h in sample_hashes.items()} == receipt.get("sample_sha256"),
                  "verification sample inventory differs")
    check_csv(input_files["per_cell.csv"], rows, ("cell_key",))
    check_csv(input_files["summary.csv"], audit.aggregate(rows, config), ("task", "shots"))
    audit.require(all(audit.digest_file(input_files[name]) == digest for name, digest in input_hashes.items()),
                  "analysis changed while reading; retry after its refresh finishes")
    return {"directory": directory, "run": run, "config": config, "cells": cells,
            "rows": {row["cell_key"]: row for row in rows}, "samples": samples,
            "demonstrations": demonstrations, "receipt": receipt,
            "evidence": {"analysis_directory": str(directory), "input_sha256": input_hashes,
                         "config_path": str(config_path), "config_sha256": receipt["config_sha256"],
                         "run_identity": run_id, "receipt_created_at_utc": receipt.get("created_at_utc"),
                         "completed_cells": len(rows), "original_summarizer_sha256": receipt.get("summarizer_sha256")}}


def check_pairing(old: dict, new: dict) -> dict:
    a, b = old["config"], new["config"]
    audit.require(a["model_name"] == "Qwen/Qwen2.5-14B" and b["model_name"] == "Qwen/Qwen3.8-27B",
                  "this comparison describes the specified Qwen2.5 and Qwen3.8 checkpoints")
    audit.require(a.get("prompt") == "Sequential events: <20 space-separated letters>\nOutcome (0 or 1): ",
                  "unexpected Qwen2.5 raw-completion prompt")
    protocol = b.get("protocol", {})
    audit.require(protocol.get("prompt_mode") == "official_chat_nonthinking" and
                  protocol.get("enable_thinking") is False and protocol.get("model_training_stage") == "posttrained",
                  "unexpected Qwen3.8 chat protocol")
    for field in ("test_seed", "n_test", "shots", "demo_seeds", "threshold"):
        audit.require(a[field] == b[field], f"planned sampling/scoring differs: {field}")
    for task in audit.TASKS:
        x, y = a["data"][task], b["data"][task]
        for field in ("tag", "test_indices", "sampled_test_positives"):
            audit.require(x[field] == y[field], f"{task} planned data differ: {field}")
        for split in ("train", "test"):
            for kind in ("features", "labels"):
                audit.require(x[split][kind]["sha256"] == y[split][kind]["sha256"],
                              f"{task} {split} {kind} source hashes differ")
    sample_ids, demo_ids = {}, {}
    for task in old["samples"].keys() & new["samples"].keys():
        audit.require(old["samples"][task] == new["samples"][task], f"{task} test indices/labels/text differ")
        sample_ids[task] = audit.identity(old["samples"][task])
    for key in old["rows"].keys() & new["rows"].keys():
        audit.require(old["demonstrations"][key] == new["demonstrations"][key],
                      f"demonstration indices/labels/text/order differ: {key}")
        demo_ids[key] = audit.identity(old["demonstrations"][key])
    return {"all_task_sampling_plans_and_source_hashes_match": True,
            "paired_sample_identity": sample_ids, "paired_demonstration_identity": demo_ids}


def paired_tables(old: dict, new: dict, matching: dict) -> tuple[list[dict], list[dict]]:
    paired = []
    for cell in audit.expected_cells():
        key, task = cell["key"], cell["task"]
        a, b = old["rows"].get(key), new["rows"].get(key)
        status = "paired" if a and b else "old_only" if a else "new_only" if b else "pending"
        row = {"task": task, "shots": cell["shots"], "demo_seed": cell["demo_seed"] if cell["demo_seed"] is not None else "",
               "cell_key": key, "status": status, "test_rows": old["config"]["n_test"]}
        for score in ("auc", "accuracy", "f1"):
            row.update({"old_" + score: a[score] if a else "", "new_" + score: b[score] if b else "",
                        "delta_" + score + "_new_minus_old": b[score] - a[score] if a and b else ""})
        row.update(paired_test_sample_identity=matching["paired_sample_identity"].get(task, "") if a and b else "",
                   paired_demonstration_identity=matching["paired_demonstration_identity"].get(key, ""))
        paired.append(row)
    groups = []
    for task in audit.TASKS:
        for shots in audit.SHOTS:
            rows = [r for r in paired if r["task"] == task and r["shots"] == shots]
            matched = [r for r in rows if r["status"] == "paired"]
            n, expected = len(matched), len(rows)
            groups.append({"task": task, "shots": shots, "status": "complete" if n == expected else "partial" if n else "pending",
                           "expected_cells": expected, "old_completed_cells": sum(r["old_auc"] != "" for r in rows),
                           "new_completed_cells": sum(r["new_auc"] != "" for r in rows), "paired_cells": n,
                           "paired_demo_seeds": ",".join(str(r["demo_seed"]) for r in matched) if shots else "",
                           "old_paired_auc_mean": statistics.mean(r["old_auc"] for r in matched) if n else "",
                           "new_paired_auc_mean": statistics.mean(r["new_auc"] for r in matched) if n else "",
                           "paired_auc_mean_delta_new_minus_old": statistics.mean(r["delta_auc_new_minus_old"] for r in matched) if n else ""})
    return paired, groups


def plot_comparison(output: Path, groups: list[dict], completed: int) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 5, figsize=(11, 3.1), sharey=True, constrained_layout=True)
    colors = ("#2166AC", "#B35806")
    for ax, task in zip(axes, audit.TASKS):
        rows = [r for r in groups if r["task"] == task]
        ax.axhline(.5, color="0.65", linestyle="--", linewidth=.8)
        for i, row in enumerate(rows):
            if row["paired_cells"]:
                for side, color, offset in zip(("old", "new"), colors, (-.06, .06)):
                    ax.plot(i + offset, row[side + "_paired_auc_mean"], "o", color=color,
                            markerfacecolor=color if row["status"] == "complete" else "white", markersize=5)
            ax.text(i, 1.035, f"{row['paired_cells']}/{row['expected_cells']}", ha="center", fontsize=8)
        ax.set(title=audit.TASK_LABELS[task], xticks=range(3), xticklabels=[0, 5, 20],
               xlabel="Demonstrations", ylim=(-.025, 1.105), xlim=(-.4, 2.4))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean test AUROC (matched cells)")
    fig.suptitle(f"Qwen comparison: {completed}/35 paired cells; counts above each shot setting", fontsize=11)
    fig.legend([Line2D([], [], color=c, marker="o", linestyle="") for c in colors],
               ["Qwen2.5-14B base/raw", "Qwen3.8-27B post-trained/chat"],
               loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    paths = [output / f"qwen_comparison.{suffix}" for suffix in ("png", "pdf", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return paths


def compare(old_dir: Path, new_dir: Path, output: Path, make_plot: bool = True) -> dict:
    old, new = load_analysis(old_dir), load_analysis(new_dir)
    matching = check_pairing(old, new)
    paired, groups = paired_tables(old, new, matching)
    output = output.resolve()
    for source in (old, new):
        audit.require(all(output != p and not output.is_relative_to(p) for p in (source["directory"], source["run"])),
                      "comparison output must be outside both input analyses and evaluator runs")
    count = sum(r["status"] == "paired" for r in paired)
    now = datetime.now(timezone.utc).isoformat()
    verification = {"created_at_utc": now, "status": "completed" if count == 35 else "partial",
                    "paired_cells": count, "expected_cells": 35, "old": old["evidence"], "new": new["evidence"],
                    "matching": matching, "comparison_script_sha256": audit.digest_file(Path(__file__)),
                    "current_prediction_verifier_sha256": audit.digest_file(Path(audit.__file__)),
                    "aggregation": "arithmetic mean of cell AUROCs over the same completed cells on both sides",
                    "delta_direction": "new minus old", "confidence_intervals_or_significance_tests": False}
    # Do not emit any comparison until both analyses and all pairings are verified.
    output.mkdir(parents=True, exist_ok=True)
    audit.write_csv(output / "paired_per_cell.csv", paired, list(paired[0]))
    audit.write_csv(output / "task_shot_comparison.csv", groups, list(groups[0]))
    figures = plot_comparison(output, groups, count) if make_plot else []
    title = "Complete" if count == 35 else "Partial"
    lines = [f"# {title} Qwen few-shot comparison", "", f"Verified {count}/35 paired cells as of {now}. "
             f"The input analyses contain {len(old['rows'])}/35 Qwen2.5 and {len(new['rows'])}/35 Qwen3.8 completed cells. "
             "This report describes those analysis snapshots; cells completed afterward are not included.", "",
             "Qwen2.5-14B used a base checkpoint with raw completion prompts. Qwen3.8-27B used a post-trained "
             "checkpoint with its official chat template and thinking disabled. Both received the same sequence "
             "data and matched ordered demonstrations, but checkpoint, size, training stage, and prompting differ. "
             "These results do not isolate a model-size or architecture effect.", "",
             "Primary AUROC uses saved FP32 label log-odds. Every reported difference is Qwen3.8 minus Qwen2.5. "
             "Each row averages cell AUROCs over exactly the same completed demonstration seeds for both models; "
             "it does not pool predictions. Zero-shot has one cell; 5- and 20-shot have three planned demonstration "
             "seeds (0, 1, 2). All cells within each task share the fixed test sample. Demo seeds are not independent "
             "training runs. No confidence intervals, significance tests, or claims about new test samples are provided.", "",
             "Raw log-odds avoid ties introduced by FP32 sigmoid saturation, but cannot recover distinctions already "
             "lost when BF16 model logits are equal. Score quantization and the particular prompting/scoring protocol "
             "limit interpretation; weak discrimination here does not establish a fundamental reasoning inability.", "",
             "| Task | Shots | Paired cells | Status | Qwen2.5 AUROC | Qwen3.8 AUROC | Difference |",
             "|---|---:|---:|---|---:|---:|---:|"]
    for row in groups:
        values = [f"{row[key]:+.6f}" if key.startswith("paired_auc_mean_delta") else f"{row[key]:.6f}"
                  for key in ("old_paired_auc_mean", "new_paired_auc_mean", "paired_auc_mean_delta_new_minus_old")] if row["paired_cells"] else ["—"] * 3
        lines.append(f"| {audit.TASK_LABELS[row['task']]} | {row['shots']} | {row['paired_cells']}/{row['expected_cells']} | "
                     + row["status"] + " | " + " | ".join(values) + " |")
    lines += ["", "Blank comparisons are pending, not zero AUROC. Partial means can change when the remaining cells finish. "
              "The per-cell CSV retains all 35 planned cells, including unpaired source metrics and blank differences.", "",
              "Validation rechecked receipt/config/metric/sample hashes, each completed artifact, and metrics from saved "
              "predictions. Both input CSVs were compared with recomputed values. All task plans and train/test data "
              "hashes match; paired cells additionally match test indices, labels, sequence text, and demonstration "
              "indices, labels, text, and order. Prompt identities intentionally differ between these two protocols.", "",
              " / ".join(audit.link(output / name, output, name) for name in
                         ("paired_per_cell.csv", "task_shot_comparison.csv", "verification.json")), "",
              "Source analyses: " + audit.link(old["directory"] / "verification.json", output, "Qwen2.5 verification") + " / "
              + audit.link(new["directory"] / "verification.json", output, "Qwen3.8 verification") + "."]
    if figures:
        lines += ["", "Figure: " + " / ".join(audit.link(p, output, p.suffix[1:].upper()) for p in figures) + ". "
                  "Dots show matched-cell means. Hollow markers indicate incomplete seed coverage; counts appear above "
                  "each shot setting. The dashed line marks AUROC 0.5. No error bars are shown."]
    (output / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verification["output_sha256"] = {p.name: audit.digest_file(p) for p in
                                    [output / "paired_per_cell.csv", output / "task_shot_comparison.csv", output / "COMPARISON.md", *figures]}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-analysis", type=Path, required=True)
    parser.add_argument("--new-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    result = compare(args.old_analysis, args.new_analysis, args.output_dir, not args.no_plot)
    print(json.dumps({"status": result["status"], "paired_cells": result["paired_cells"],
                      "expected_cells": 35, "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
