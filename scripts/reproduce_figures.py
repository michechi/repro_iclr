#!/usr/bin/env python3
"""Regenerate attention figures from bundled aggregate CSVs, without inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICKS = {12: [1, 4, 8, 12], 16: [1, 4, 8, 12, 16],
         32: [1, 8, 16, 24, 32], 48: [1, 12, 24, 36, 48],
         64: [4, 16, 28, 40, 52, 64]}


def save(fig, output, stem):
    for extension in ("pdf", "png", "svg"):
        metadata = {"Creator": "Anonymous reproduction package"}
        if extension == "pdf":
            metadata.update(Author="", CreationDate=None, ModDate=None)
        elif extension == "svg":
            metadata["Date"] = None
        fig.savefig(output / f"{stem}.{extension}", dpi=220, metadata=metadata)
    plt.close(fig)


def synthetic(table, kind, output, stem):
    kip = kind == "kip_m4"
    metrics = [("key_to_key", "Key targets", "#2767aa"),
               ("key_to_inversion", "Inversion-pair targets", "#b73545"),
               ("key_to_noninversion", "Non-inversion-pair targets", "#2f8b57"),
               ("key_to_distractor", "Distractor targets", "#687787")] if kip else [
               ("key_to_lag_key", "Lag-connected key targets", "#b73545"),
               ("key_to_other_key", "Other key targets", "#2767aa"),
               ("key_to_distractor", "Non-key event targets", "#687787")]
    grouping = ["model", "recipe" if kip else "condition"]
    fig, axes = plt.subplots(2, 2 if kip else 3, figsize=(10, 6.3) if kip else (10.8, 6),
                             sharey=kip, layout="constrained" if not kip else None)
    panels = list(table.groupby(grouping, sort=False))
    assert len(panels) == (4 if kip else 5)
    for i, ((model, context), panel) in enumerate(panels):
        ax = axes.flat[i]
        assert not panel.duplicated(["metric", "plotted_layer"]).any()
        for metric, label, color in metrics:
            rows = panel[panel.metric == metric].sort_values("plotted_layer")
            assert len(rows) and np.isfinite(rows.mean_enrichment).all()
            ax.plot(rows.plotted_layer, rows.mean_enrichment, color=color,
                    linewidth=1.6, marker="o", markersize=2.3)
        ax.axhline(1, color="#303030", linestyle="--", linewidth=.9, alpha=.75)
        ax.set_title(f"({chr(97 + i)}) {model}\n{context}", fontsize=10)
        ax.set_xlabel("Model block (1-based)" if kip else "Physical layer (1-based)")
        if not kip or i % 2 == 0:
            ax.set_ylabel("Attention enrichment")
        ax.set_xticks(TICKS[int(panel.plotted_layer.max())])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.15)
    handles = [Line2D([0], [0], color=color, linewidth=2, label=label)
               for _, label, color in metrics]
    handles.append(Line2D([0], [0], color="#303030", linestyle="--",
                          label="Uniform visible-token baseline"))
    if kip:
        axes.flat[0].set_ylim(0, float(table.mean_enrichment.max()) * 1.08)
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        fig.suptitle("KIP-M4: the four audited checkpoints with test AUROC 1.0", fontsize=12)
        fig.tight_layout(rect=(0, .095, 1, .95))
    else:
        axes.flat[5].axis("off")
        axes.flat[5].legend(handles=handles, loc="center left", frameon=False,
                            fontsize=10, labelspacing=1.4)
    save(fig, output, stem)


def clinical(table, output, stem):
    models = [("BERT-base", "512-token limit"), ("RoBERTa-base", "512-token limit"),
              ("Llama-3.1-8B", "Full histories"),
              ("Qwen3.8-27B", "Full histories; standard blocks")]
    metrics = [("event_mass", "Event attention mass"),
               ("normalized_token_entropy", "Normalized token entropy")]
    assert not table.duplicated(["model", "condition", "plotted_layer"]).any()
    assert (table.patients == 1000).all()
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), sharey="row")
    for col, (model, context) in enumerate(models):
        for condition, color, style in [("ordered", "#2767aa", "-"),
                                        ("shuffled", "#d57821", "--")]:
            rows = table[(table.model == model) & (table.condition == condition)].sort_values("plotted_layer")
            assert len(rows)
            for row, (metric, label) in enumerate(metrics):
                values = rows[metric]
                assert np.isfinite(values).all() and values.between(0, 1.0001).all()
                ax = axes[row, col]
                ax.plot(rows.plotted_layer, values, color=color, linestyle=style,
                        linewidth=1.6, label=condition.capitalize())
                ax.set_ylim(0, 1.02)
                ax.grid(axis="y", alpha=.18)
                ax.set_xlabel("Physical layer (1-based)")
                ax.set_xticks(TICKS[int(rows.plotted_layer.max())])
                if col == 0:
                    ax.set_ylabel(label)
                if row == 0:
                    ax.set_title(f"{model}\n{context}", fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("MIMIC-IV: CKD to ESRD | paired attention audit", fontsize=13)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, output, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "anonymous"})
    provenance = {}
    for kind in ("oc", "kip_m4", "ckd"):
        stem = f"attention_{kind}_profiles_20260925"
        source = ROOT / "paper/figures" / f"{stem}.csv"
        table = pd.read_csv(source)
        assert (table.plotted_layer == table.physical_layer_zero_based + 1).all()
        if kind == "ckd":
            clinical(table, args.output_dir, stem)
        else:
            synthetic(table, kind, args.output_dir, stem)
        provenance[source.name] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                                    "aggregate_rows": len(table)}
        print(f"Generated {stem}.pdf/.png/.svg")
    (args.output_dir / "figure_inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
