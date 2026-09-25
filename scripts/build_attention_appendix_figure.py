"""Build the appendix OC figure from completed, immutable attention summaries.

Run from the repository with python -B. No GPU is used.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/figures"
STEM = "attention_oc_profiles_20260925"
RUNS = [
    ("BERT-base", "OC-Deterministic, 200K", "results/encoder_oc_ckd_attention_20260923/attention/bert_oc_ordered"),
    ("RoBERTa-base", "OC-Deterministic, 200K", "results/encoder_oc_ckd_attention_20260923/attention/roberta_oc_ordered"),
    ("Llama-3.1-8B", "OC-Deterministic, 200K", "results/llama8_oc_ckd_attention_20260924/attention/llama8_oc_ordered"),
    ("Qwen3.8-27B", "OC-Deterministic, 400K", "results/qwen_oc_clinical_attention_20260922_seed9950/attention/ocdet"),
    ("Qwen3.8-27B", "OC-Noisy, 400K", "results/qwen38_27b_oc_noisy_attention_20260924_seed9950/attention/ocnoise"),
]
METRICS = [
    ("key_to_lag_key", "Lag-connected key targets", "#b73545"),
    ("key_to_other_key", "Other key targets", "#2767aa"),
    ("key_to_distractor", "Non-key event targets", "#687787"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), layout="constrained")
    source_records, aggregate = [], []
    sample_indices = None
    for i, (name, condition, relative) in enumerate(RUNS):
        directory = ROOT / relative
        manifest = json.loads((directory / "manifest.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["status"] == "completed" and metrics["samples"] == 1000
        if sample_indices is None:
            sample_indices = manifest["sample_indices"]
        assert sample_indices == manifest["sample_indices"]
        table = pd.read_csv(directory / "attention_summary.csv")
        table = table[table.condition == "all"]
        axis = axes.flat[i]
        for key, label, color in METRICS:
            rows = table[table.metric == key]
            assert rows.groupby("layer").examples.nunique().eq(1).all()
            profile = rows.groupby("layer").agg(
                enrichment=("mean_enrichment", "mean"),
                mass=("mean_mass", "mean"), support=("examples", "first"))
            assert np.isfinite(profile.enrichment).all()
            axis.plot(profile.index + 1, profile.enrichment, color=color,
                      linewidth=1.6, marker="o", markersize=2.5)
            for layer, row in profile.iterrows():
                aggregate.append(dict(model=name, condition=condition, metric=key,
                    physical_layer_zero_based=int(layer), plotted_layer=int(layer) + 1,
                    mean_enrichment=float(row.enrichment), mean_mass=float(row.mass),
                    eligible_examples=int(row.support)))
        axis.axhline(1, color="#303030", linewidth=.8, linestyle="--", alpha=.75)
        axis.set_ylim(bottom=0)
        axis.set_title(f"({chr(97 + i)}) {name}\n{condition}", fontsize=10, pad=8)
        axis.set_xlabel("Physical layer (1-based)")
        axis.set_ylabel("Attention enrichment")
        layers = np.sort(table.layer.unique()) + 1
        if len(layers) == 12:
            axis.set_xticks([1, 4, 8, 12])
        elif layers[-1] == 32:
            axis.set_xticks([1, 8, 16, 24, 32])
        else:
            axis.set_xticks([4, 16, 28, 40, 52, 64])
        axis.grid(axis="y", alpha=.15)
        source_records.append({"model": name, "condition": condition, "directory": relative,
            "source_sha256": {p: digest(directory / p) for p in
                              ("attention_summary.csv", "manifest.json", "metrics.json")}})
    axes.flat[5].axis("off")
    handles = [Line2D([0], [0], color=color, linewidth=2, label=label)
               for _, label, color in METRICS]
    handles.append(Line2D([0], [0], color="#303030", linestyle="--", linewidth=1,
                          label="Uniform visible-token baseline"))
    axes.flat[5].legend(handles=handles, loc="center left", frameon=False,
                        fontsize=10, labelspacing=1.4)
    for extension in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{extension}", dpi=220)
    plt.close(fig)
    with (OUTPUT / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    (OUTPUT / f"{STEM}.json").write_text(json.dumps({
        "generator_sha256": digest(Path(__file__)), "sources": source_records,
        "samples": len(sample_indices), "condition": "all",
        "aggregation": "equal-head mean of saved per-example mass/uniform-baseline ratios",
        "layer_numbering": "source zero-based physical indices plus one",
        "scope": "replot completed audits; no new inference, training or patient-level export"
    }, indent=2) + "\n")
    print(OUTPUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
