"""Plot the four successful M4 checkpoint audits from saved attention summaries."""
from pathlib import Path
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
STEM = "attention_kip_m4_profiles_20260925"
RUNS = [
    ("Llama-3.2-1B", "Full fine-tuning, 400K", "llama1b_full_kip_m4"),
    ("Llama-3.1-8B", "LoRA, 200K", "llama8b_lora_m4_0.50"),
    ("Qwen2.5-14B", "QLoRA, 200K", "qwen14b_qlora_m4_0.50"),
    ("Qwen3.8-27B", "QLoRA, 200K; standard blocks only", "qwen27b_qlora_m4_0.50"),
]
METRICS = [
    ("key_to_key", "Key targets", "#2767aa"),
    ("key_to_inversion", "Inversion-pair targets", "#b73545"),
    ("key_to_noninversion", "Non-inversion-pair targets", "#2f8b57"),
    ("key_to_distractor", "Distractor targets", "#687787"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.3), sharey=True)
    records, sources, common_indices = [], [], None
    maximum = 1.
    for i, (name, recipe, job) in enumerate(RUNS):
        directory = ROOT / "results/kip_checkpoint_attention_20260922/jobs" / job
        manifest = json.loads((directory / "manifest.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metrics["status"] == "completed" and metrics["samples"] == 1000
        assert metrics["full_test_auc_reference"] == 1.0 and not metrics["is_smoke"]
        if common_indices is None:
            common_indices = manifest["sample_indices"]
        assert manifest["sample_indices"] == common_indices
        table = pd.read_csv(directory / "attention_summary.csv")
        table = table[table.condition == "all"]
        assert not table.duplicated(["layer", "head", "metric"]).any()
        axis = axes.flat[i]
        for key, label, color in METRICS:
            rows = table[table.metric == key]
            assert rows.groupby("layer").examples.nunique().eq(1).all()
            profile = rows.groupby("layer").agg(
                enrichment=("mean_enrichment", "mean"),
                mass=("mean_mass", "mean"), support=("examples", "first"))
            assert np.isfinite(profile.enrichment).all()
            maximum = max(maximum, float(profile.enrichment.max()))
            axis.plot(profile.index + 1, profile.enrichment, linewidth=1.6,
                      color=color, marker="o", markersize=2.3)
            for layer, values in profile.iterrows():
                records.append({"model": name, "recipe": recipe, "job": job,
                    "metric": key, "physical_layer_zero_based": int(layer),
                    "plotted_layer": int(layer) + 1,
                    "mean_enrichment": float(values.enrichment),
                    "mean_mass": float(values.mass), "eligible_examples": int(values.support)})
        axis.axhline(1, color="#303030", linewidth=.9, linestyle="--", alpha=.75)
        axis.grid(axis="y", alpha=.15)
        axis.set_title(f"({chr(97+i)}) {name}\n{recipe}", fontsize=10)
        axis.set_xlabel("Model block (1-based)")
        if i % 2 == 0:
            axis.set_ylabel("Attention enrichment")
        last_layer = int(table.layer.max()) + 1
        axis.set_xticks({16: [1, 4, 8, 12, 16], 32: [1, 8, 16, 24, 32],
                         48: [1, 12, 24, 36, 48], 64: [4, 16, 28, 40, 52, 64]}[last_layer])
        sources.append({"job": job, "directory": str(directory.relative_to(ROOT)),
            "source_sha256": {p: digest(directory / p) for p in
                              ["attention_summary.csv", "manifest.json", "metrics.json"]}})
    axes.flat[0].set_ylim(0, maximum * 1.08)
    handles = [Line2D([0], [0], color=color, linewidth=2, label=label)
               for _, label, color in METRICS]
    handles.append(Line2D([0], [0], color="#303030", linestyle="--", linewidth=1,
                          label="Uniform visible-token baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("KIP-M4: the four audited checkpoints with test AUROC 1.0", fontsize=12)
    fig.tight_layout(rect=(0, .095, 1, .95))
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{suffix}", dpi=220)
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUTPUT / f"{STEM}.csv", index=False)
    (OUTPUT / f"{STEM}.json").write_text(json.dumps({
        "generator_sha256": digest(Path(__file__)), "sources": sources,
        "samples": len(common_indices), "condition": "all",
        "aggregation": "equal-head mean of saved per-example mass/uniform-baseline ratios",
        "selection": "all four audited M4 checkpoints with reference full-test AUROC exactly 1",
        "layer_numbering": "source zero-based physical indices plus one",
        "limitations": "descriptive selected-checkpoint comparison; differing data sizes/recipes; overlapping target categories; original eager/native methods retained",
        "scope": "replot only; no new training or inference"
    }, indent=2) + "\n")
    print(OUTPUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
