"""Build the Appendix G CKD attention figure from completed attention summaries."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper/figures"
STEM = "attention_ckd_profiles"
RUNS = [
    ("BERT-base", "512-token limit", "results/encoder_oc_ckd_attention/attention/bert_ckd_{condition}"),
    ("RoBERTa-base", "512-token limit", "results/encoder_oc_ckd_attention/attention/roberta_ckd_{condition}"),
    ("Llama-3.1-8B", "Full histories", "results/llama8_oc_ckd_attention/attention/llama8_ckd_{condition}"),
    ("Qwen3.8-27B", "Full histories; standard blocks", "results/qwen_oc_clinical_attention/clinical_attention/{condition}"),
]
METRICS = [("event_mass", "Event attention mass"),
           ("normalized_token_entropy", "Normalized token entropy")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 5.5), sharey="row")
    sources, aggregates = [], []
    common_indices, common_labels = None, None
    for column, (name, context, template) in enumerate(RUNS):
        for condition, color, style in [("ordered", "#2767aa", "-"),
                                         ("shuffled", "#d57821", "--")]:
            directory = ROOT / template.format(condition=condition)
            manifest = json.loads((directory / "manifest.json").read_text())
            if common_indices is None:
                common_indices = manifest["sample_indices"]
            assert manifest["sample_indices"] == common_indices
            assert len(common_indices) == 1000
            metric_path = directory / "metrics.json"
            if not metric_path.exists():
                metric_path = directory.parent / "metrics.json"
            assert json.loads(metric_path.read_text())["status"] == "completed"
            table_path = directory / "per_patient_head_summary.csv.gz"
            table = pd.read_csv(table_path)
            assert not table.duplicated(["row_index", "layer", "head"]).any()
            assert set(table.row_index.unique()) == set(common_indices)
            assert table.groupby("row_index").label.nunique().eq(1).all()
            labels = table.groupby("row_index").label.first().sort_index()
            assert labels.sum() == 77
            if common_labels is None:
                common_labels = labels
            assert labels.equals(common_labels)
            columns = [key for key, _ in METRICS]
            per_patient = table.groupby(["row_index", "layer"])[columns].mean()
            assert per_patient.groupby("layer").size().eq(1000).all()
            profile = per_patient.groupby("layer").mean()
            assert np.isfinite(profile.to_numpy()).all()
            assert ((profile >= 0) & (profile <= 1.0001)).all().all()
            for row, (key, label) in enumerate(METRICS):
                axis = axes[row, column]
                axis.plot(profile.index + 1, profile[key], color=color,
                          linestyle=style, linewidth=1.6, label=condition.capitalize())
                axis.set_ylim(0, 1.02)
                axis.grid(axis="y", alpha=.18)
                axis.set_xlabel("Physical layer (1-based)")
                if column == 0:
                    axis.set_ylabel(label)
                maximum_layer = int(profile.index.max()) + 1
                ticks = {12: [1, 4, 8, 12], 24: [1, 6, 12, 18, 24], 32: [1, 8, 16, 24, 32],
                         64: [4, 16, 28, 40, 52, 64]}[maximum_layer]
                axis.set_xticks(ticks)
                if row == 0:
                    axis.set_title(f"{name}\n{context}", fontsize=10)
            for layer, values in profile.iterrows():
                aggregates.append({"model": name, "condition": condition,
                    "physical_layer_zero_based": int(layer), "plotted_layer": int(layer) + 1,
                    "patients": 1000, **{key: float(values[key]) for key in columns}})
            sources.append({"model": name, "condition": condition,
                "directory": str(directory.relative_to(ROOT)),
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  [table_path, directory / "manifest.json", metric_path]}})
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("MIMIC-IV: CKD to ESRD | paired attention audit", fontsize=13)
    fig.tight_layout(rect=(0, .05, 1, .94))
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(OUTPUT / f"{STEM}.{suffix}", dpi=220)
    plt.close(fig)
    pd.DataFrame(aggregates).to_csv(OUTPUT / f"{STEM}.csv", index=False)
    (OUTPUT / f"{STEM}.json").write_text(json.dumps({
        "generator_sha256": digest(Path(__file__)), "sources": sources,
        "patients": 1000, "positive_patients": 77,
        "aggregation": "equal heads within each patient/layer, then equal patients",
        "entropy_definition": "minus sum a_j log(a_j), divided by log of visible token count",
        "event_mass_definition": "sum of readout attention on complete visible CCS-event tokens"
    }, indent=2) + "\n")
    print(OUTPUT / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
