#!/usr/bin/env python3
"""Export completed KIP fraction results separately by task and seed as standalone figures."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Write plots to this directory")
    args = parser.parse_args()
    with (args.run_root / "summary.csv").open() as handle:
        all_rows = list(csv.DictReader(handle))
        rows = [row for row in all_rows if row["status"] == "completed"]
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"qwen14b_lora": "Qwen 2.5 14B LoRA", "qwen27b_qlora": "Qwen 3.8 27B QLoRA",
              "llama8b_lora": "LLaMA 3.1 8B LoRA", "llama8b_full": "LLaMA 3.1 8B full"}
    tags = sorted({row.get("tag") or "kip_m4" for row in rows})
    fig, axes = plt.subplots(len(tags), 2, figsize=(12, 4.8 * len(tags)),
                             layout="constrained", squeeze=False)
    for tag, pair in zip(tags, axes):
        for arm, label in labels.items():
            arm_rows = [row for row in rows if row["arm"] == arm and (row.get("tag") or "kip_m4") == tag]
            for seed in sorted({row["seed"] for row in arm_rows}, key=int):
                selected = sorted([row for row in arm_rows if row["seed"] == seed],
                                  key=lambda row: float(row["fraction"]))
                x = [100 * float(row["fraction"]) for row in selected]
                for axis, metric in zip(pair, ["test_auc", "test_f1"]):
                    axis.plot(x, [float(row[metric]) for row in selected], "o-",
                              label=f"{label} / seed {seed}", linewidth=1.2)
        for axis, ylabel in zip(pair, ["Test AUROC", "Test F1 (validation threshold)"]):
            axis.set(xlabel="Training data (% of 400,000 examples)", ylabel=ylabel, ylim=(0, 1.025),
                     title=tag.upper().replace("_", " "), xscale="log", xticks=[1, 10, 30, 50, 75, 100])
            axis.set_xticklabels(["1", "10", "30", "50", "75", "100"])
            axis.grid(alpha=0.2)
        pair[0].axhline(0.5, color="gray", linestyle=":", linewidth=1)
        pair[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("KIP ordered · fixed validation/test: 50,000 each")
    destination = args.output_dir or args.run_root / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        path = destination / f"data_efficiency.{extension}"
        temporary = destination / f".data_efficiency.tmp.{extension}"
        fig.savefig(temporary, dpi=180)
        temporary.replace(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
