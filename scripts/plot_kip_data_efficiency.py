#!/usr/bin/env python3
"""Export the completed KIP M4 fraction results as standalone figures."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Write plots separately from the released results')
    args = parser.parse_args()
    with (args.run_root / 'summary.csv').open() as handle:
        all_rows = list(csv.DictReader(handle))
        rows = [row for row in all_rows if row['status'] == 'completed']
    if not rows:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {'qwen27b_qlora': 'Qwen 3.8 27B QLoRA', 'qwen14b_qlora': 'Qwen 2.5 14B QLoRA',
              'llama8b_lora': 'LLaMA 3.1 8B LoRA', 'llama8b_full': 'LLaMA 3.1 8B full'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for arm, label in labels.items():
        selected = sorted([row for row in rows if row['arm'] == arm], key=lambda row: float(row['fraction']))
        if not selected:
            continue
        x = [100 * float(row['fraction']) for row in selected]
        for axis, metric in zip(axes, ['test_auc', 'test_f1']):
            axis.plot(x, [float(row[metric]) for row in selected], 'o-', label=label, linewidth=1.8)
    for axis, ylabel in zip(axes, ['Test AUROC', 'Test F1 (threshold selected on validation)']):
        axis.set(xlabel='Training data (% of 400,000 examples)', ylabel=ylabel, ylim=(0, 1.025),
                 xscale='log', xticks=[1, 10, 30, 50, 75, 100])
        axis.set_xticklabels(['1', '10', '30', '50', '75', '100'])
        axis.grid(alpha=0.2)
    axes[0].axhline(0.5, color='gray', linestyle=':', linewidth=1)
    axes[0].legend(fontsize=8, loc='lower right')
    title = 'KIP M4 ordered · seed 9950 · fixed validation/test: 50,000 each\nCompleted fits only; single-seed curves'
    plan_path = args.run_root / 'launch_plan.json'
    if any(row['status'] == 'skipped_saturated' for row in all_rows) or (plan_path.is_file() and json.loads(plan_path.read_text()).get('stop_on_perfect_test')):
        title += '\nTest-informed stopping; larger fractions skipped after test AUROC = 1'
    fig.suptitle(title)
    destination = args.output_dir or args.run_root / 'plots'
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf', 'svg'):
        path = destination / f'data_efficiency.{extension}'
        temporary = destination / f'.data_efficiency.tmp.{extension}'
        metadata = {'Creator': 'Anonymous reproduction package'}
        if extension == 'pdf':
            metadata.update(Author='', CreationDate=None, ModDate=None)
        elif extension == 'svg':
            metadata['Date'] = None
        fig.savefig(temporary, dpi=180, metadata=metadata)
        temporary.replace(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
