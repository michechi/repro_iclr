"""Attention and count-preserving input audit of a fitted Qwen OC classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import time

import numpy as np
import pandas as pd

from src.analysis.kip_attention_core import event_matrix, fixed_sample
from src.analysis.kip_checkpoint_attention import (
    capture_native_attention, digest, encode, infer, load_checkpoint, read_json, write_json,
)
from src.analysis.oc_attention_core import METRICS, label, oc_statistics, perturbations


def reference_batch_groups(sample_indices, evaluation_indices, batch_size):
    """Group sampled rows without changing the fitted model's evaluation context."""
    samples, order = np.asarray(sample_indices), np.asarray(evaluation_indices)
    if (not isinstance(batch_size, int) or batch_size < 1 or order.ndim != 1 or samples.ndim != 1
            or not np.issubdtype(order.dtype, np.integer) or not np.issubdtype(samples.dtype, np.integer)
            or len(np.unique(order)) != len(order) or len(np.unique(samples)) != len(samples)
            or np.any(order < 0)):
        raise ValueError("invalid original evaluation order or batch size")
    positions = {int(row): position for position, row in enumerate(order)}
    groups = {}
    for n, row in enumerate(samples):
        if int(row) not in positions:
            raise ValueError("sample row absent from original test evaluation")
        position = positions[int(row)]
        start = position // batch_size * batch_size
        groups.setdefault(start, []).append((n, position - start))
    return [(order[start:start + batch_size], groups[start]) for start in sorted(groups)]


def replace_batch_row(inputs, row, token_ids):
    """Keep native BF16 batch geometry fixed for a token-count-matched control."""
    mask = inputs["attention_mask"][row].bool()
    if int(mask.sum()) != len(token_ids):
        raise ValueError("counterfactual changes token count and would change evaluation geometry")
    changed = {key: value.clone() for key, value in inputs.items()}
    changed["input_ids"][row, mask] = token_ids.to(changed["input_ids"].device)
    return changed


def run_audit(run_root, tag):
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.experiments.kip_encoder_training import DynamicPadding
    from src.experiments.paper_pretrained_training import PaperTokenizedDataset
    root = Path(run_root)
    plan = read_json(root / "launch_plan.json")
    state = read_json(root / "queue_status.json")
    completed = next(j for j in state["jobs"] if j["key"] == f"fit_{tag}")
    if completed["status"] != "completed":
        raise ValueError("OC fit must complete before attention audit")
    run = Path(completed["output_dir"])
    output = root / "attention" / tag
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("completed audit cannot be overwritten")
    started = time.monotonic()

    def progress(status, **details):
        write_json(output / "status.json", dict(status=status, pid=os.getpid(), tag=tag,
                   updated_at_unix=time.time(), wall_seconds=time.monotonic() - started, **details))

    progress("loading_data")
    cfg = read_json(run / "manifest.json")["config"]
    source = cfg["data"]["test"]
    for name in ("features", "labels"):
        if digest(source[name]["path"]) != source[name]["sha256"]:
            raise ValueError("OC held-out data changed after training")
    sequences = pd.read_csv(source["features"]["path"], dtype=str, keep_default_na=False)["Sequences"].tolist()
    full_labels = pd.read_csv(source["labels"]["path"])["Outcome"].to_numpy(dtype=int)
    clean = pd.read_csv(Path(plan["data_dir"]) / "y_test_ocdet.csv")["Outcome"].to_numpy(dtype=int)
    # Clean-label stratification provides identical rows for deterministic/noisy fits.
    indices = fixed_sample(clean, plan["attention_samples"], plan["seed"])
    clean, observed = clean[indices], full_labels[indices]
    if not np.array_equal([label(sequences[i]) for i in indices], clean):
        raise ValueError("canonical OC oracle differs from sampled clean labels")
    reference = pd.read_csv(run / "test_predictions.csv").set_index("row_index").loc[indices]
    if not np.array_equal(reference["label"].to_numpy(), observed):
        raise ValueError("reference prediction labels differ from sampled observed labels")
    job = dict(kind="efficiency", run=str(run), reference=str(run), checkpoint=str(run / "best_trainable.pt"),
               arm="qwen27b_qlora", model_source=plan["model_source"]["snapshot"])
    checkpoint_hash = digest(job["checkpoint"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    progress("loading_checkpoint")
    model, tokenizer, model_meta = load_checkpoint(job)
    dataset = PaperTokenizedDataset(sequences, full_labels, tokenizer, cfg["max_length"], decoder=True)
    collate = DynamicPadding(tokenizer.pad_token_id, cfg["max_length"])
    if digest(run / "test_indices.npy") != cfg["index_sha256"]["test"]:
        raise ValueError("OC evaluation order changed after training")
    evaluation_indices = np.load(run / "test_indices.npy", allow_pickle=False)
    if np.any(evaluation_indices >= len(dataset)):
        raise ValueError("original OC evaluation rows exceed the fixed test split")
    batches = reference_batch_groups(indices, evaluation_indices, cfg["eval_batch_size"])
    layers = [i for i, kind in enumerate(model.config.layer_types) if kind == "full_attention"]
    cf_positions = set(np.random.default_rng(plan["seed"] + 1).choice(len(indices), plan["perturbations"], replace=False))
    masses, uniform = [None] * len(indices), [None] * len(indices)
    predictions, cf_rows, examples = np.full(len(indices), np.nan), [], [None] * min(3, len(indices))
    difference, done = None, 0
    for batch_rows, selected in batches:
        batch = collate([dataset[int(row)] for row in batch_rows])
        inputs = {key: value.to("cuda") for key, value in batch.items() if key != "labels"}
        lengths = {offset: len(dataset.input_ids[int(batch_rows[offset])]) for _, offset in selected}
        if difference is None:
            original, _ = infer(model, inputs, True)
        captured = {}
        with capture_native_attention(captured, lengths):
            p, _ = infer(model, inputs, True)
        if sorted(captured) != layers:
            raise ValueError("not all OC standard-attention layers were captured")
        if difference is None:
            difference = float(np.max(np.abs(p - original)))
            if difference > 1e-6:
                raise ValueError("OC attention capture changed native SDPA outputs")
        for n, offset in selected:
            row = int(indices[n])
            encoded, groups, letters = encode(tokenizer, sequences[row], "cpu", cfg["max_length"], decoder=True)
            if encoded["input_ids"][0].tolist() != dataset.input_ids[row]:
                raise ValueError("OC attention token IDs differ from training tokenization")
            predictions[n] = float(p[offset])
            error = abs(predictions[n] - float(reference.iloc[n]["probability"]))
            if error > .02:
                write_json(output / "reproduction_failure.json", dict(row_index=row,
                    max_probability_error=float(error), tolerance=.02, evaluation_batch_rows=batch_rows.tolist(),
                    inference="original evaluation batch, padding, and native SDPA"))
                raise ValueError("restored predictions differ from training's test predictions")
            layer_mass, layer_base, example = [], [], []
            for layer_id in layers:
                attention = captured[layer_id][offset]
                value, base = oc_statistics(attention, groups, letters)
                layer_mass.append(value)
                layer_base.append(base)
                if n < 3:
                    example.append(event_matrix(attention, groups))
            masses[n], uniform[n] = np.stack(layer_mass), np.stack(layer_base)
            if n < 3:
                examples[n] = np.stack(example)
            if n in cf_positions:
                for variant, (changed, expected) in perturbations(sequences[row], plan["seed"] + row).items():
                    encoded2, _, _ = encode(tokenizer, changed, "cpu", cfg["max_length"], decoder=True)
                    inputs2 = replace_batch_row(inputs, offset, encoded2["input_ids"][0])
                    p2, _ = infer(model, inputs2, True)
                    cf_rows.append(dict(row_index=row, variant=variant, original_clean_label=int(clean[n]),
                        expected_clean_label=int(expected), original_probability=predictions[n], probability=float(p2[offset]),
                        original_sequence=sequences[row], changed_sequence=changed))
            done += 1
        progress("extracting", samples_completed=done, samples=len(indices))
    write_json(output / "manifest.json", dict(plan_sha256=plan["plan_sha256"], source_run=str(run),
               checkpoint_sha256=checkpoint_hash, source_metrics_sha256=digest(run / "metrics.json"),
               sample_indices=indices.tolist(), model=model_meta, backend_probability_delta=difference,
               physical_layer_ids=layers, linear_attention_layers_excluded=True,
               evaluation_indices_sha256=digest(run / "test_indices.npy"), eval_batch_size=cfg["eval_batch_size"],
               inference="original evaluation batch membership, right padding to multiple 8, native SDPA; no truncation or KV cache",
               counterfactual_context="replace only selected row; same batch companions, token count, padding, and native SDPA",
               statistics="key-to-lag-key, other keys, distractors, self, prompt; causal visible-token normalization"))
    probability = np.array(predictions)
    error = np.abs(probability - reference["probability"].to_numpy())
    if float(error.max()) > .02:
        write_json(output / "reproduction_failure.json", dict(max_probability_error=float(error.max()), tolerance=.02))
        raise ValueError("restored predictions differ from training's test predictions")
    mass, base = np.stack(masses).astype(np.float32), np.stack(uniform).astype(np.float32)
    np.savez_compressed(output / "per_example_attention.npz", masses=mass, uniform_baseline=base,
                        physical_layer_ids=np.array(layers), metric_names=np.array(METRICS), row_indices=indices,
                        observed_labels=observed, clean_labels=clean, probabilities=probability)
    pd.DataFrame(dict(row_index=indices, observed_label=observed, clean_label=clean, probability=probability,
                     reference_probability=reference["probability"].to_numpy())).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(cf_rows).to_csv(output / "counterfactuals.csv", index=False)
    correct = (probability >= .5) == clean
    conditions = dict(all=np.ones(len(indices), bool), clean_positive=clean == 1, clean_negative=clean == 0,
                      clean_positive_correct=(clean == 1) & correct, clean_positive_missed=(clean == 1) & ~correct,
                      observed_correct=(probability >= .5) == observed, observed_incorrect=(probability >= .5) != observed)
    summaries = []
    for condition, selected in conditions.items():
        for l, physical in enumerate(layers):
            for m, name in enumerate(METRICS):
                for h in range(mass.shape[-1]):
                    v, b = mass[selected, l, m, h], base[selected, l, m]
                    valid = np.isfinite(v) & np.isfinite(b) & (b > 0)
                    if valid.any():
                        summaries.append(dict(condition=condition, layer=physical, head=h, metric=name,
                            examples=int(valid.sum()), mean_mass=float(v[valid].mean()),
                            mean_uniform_baseline=float(b[valid].mean()), mean_enrichment=float((v[valid]/b[valid]).mean())))
    pd.DataFrame(summaries).to_csv(output / "attention_summary.csv", index=False)
    metrics = dict(status="completed", tag=tag, samples=len(indices),
                   sample_observed_auc=float(roc_auc_score(observed, probability)),
                   sample_clean_auc=float(roc_auc_score(clean, probability)),
                   sample_observed_ap=float(average_precision_score(observed, probability)),
                   full_test_reference=read_json(run / "metrics.json")["test"],
                   max_reference_probability_error=float(error.max()), mean_reference_probability_error=float(error.mean()),
                   counterfactuals={}, physical_layer_ids=layers)
    for variant in sorted({r["variant"] for r in cf_rows}):
        rows = [r for r in cf_rows if r["variant"] == variant]
        p = np.array([r["probability"] for r in rows]); old = np.array([r["original_probability"] for r in rows])
        target = np.array([r["expected_clean_label"] for r in rows])
        metrics["counterfactuals"][variant] = dict(samples=len(rows),
            clean_accuracy_at_0_5=float(np.mean((p >= .5) == target)),
            prediction_flip_rate=float(np.mean((p >= .5) != (old >= .5))),
            mean_absolute_probability_change=float(np.abs(p - old).mean()))
    progress("plotting")
    plots(output, summaries, examples, [sequences[i].split("\x1f") for i in indices[:3]], layers, tag)
    if digest(job["checkpoint"]) != checkpoint_hash:
        raise ValueError("selected weights changed during audit")
    metrics["wall_seconds"] = time.monotonic() - started
    metrics["artifact_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"}
    write_json(output / "metrics.json", metrics)
    progress("completed", samples_completed=len(indices))


def plots(output, summaries, examples, letters, layers, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    table = pd.DataFrame(summaries)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("key_to_lag_key", "key_to_other_key", "key_to_distractor"):
        rows = table[(table.condition == "all") & (table.metric == name)].groupby("layer")
        for axis, column in zip(axes, ("mean_mass", "mean_enrichment")):
            values = rows[column].mean(); axis.plot(values.index, values.values, marker=".", label=name)
    axes[0].set_ylabel("Mean attention mass")
    axes[1].set_ylabel("Enrichment over uniform visible tokens")
    axes[1].axhline(1, color="gray", linestyle="--")
    for axis in axes:
        axis.set_xlabel("Physical layer (zero based)"); axis.legend(fontsize=8)
    fig.suptitle(f"Qwen27 {tag}: standard attention only; descriptive audit")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"layer_profiles.{suffix}", dpi=150)
    plt.close(fig)
    selected = sorted(set([0, len(layers)//2, len(layers)-1]))
    for n, example in enumerate(examples):
        fig, axes = plt.subplots(1, len(selected), figsize=(5*len(selected), 4), squeeze=False)
        maximum = max(float(example[l].max()) for l in selected)
        for axis, l in zip(axes[0], selected):
            im = axis.imshow(example[l], vmin=0, vmax=maximum, cmap="viridis")
            axis.set_xticks(range(20), letters[n], fontsize=7); axis.set_yticks(range(20), letters[n], fontsize=7)
            axis.set_title(f"Physical layer {layers[l]}")
            axis.set_xlabel("Target event"); axis.set_ylabel("Query event")
        fig.colorbar(im, ax=list(axes[0]), shrink=.7)
        fig.suptitle(f"{tag}, fixed sample {n}; not selected by outcome")
        fig.savefig(output / f"example_{n}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", choices=("ocdet", "ocnoise"), required=True)
    args = parser.parse_args()
    run_audit(args.run_root, args.tag)
