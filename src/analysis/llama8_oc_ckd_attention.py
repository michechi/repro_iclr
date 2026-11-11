"""Native-attention audits for Llama-3.1-8B LoRA on OC and full-context CKD.

OC retains full short-sequence attention and count-preserving controls. CKD
recomputes only the last valid query, so diagnostic storage grows linearly
with history length. Both use the original evaluation batch membership.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.analysis.ckd_attention_audit import (
    clinical_groups, clinical_plots, summarize_readout,
)
from src.analysis.kip_attention_core import event_matrix, fixed_sample, letter_token_groups
from src.analysis.kip_checkpoint_attention import (
    capture_native_attention, digest, infer, load_checkpoint, read_json, write_json,
)
from src.analysis.oc_attention_core import METRICS, label, oc_statistics, perturbations, resolve_oc_rule
from src.analysis.qwen_oc_audit import reference_batch_groups, replace_batch_row
from src.experiments.kip_encoder_training import DynamicPadding


@contextmanager
def capture_readout_attention(captured, row_lengths):
    """Capture final unpadded queries without replacing Llama's native output.

    Each captured[layer][row] is [query heads, valid key tokens]. Q/K are the
    exact projections after rotary embedding. Right padding and future tokens
    are excluded explicitly, including when SDPA's causal mask is implicit.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import LlamaAttention

    native = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if not row_lengths:
        raise ValueError("at least one selected row is required")

    def capture(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if not isinstance(module, LlamaAttention):
            raise ValueError("readout audit requires a Llama attention module")
        if (module.training or dropout or query.shape[-2] != key.shape[-2]
                or not getattr(module, "is_causal", False)
                or kwargs.get("position_bias") is not None):
            raise ValueError("readout audit requires uncached causal Llama evaluation")
        layer = getattr(module, "layer_idx", None)
        if layer is None or layer in captured:
            raise ValueError("attention modules must have unique physical layer indices")
        layer_rows = {}
        for row, length in row_lengths.items():
            if (not isinstance(row, (int, np.integer)) or not isinstance(length, (int, np.integer))
                    or not 0 <= row < query.shape[0] or not 1 <= length <= query.shape[-2]):
                raise ValueError("invalid selected row or unpadded sequence length")
            position = length - 1
            q = query[row:row + 1, :, position:position + 1]
            k = key[row:row + 1, :, :length]
            if q.shape[1] % k.shape[1]:
                raise ValueError("invalid grouped-query attention head counts")
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * (
                scaling if scaling is not None else query.shape[-1] ** -.5)
            if attention_mask is not None:
                if (attention_mask.ndim != 4 or attention_mask.shape[0] not in (1, query.shape[0])
                        or attention_mask.shape[-2] not in (1, query.shape[-2])
                        or attention_mask.shape[-1] != key.shape[-2]):
                    raise ValueError("unsupported native Llama attention mask")
                mask_row = 0 if attention_mask.shape[0] == 1 else row
                mask_position = 0 if attention_mask.shape[-2] == 1 else position
                mask = attention_mask[mask_row:mask_row + 1, :, mask_position:mask_position + 1, :length]
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            result = weights[0, :, 0].float().cpu().numpy()
            if not np.isfinite(result).all():
                raise ValueError("nonfinite readout attention")
            layer_rows[int(row)] = result
        captured[int(layer)] = layer_rows
        return native(module, query, key, value, attention_mask,
                      dropout=dropout, scaling=scaling, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", capture)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", native)


def encode_events(tokenizer, sequence, task, maximum):
    from src.experiments.llama8_oc_ckd import prompt

    encoded = tokenizer(prompt(sequence, task), truncation=False,
                        return_offsets_mapping=True, return_token_type_ids=False)
    if len(encoded["input_ids"]) > maximum:
        raise ValueError("attention analysis would truncate a sequence")
    events = sequence.split("\x1f")
    groups = (letter_token_groups(encoded["offset_mapping"], events) if task == "oc"
              else clinical_groups(encoded["offset_mapping"], events))
    return encoded, events, groups


def sampled_rows(labels, evaluation_order, samples, seed, *, lengths=None, protected_indices=()):
    """Select within the fitted evaluation rows, including restricted smoke sets."""
    order = np.asarray(evaluation_order)
    if (order.ndim != 1 or not np.issubdtype(order.dtype, np.integer)
            or len(order) != len(np.unique(order)) or np.any(order < 0)
            or np.any(order >= len(labels))):
        raise ValueError("invalid original evaluation row indices")
    if lengths is not None:
        from src.experiments.llama8_oc_ckd import smoke_indices
        positions = {int(row): position for position, row in enumerate(order)}
        if any(int(row) not in positions for row in protected_indices):
            raise ValueError("protected smoke row absent from original evaluation")
        selected = smoke_indices(np.asarray(labels)[order], np.asarray(lengths)[order],
            min(samples, len(order)), seed, protected_indices=[positions[int(row)] for row in protected_indices])
    else:
        selected = fixed_sample(np.asarray(labels)[order], min(samples, len(order)), seed)
    return order[selected]


def validate_fit(plan, job, run, manifest, metrics):
    if (metrics.get("status") != "completed" or metrics["is_smoke"] != job["smoke"]
            or manifest["config"].get("arm") != "llama8b_lora"):
        raise ValueError("source fit incomplete or wrong model/smoke status")
    config = manifest["config"]
    if (any(metrics.get(key) != job[key] for key in ("model", "task", "condition"))
            or any(config.get(key) != job[key] for key in ("task", "condition"))
            or config.get("is_smoke") != job["smoke"] or config.get("is_dry_run", False)):
        raise ValueError("selected fit task, condition, or model identity differs")
    if any(record.get("plan_sha256") != plan["plan_sha256"] for record in (manifest, config, metrics)):
        raise ValueError("selected fit belongs to a different frozen plan")
    checkpoint = run / "best_trainable.pt"
    if digest(checkpoint) != metrics["selected_checkpoint_sha256"]:
        raise ValueError("selected weights changed")
    for split in ("train", "val", "test"):
        if digest(run / f"{split}_indices.npy") != manifest["index_sha256"][split]:
            raise ValueError("source split indices changed")
    for filename, expected in plan.get("source_sha256", {}).items():
        if digest(Path(__file__).resolve().parents[2] / filename) != expected:
            raise ValueError("frozen analysis source changed: " + filename)
    return checkpoint


def audit(args):
    from src.experiments.llama8_oc_ckd import TextDataset, classification_metrics, read_data

    root = Path(args.run_root)
    plan = read_json(root / "launch_plan.json")
    job = next(j for j in plan["jobs"] if j["key"] == args.job)
    if job["kind"] != "attention" or job["model"] != "llama8b_lora":
        raise ValueError("not a Llama8 LoRA attention job")
    fit = next(j for j in plan["jobs"] if j["key"] == job["fit_job"])
    if any(fit[k] != job[k] for k in ("model", "task", "condition", "smoke")):
        raise ValueError("attention and fit job identities differ")
    state_file = root / "queue_status.json"
    if state_file.exists():
        active = next(j for j in read_json(state_file)["jobs"] if j["key"] == fit["key"])
        if active.get("status") != "completed":
            raise ValueError("source fit has not completed")
        fit = {**fit, "output_dir": active.get("output_dir", fit["output_dir"])}
    run, output = Path(fit["output_dir"]), Path(job["output_dir"])
    if (output / "metrics.json").exists():
        raise FileExistsError("completed attention audit cannot be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    manifest, fit_metrics = read_json(run / "manifest.json"), read_json(run / "metrics.json")
    checkpoint = validate_fit(plan, job, run, manifest, fit_metrics)
    checkpoint_hash = digest(checkpoint)
    recipe = manifest["recipe"]
    started = time.monotonic()

    def progress(status, **detail):
        write_json(output / "status.json", dict(status=status, job=args.job, pid=os.getpid(),
            updated_at_unix=time.time(), wall_seconds=time.monotonic() - started, **detail))

    progress("loading_checkpoint")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    oc_rule = resolve_oc_rule(plan, job=job) if job["task"] == "oc" else None
    model_job = dict(kind="efficiency", run=str(run), checkpoint=str(checkpoint),
                     arm="llama8b_lora", model_source=plan["models"]["llama8b_lora"]["path"])
    model, tokenizer, model_meta = load_checkpoint(model_job, device=device)
    model.eval()
    sequences, labels = read_data(plan, job["task"], job["condition"], "test")
    order = np.load(run / "test_indices.npy", allow_pickle=False)
    dataset = TextDataset(sequences, labels, tokenizer, job["task"], recipe["max_length"])
    protected = []
    if job["task"] == "ckd":
        other = "shuffled" if job["condition"] == "ordered" else "ordered"
        other_sequences, other_labels = read_data(plan, "ckd", other, "test")
        if not np.array_equal(labels, other_labels):
            raise ValueError("paired CKD labels differ")
        if job["smoke"]:
            other_data = TextDataset(other_sequences, other_labels, tokenizer, "ckd", recipe["max_length"])
            # The same paired worst-case histories are protected for both conditions.
            protected = [int(order[np.argmax(dataset.lengths[order])]),
                         int(order[np.argmax(other_data.lengths[order])])]
            sample_lengths = np.maximum(dataset.lengths, other_data.lengths)
            del other_data
    if job["smoke"] and job["task"] != "ckd":
        sample_lengths = dataset.lengths
    indices = sampled_rows(labels, order, plan.get("smoke_samples", 16) if job["smoke"] else plan["samples"],
        plan["sample_seed"], lengths=sample_lengths if job["smoke"] else None, protected_indices=protected)
    count = len(indices)
    reference = pd.read_csv(run / "test_predictions.csv").set_index("row_index").loc[indices]
    if not np.array_equal(reference["label"].to_numpy(), labels[indices]):
        raise ValueError("reference prediction labels differ")
    if job["task"] == "ckd":
        for row in indices:
            if sorted(sequences[row].split("\x1f")) != sorted(other_sequences[row].split("\x1f")):
                raise ValueError("paired CKD event multisets differ")
    collate = DynamicPadding(tokenizer.pad_token_id, recipe["max_length"])
    batches = reference_batch_groups(indices, order, recipe["eval_batch_size"])
    layers = list(range(model.config.num_hidden_layers))
    probabilities = np.full(count, np.nan)
    examples = [None] * min(3, count)
    masses, baselines = [None] * count, [None] * count
    records, code_stats, patient_mass, counterfactual, coverage = [], {}, {}, [], []
    control_count = min(count, plan.get("smoke_perturbations", 4) if job["smoke"] else plan["perturbations"])
    cf_positions = set(np.random.default_rng(plan["sample_seed"] + 1).choice(count, control_count, replace=False))
    done, native_delta = 0, None
    for batch_rows, selected in batches:
        rows = [dataset[int(i)] for i in batch_rows]
        inputs = {k: v.to(device) for k, v in collate(rows).items() if k != "labels"}
        lengths = {offset: len(rows[offset][0]) for _, offset in selected}
        if native_delta is None:
            before, _ = infer(model, inputs, True)
        captured = {}
        manager = capture_native_attention if job["task"] == "oc" else capture_readout_attention
        with manager(captured, lengths):
            p, _ = infer(model, inputs, True)
        if sorted(captured) != layers:
            raise ValueError("not all physical Llama attention layers were captured")
        if native_delta is None:
            native_delta = float(np.max(np.abs(p - before)))
            if native_delta > 1e-6:
                raise ValueError("attention capture changed native predictions")
        for n, offset in selected:
            row = int(indices[n])
            probabilities[n] = float(p[offset])
            error = abs(probabilities[n] - float(reference.iloc[n]["probability"]))
            if not np.isfinite(error) or error > .02:
                write_json(output / "reproduction_failure.json", dict(row_index=row,
                    max_probability_error=error, tolerance=.02, evaluation_batch_rows=batch_rows.tolist()))
                raise ValueError("original batched test predictions do not reproduce")
            encoded, events, groups = encode_events(tokenizer, sequences[row], job["task"], recipe["max_length"])
            if encoded["input_ids"] != dataset.input_ids[row]:
                raise ValueError("attention tokenization differs from training")
            if job["task"] == "oc":
                keys, lag = oc_rule["keys"], oc_rule["lag"]
                if label(sequences[row], keys, lag) != labels[row]:
                    raise ValueError("deterministic OC oracle differs from saved labels")
                layer_mass, layer_base, matrices = [], [], []
                for layer in layers:
                    attention = captured[layer][offset]
                    mass, baseline = oc_statistics(attention, groups, events, keys, lag, causal=True)
                    layer_mass.append(mass)
                    layer_base.append(baseline)
                    if n < len(examples):
                        matrices.append(event_matrix(attention, groups))
                masses[n], baselines[n] = np.stack(layer_mass), np.stack(layer_base)
                if n < len(examples):
                    examples[n] = (events, np.stack(matrices))
                if n in cf_positions:
                    for variant, (changed, target) in perturbations(sequences[row], keys, lag, plan["sample_seed"] + row).items():
                        changed_encoded, _, _ = encode_events(tokenizer, changed, "oc", recipe["max_length"])
                        changed_inputs = replace_batch_row(inputs, offset, torch.tensor(changed_encoded["input_ids"]))
                        changed_p, _ = infer(model, changed_inputs, True)
                        counterfactual.append(dict(row_index=row, variant=variant,
                            original_label=int(labels[row]), expected_label=int(target),
                            original_probability=probabilities[n], probability=float(changed_p[offset]),
                            original_sequence=sequences[row], changed_sequence=changed))
            else:
                distribution = defaultdict(float)
                code_indices = {code: [i for i, c in enumerate(events) if c == code] for code in sorted(set(events))}
                coverage.append(dict(row_index=row, events=len(events), tokens=len(encoded["input_ids"]), truncated=False))
                example_layers = []
                quartiles = np.minimum(np.arange(len(events)) * 4 // len(events), 3)
                for layer in layers:
                    weights = captured[layer][offset]
                    mass, baseline, entropy = summarize_readout(weights, groups)
                    example_layers.append(mass.mean(0))
                    for head in range(len(mass)):
                        records.append(dict(row_index=row, label=int(labels[row]), layer=layer, head=head,
                            events=len(events), tokens=weights.shape[-1], event_mass=float(mass[head].sum()),
                            prompt_special_mass=float(1 - mass[head].sum()), normalized_token_entropy=float(entropy[head]),
                            **{f"input_quartile_{q + 1}_mass": float(mass[head, quartiles == q].sum()) for q in range(4)}))
                    for code, positions in code_indices.items():
                        value = mass[:, positions].sum(-1)
                        expected = float(baseline[positions].sum())
                        key = (layer, code)
                        if key not in code_stats:
                            code_stats[key] = [0, np.zeros(len(value)), np.zeros(len(value))]
                        code_stats[key][0] += 1
                        code_stats[key][1] += value
                        code_stats[key][2] += value / expected
                        distribution[code] += float(value.mean()) / len(layers)
                patient_mass[str(row)] = dict(distribution)
                if n < len(examples):
                    examples[n] = (events, np.stack(example_layers))
            done += 1
        progress("extracting", samples_completed=done, samples=count)
    errors = np.abs(probabilities - reference["probability"].to_numpy())
    if not np.isfinite(probabilities).all() or done != count:
        raise ValueError("incomplete or invalid probabilities")
    metrics = dict(status="completed", task=job["task"], model="llama8b_lora", condition=job["condition"],
        is_smoke=job["smoke"], samples=count, plan_sha256=plan["plan_sha256"],
        max_reference_probability_error=float(errors.max()), mean_reference_probability_error=float(errors.mean()),
        capture_probability_delta=native_delta, full_test_reference=fit_metrics["test"],
        sample_metrics=classification_metrics(labels[indices], probabilities, .5),
        source_checkpoint_sha256=checkpoint_hash, attention_readout="last unmasked token", causal=True,
        physical_layer_ids=layers, interpretation="descriptive attention, not causal feature importance")
    pd.DataFrame(dict(row_index=indices, label=labels[indices], probability=probabilities,
        reference_probability=reference["probability"].to_numpy())).to_csv(output / "predictions.csv", index=False)
    progress("saving_analysis", samples_completed=count, samples=count)
    title = f"Llama-3.1-8B LoRA | {job['task']} {job['condition']}"
    if job["task"] == "oc":
        mass, base = np.stack(masses).astype(np.float32), np.stack(baselines).astype(np.float32)
        np.savez_compressed(output / "per_example_attention.npz", masses=mass, uniform_baseline=base,
            row_indices=indices, labels=labels[indices], probabilities=probabilities,
            physical_layer_ids=np.array(layers), metric_names=np.array(METRICS))
        correct = (probabilities >= .5) == labels[indices]
        conditions = dict(all=np.ones(count, bool), positive=labels[indices] == 1, negative=labels[indices] == 0,
                          correct=correct, incorrect=~correct)
        summaries = []
        for condition, selected in conditions.items():
            for layer_index, layer in enumerate(layers):
                for metric_index, name in enumerate(METRICS):
                    for head in range(mass.shape[-1]):
                        values, expected = mass[selected, layer_index, metric_index, head], base[selected, layer_index, metric_index]
                        valid = np.isfinite(values) & np.isfinite(expected) & (expected > 0)
                        if valid.any():
                            summaries.append(dict(condition=condition, layer=layer, head=head, metric=name,
                                examples=int(valid.sum()), mean_mass=float(values[valid].mean()),
                                mean_uniform_baseline=float(expected[valid].mean()),
                                mean_enrichment=float((values[valid] / expected[valid]).mean())))
        pd.DataFrame(summaries).to_csv(output / "attention_summary.csv", index=False)
        pd.DataFrame(counterfactual).to_csv(output / "counterfactuals.csv", index=False)
        metrics["counterfactuals"] = {}
        for variant in sorted({r["variant"] for r in counterfactual}):
            rows = [r for r in counterfactual if r["variant"] == variant]
            new = np.array([r["probability"] for r in rows])
            old = np.array([r["original_probability"] for r in rows])
            target = np.array([r["expected_label"] for r in rows])
            metrics["counterfactuals"][variant] = dict(samples=len(rows),
                accuracy_at_0_5=float(np.mean((new >= .5) == target)),
                prediction_flip_rate=float(np.mean((new >= .5) != (old >= .5))),
                mean_absolute_probability_change=float(np.abs(new - old).mean()))
        oc_plots(output, pd.DataFrame(summaries), examples, layers, title)
    else:
        pd.DataFrame(records).to_csv(output / "per_patient_head_summary.csv.gz", index=False)
        pd.DataFrame(coverage).to_csv(output / "context_coverage.csv", index=False)
        code_rows = []
        for (layer, code), (support, total, enrichment) in code_stats.items():
            for head in range(len(total)):
                code_rows.append(dict(layer=layer, head=head, code=code, patients_with_code=support,
                    mean_mass_present=float(total[head] / support), mean_mass_all_patients=float(total[head] / count),
                    mean_enrichment_present=float(enrichment[head] / support)))
        pd.DataFrame(code_rows).to_csv(output / "code_attention_summary.csv.gz", index=False)
        write_json(output / "patient_code_mass.json", patient_mass)
        metrics.update(context_policy="full history; no truncation", truncated_patients=0,
                       max_tokens=max(row["tokens"] for row in coverage),
                       positive_patients=int(labels[indices].sum()))
        clinical_plots(output, code_rows, records, examples, layers, "Llama-3.1-8B LoRA " + job["condition"])
    write_json(output / "manifest.json", dict(plan_sha256=plan["plan_sha256"],
        source_run=str(run), source_manifest_sha256=digest(run / "manifest.json"),
        source_metrics_sha256=digest(run / "metrics.json"), source_checkpoint_sha256=checkpoint_hash,
        sample_indices=indices.tolist(), model=model_meta, physical_layer_ids=layers,
        capture_probability_delta=native_delta, evaluation_indices_sha256=digest(run / "test_indices.npy"),
        eval_batch_size=recipe["eval_batch_size"], readout_query="last unmasked token", causal=True,
        capture="post-rotary Q/K diagnostics; native SDPA output preserved",
        inference="original evaluation batch membership, right padding to multiple 8; full context, no truncation or KV cache",
        counterfactual_context="replace only selected row; same companions, token count, padding, and native SDPA" if job["task"] == "oc" else None,
        source_sha256=plan.get("source_sha256", {})))
    if digest(checkpoint) != checkpoint_hash:
        raise ValueError("selected weights changed during read-only analysis")
    metrics.update(wall_seconds=time.monotonic() - started,
                   artifact_sha256={p.name: digest(p) for p in output.iterdir()
                                    if p.is_file() and p.name not in {"metrics.json", "status.json"}})
    write_json(output / "metrics.json", metrics)
    progress("completed", samples_completed=count)


def oc_plots(output, table, examples, layers, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("key_to_lag_key", "key_to_other_key", "key_to_distractor"):
        rows = table[(table.condition == "all") & (table.metric == name)].groupby("layer")
        for axis, column in zip(axes, ("mean_mass", "mean_enrichment")):
            values = rows[column].mean()
            axis.plot(values.index, values.values, marker=".", label=name)
    axes[0].set_ylabel("Mean attention mass")
    axes[1].set_ylabel("Enrichment over uniform visible tokens")
    axes[1].axhline(1, color="gray", linestyle="--")
    for axis in axes:
        axis.set_xlabel("Physical layer (zero based)")
        axis.legend(fontsize=8)
    fig.suptitle(title + " | causal attention; descriptive audit")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"layer_profiles.{suffix}", dpi=150)
    plt.close(fig)
    shown = sorted(set([0, len(layers) // 2, len(layers) - 1]))
    for n, (events, mass) in enumerate(examples):
        fig, axes = plt.subplots(1, len(shown), figsize=(5 * len(shown), 4), squeeze=False)
        maximum = max(float(mass[layer].max()) for layer in shown)
        for axis, layer in zip(axes[0], shown):
            image = axis.imshow(mass[layer], vmin=0, vmax=maximum, cmap="viridis")
            axis.set_xticks(range(20), events, fontsize=7)
            axis.set_yticks(range(20), events, fontsize=7)
            axis.set_title(f"Physical layer {layers[layer]}")
            axis.set_xlabel("Target event")
            axis.set_ylabel("Query event")
        fig.colorbar(image, ax=list(axes[0]), shrink=.7)
        fig.suptitle(f"{title}, fixed sample {n}; not selected by outcome")
        fig.savefig(output / f"example_{n}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def paired_summary(run_root):
    """Compare code distributions for the same complete CKD histories."""
    from src.experiments.llama8_oc_ckd import read_data

    root = Path(run_root)
    plan = read_json(root / "launch_plan.json")
    output = root / "ckd_paired_comparison"
    if (output / "metrics.json").exists():
        raise FileExistsError("completed paired summary cannot be overwritten")
    jobs = {condition: next(j for j in plan["jobs"] if j["kind"] == "attention"
                            and j["task"] == "ckd" and j["condition"] == condition and not j["smoke"])
            for condition in ("ordered", "shuffled")}
    dirs = {condition: Path(job["output_dir"]) for condition, job in jobs.items()}
    metrics = {condition: read_json(path / "metrics.json") for condition, path in dirs.items()}
    if any(m.get("status") != "completed" or m["is_smoke"] for m in metrics.values()):
        raise ValueError("both production attention audits must complete")
    distributions = {condition: read_json(path / "patient_code_mass.json") for condition, path in dirs.items()}
    if not distributions["ordered"] or set(distributions["ordered"]) != set(distributions["shuffled"]):
        raise ValueError("different or empty sampled CKD patients")
    ordered, labels = read_data(plan, "ckd", "ordered", "test")
    shuffled, other_labels = read_data(plan, "ckd", "shuffled", "test")
    if not np.array_equal(labels, other_labels):
        raise ValueError("paired CKD labels differ")
    rows = []
    for row_text in sorted(distributions["ordered"], key=int):
        row = int(row_text)
        if sorted(ordered[row].split("\x1f")) != sorted(shuffled[row].split("\x1f")):
            raise ValueError("paired CKD histories differ")
        left, right = distributions["ordered"][row_text], distributions["shuffled"][row_text]
        if set(left) != set(right):
            raise ValueError("full-context attention must cover identical patient code sets")
        codes = sorted(left)
        a, b = np.array([left[c] for c in codes]), np.array([right[c] for c in codes])
        if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.sum(), b.sum()) <= 0:
            raise ValueError("invalid patient attention distribution")
        a, b = a / a.sum(), b / b.sum()
        rows.append(dict(row_index=row, label=int(labels[row]),
                         event_conditioned_code_overlap=float(1 - .5 * np.abs(a - b).sum())))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "paired_code_attention_overlap.csv", index=False)
    write_json(output / "metrics.json", dict(status="completed", model="llama8b_lora",
        plan_sha256=plan["plan_sha256"], conditions=metrics, paired_patients=len(rows),
        mean_event_conditioned_code_overlap=float(np.mean([r["event_conditioned_code_overlap"] for r in rows])),
        context_policy="full histories; paired labels and complete code multisets verified",
        interpretation="different condition-trained weights and input orders; descriptive, not causal attribution",
        artifact_sha256={"paired_code_attention_overlap.csv": digest(output / "paired_code_attention_overlap.csv")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--paired-summary", action="store_true")
    args = parser.parse_args()
    if not args.paired_summary and not args.job:
        parser.error("--job is required unless --paired-summary is used")
    paired_summary(args.run_root) if args.paired_summary else audit(args)
