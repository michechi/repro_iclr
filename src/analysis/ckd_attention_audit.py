"""Readout-query attention on paired CKD test patients without square attention storage."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import argparse
import gc
import os
import time

import numpy as np
import pandas as pd

from src.analysis.kip_attention_core import fixed_sample
from src.analysis.kip_checkpoint_attention import digest, infer, load_checkpoint, read_json, write_json


@contextmanager
def capture_readout_attention(captured, *, row=0, length=None):
    """Preserve native SDPA outputs; recompute only the final query's softmax.

    Q and K arrive after Qwen normalization and rotary embedding. Recomputing
    one query costs O(tokens), while the model keeps its usual efficient SDPA.
    ``row`` and ``length`` select a patient's last unmasked query from the
    original padded evaluation batch, preserving BF16 kernel/batch geometry.
    The interface replacement exists only in this worker and is always restored.
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, repeat_kv
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if isinstance(module, Qwen3_5Attention):
            size = query.shape[-2] if length is None else length
            if (module.training or dropout or query.shape[-2] != key.shape[-2]
                    or not 0 <= row < query.shape[0] or not 1 <= size <= query.shape[-2]
                    or (length is None and query.shape[0] != 1)):
                raise ValueError("readout audit requires an explicit valid row/length for uncached evaluation")
            position = size - 1
            repeated = repeat_kv(key[row:row + 1], module.num_key_value_groups)
            scores = torch.matmul(query[row:row + 1, :, position:position + 1, :], repeated.transpose(2, 3)) * (module.scaling if scaling is None else scaling)
            if attention_mask is not None:
                if attention_mask.ndim != 4:
                    raise ValueError("unsupported clinical attention mask")
                mask_row = 0 if attention_mask.shape[0] == 1 else row
                mask_position = 0 if attention_mask.shape[-2] == 1 else position
                mask = attention_mask[mask_row:mask_row + 1, :, mask_position:mask_position + 1, :]
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            # Native SDPA can use an implicit causal mask. The selected query
            # must exclude both future positions and this patient's right pad.
            scores[..., size:] = float("-inf")
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            captured[module.layer_idx] = weights[0, :, 0, :size].float().cpu().numpy()
            del repeated, scores, weights
        return original(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", capture)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


def clinical_groups(offsets, events):
    from bisect import bisect_right
    prefix = len("Sequential events: ")
    starts, ends = [], []
    for event in events:
        starts.append(prefix); ends.append(prefix + len(event)); prefix += len(event) + 1
    groups = [[] for _ in events]
    for i, (a, b) in enumerate(offsets):
        if a == b:
            continue
        # Leading whitespace may be part of a token; align by overlap.
        position = bisect_right(ends, a)
        if position < len(events) and starts[position] < b:
            if position + 1 < len(events) and starts[position + 1] < b:
                raise ValueError("token overlaps more than one clinical event")
            groups[position].append(i)
    if any(not group for group in groups):
        raise ValueError("unmapped clinical event")
    return groups


def clinical_encode(tokenizer, sequence, device, maximum):
    import torch
    from src.mimic.qwen_ckd_training import sequence_prompt
    events = sequence.split("\x1f")
    encoded = tokenizer(sequence_prompt(sequence), truncation=False, return_offsets_mapping=True)
    if len(encoded["input_ids"]) > maximum:
        raise ValueError("clinical audit would truncate a sequence")
    groups = clinical_groups(encoded["offset_mapping"], events)
    tensors = {k: torch.tensor([encoded[k]], device=device) for k in ("input_ids", "attention_mask")}
    return tensors, events, groups


def summarize_readout(weights, groups):
    """Event mass, uniform token baseline, and entropy for each standard head."""
    if weights.ndim != 2 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("invalid readout attention")
    if np.max(np.abs(weights.sum(-1) - 1)) > .025:
        raise ValueError("readout attention is not normalized")
    mass = np.stack([weights[:, g].sum(-1) for g in groups], axis=-1)
    baseline = np.array([len(g) / weights.shape[-1] for g in groups])
    entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(-1) / np.log(weights.shape[-1])
    return mass, baseline, entropy


def audit_clinical(run_root):
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.experiments.kip_encoder_training import DynamicPadding
    from src.mimic.qwen_ckd_training import ClinicalDataset
    root = Path(run_root)
    plan = read_json(root / "launch_plan.json")
    ckd_root = Path(plan["ckd_root"])
    ckd_state = read_json(ckd_root / "queue_status.json")
    if ckd_state["status"] != "completed":
        raise ValueError("both CKD fits must complete before paired analysis")
    output = root / "clinical_attention"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("completed clinical audit exists")
    started = time.monotonic()

    def progress(status, **details):
        write_json(output / "status.json", dict(status=status, pid=os.getpid(), updated_at_unix=time.time(),
                   wall_seconds=time.monotonic() - started, **details))

    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    data_dir = Path(plan["clinical_data_dir"])
    labels = pd.read_csv(data_dir / "y_test_ordered.csv")["Outcome"].to_numpy(dtype=int)
    other_labels = pd.read_csv(data_dir / "y_test_shuffled.csv")["Outcome"].to_numpy(dtype=int)
    if not np.array_equal(labels, other_labels):
        raise ValueError("paired clinical labels mismatch")
    indices = fixed_sample(labels, plan["clinical_samples"], plan["seed"])
    paired_codes, all_metrics, distributions = {}, {}, {}
    for condition in ("ordered", "shuffled"):
        task = next(j for j in ckd_state["jobs"] if j["key"] == condition)
        run = Path(task["metrics_path"]).parent
        cfg = read_json(run / "manifest.json")["config"]
        if cfg["data"] != condition or cfg["is_smoke"] or cfg["is_dry_run"]:
            raise ValueError("wrong clinical source checkpoint")
        checkpoint = run / "best_trainable.pt"
        checkpoint_hash = digest(checkpoint)
        reference = pd.read_csv(run / "test_predictions.csv").set_index("row_index").loc[indices]
        if not np.array_equal(reference["label"].to_numpy(), labels[indices]):
            raise ValueError("reference patient rows/labels mismatch")
        sequences = pd.read_csv(data_dir / f"X_test_{condition}.csv", dtype=str, keep_default_na=False)["Sequences"].tolist()
        if digest(run / "test_indices.npy") != cfg["index_sha256"]["test"]:
            raise ValueError("clinical evaluation order changed after training")
        evaluation_indices = np.load(run / "test_indices.npy", allow_pickle=False)
        if (evaluation_indices.ndim != 1 or len(np.unique(evaluation_indices)) != len(evaluation_indices)
                or not np.issubdtype(evaluation_indices.dtype, np.integer)
                or np.any(evaluation_indices < 0) or np.any(evaluation_indices >= len(sequences))):
            raise ValueError("invalid original clinical evaluation order")
        row_positions = {int(row): position for position, row in enumerate(evaluation_indices)}
        if any(int(row) not in row_positions for row in indices):
            raise ValueError("clinical attention sample is absent from the original test evaluation")
        eval_batch_size = cfg["eval_batch_size"]
        if not isinstance(eval_batch_size, int) or eval_batch_size < 1:
            raise ValueError("invalid original clinical evaluation batch size")
        job = dict(kind="efficiency", run=str(run), reference=str(run), checkpoint=str(checkpoint),
                   arm="qwen27b_qlora", model_source=plan["model_source"]["snapshot"])
        progress("loading_checkpoint", condition=condition)
        model, tokenizer, model_meta = load_checkpoint(job)
        dataset = ClinicalDataset(sequences, labels, tokenizer, cfg["max_length"])
        collate = DynamicPadding(tokenizer.pad_token_id, cfg["max_length"])
        expected_layers = [i for i, typ in enumerate(model.config.layer_types) if typ == "full_attention"]
        condition_dir = output / condition
        condition_dir.mkdir(exist_ok=True)
        records, predictions, patient_code_mass, code_stats, sample_examples = [], [], {}, {}, []
        max_tokens, backend_delta = 0, None
        for n, row in enumerate(indices):
            encoded, events, groups = clinical_encode(tokenizer, sequences[row], "cpu", cfg["max_length"])
            length = encoded["input_ids"].shape[-1]
            if encoded["input_ids"][0].tolist() != dataset.input_ids[row]:
                raise ValueError("clinical attention token IDs differ from training tokenization")
            position = row_positions[int(row)]
            start = position // eval_batch_size * eval_batch_size
            batch_rows = evaluation_indices[start:start + eval_batch_size]
            batch_row = position - start
            batch = collate([dataset[int(i)] for i in batch_rows])
            inputs = {k: v.to("cuda") for k, v in batch.items() if k != "labels"}
            max_tokens = max(max_tokens, length)
            if condition == "ordered":
                paired_codes[int(row)] = sorted(events)
            elif paired_codes[int(row)] != sorted(events):
                raise ValueError("clinical shuffled row does not preserve the same patient's event multiset")
            if n == 0:
                original, _ = infer(model, inputs, True)
            captured = {}
            with capture_readout_attention(captured, row=batch_row, length=length):
                p, _ = infer(model, inputs, True)
            if sorted(captured) != expected_layers:
                raise ValueError("not all standard-attention layers were captured")
            if n == 0:
                backend_delta = float(np.max(np.abs(p - original)))
                if backend_delta > 1e-6:
                    raise ValueError("readout capture changed native SDPA predictions")
            predictions.append(float(p[batch_row]))
            error = abs(float(p[batch_row]) - float(reference.iloc[n]["probability"]))
            if error > .02:
                write_json(condition_dir / "reproduction_failure.json", dict(
                    row_index=int(row), max_probability_error=error, tolerance=.02,
                    evaluation_batch_rows=batch_rows.tolist(), padded_width=inputs["input_ids"].shape[-1],
                    inference="original evaluation batch, padding, and native SDPA"))
                raise ValueError("clinical checkpoint predictions do not reproduce reference within BF16 tolerance")
            code_indices = {code: [i for i, c in enumerate(events) if c == code] for code in sorted(set(events))}
            distribution = defaultdict(float)
            example_layers = []
            for layer in expected_layers:
                mass, baseline, entropy = summarize_readout(captured[layer], groups)
                head_mass = mass.mean(0)
                example_layers.append(head_mass)
                bins = np.minimum((np.arange(len(events)) * 4 // len(events)), 3)
                for h in range(mass.shape[0]):
                    records.append(dict(row_index=int(row), label=int(labels[row]), layer=layer, head=h,
                        events=len(events), tokens=captured[layer].shape[-1], event_mass=float(mass[h].sum()),
                        prompt_special_mass=float(1 - mass[h].sum()), normalized_token_entropy=float(entropy[h]),
                        **{f"input_quartile_{b+1}_mass": float(mass[h, bins == b].sum()) for b in range(4)}))
                for code, positions in code_indices.items():
                    value = mass[:, positions].sum(-1)
                    expected = float(baseline[positions].sum())
                    k = (layer, code)
                    if k not in code_stats:
                        code_stats[k] = [0, np.zeros_like(value, dtype=np.float64), np.zeros_like(value, dtype=np.float64)]
                    code_stats[k][0] += 1
                    code_stats[k][1] += value
                    code_stats[k][2] += value / expected
                    distribution[code] += float(value.mean()) / len(expected_layers)
            patient_code_mass[int(row)] = dict(distribution)
            if n < 3:
                sample_examples.append((events, np.stack(example_layers)))
            if n % 20 == 0 or n + 1 == len(indices):
                progress("extracting", condition=condition, samples_completed=n + 1, samples=len(indices), max_tokens_seen=max_tokens)
        probability = np.array(predictions)
        errors = np.abs(probability - reference["probability"].to_numpy())
        if float(errors.max()) > .02:
            write_json(condition_dir / "reproduction_failure.json", dict(max_probability_error=float(errors.max()), tolerance=.02))
            raise ValueError("clinical checkpoint predictions do not reproduce reference within BF16 tolerance")
        pd.DataFrame(records).to_csv(condition_dir / "per_patient_head_summary.csv.gz", index=False)
        pd.DataFrame(dict(row_index=indices, label=labels[indices], probability=probability,
                         reference_probability=reference["probability"].to_numpy())).to_csv(condition_dir / "predictions.csv", index=False)
        code_rows = []
        for (layer, code), (count, total, enrichment) in code_stats.items():
            for head in range(len(total)):
                code_rows.append(dict(layer=layer, head=head, code=code, patients_with_code=count,
                    mean_mass_present=float(total[head]/count), mean_mass_all_patients=float(total[head]/len(indices)),
                    mean_enrichment_present=float(enrichment[head]/count)))
        pd.DataFrame(code_rows).to_csv(condition_dir / "code_attention_summary.csv", index=False)
        write_json(condition_dir / "patient_code_mass.json", patient_code_mass)
        write_json(condition_dir / "manifest.json", dict(source_run=str(run), source_checkpoint_sha256=checkpoint_hash,
                    source_manifest_sha256=digest(run / "manifest.json"), model=model_meta,
                    sample_indices=indices.tolist(), standard_layer_ids=expected_layers, capture_probability_delta=backend_delta,
                    capture="final readout query recomputed from normalized rotary Q/K; native SDPA outputs unchanged",
                    inference="original full-length prompt, evaluation batch membership, right padding to multiple 8, native SDPA; no truncation or KV cache",
                    eval_batch_size=eval_batch_size, evaluation_indices_sha256=digest(run / "test_indices.npy")))
        all_metrics[condition] = dict(samples=len(indices), positive_patients=int(labels[indices].sum()),
            sample_auc=float(roc_auc_score(labels[indices], probability)), sample_ap=float(average_precision_score(labels[indices], probability)),
            max_reference_probability_error=float(errors.max()), mean_reference_probability_error=float(errors.mean()),
            max_tokens=max_tokens, full_test_reference=read_json(run / "metrics.json")["test"])
        distributions[condition] = patient_code_mass
        clinical_plots(condition_dir, code_rows, records, sample_examples, expected_layers, condition)
        if digest(checkpoint) != checkpoint_hash:
            raise ValueError("clinical checkpoint changed during read-only audit")
        del model, tokenizer, captured
        gc.collect(); torch.cuda.empty_cache()
    comparisons = []
    for row in indices:
        left, right = distributions["ordered"][int(row)], distributions["shuffled"][int(row)]
        codes = sorted(set(left) | set(right))
        a = np.array([left.get(c, 0) for c in codes]); b = np.array([right.get(c, 0) for c in codes])
        if a.sum() > 0 and b.sum() > 0:
            a /= a.sum(); b /= b.sum()
            comparisons.append(dict(row_index=int(row), label=int(labels[row]), event_conditioned_code_overlap=float(1 - .5*np.abs(a-b).sum())))
    pd.DataFrame(comparisons).to_csv(output / "paired_code_attention_overlap.csv", index=False)
    result = dict(status="completed", conditions=all_metrics, paired_patients=len(comparisons),
                  mean_event_conditioned_code_overlap=float(np.mean([r["event_conditioned_code_overlap"] for r in comparisons])),
                  interpretation="different condition-trained models and different input orders; descriptive, not causal attribution",
                  wall_seconds=time.monotonic() - started)
    result["artifact_sha256"] = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file() and p.name != "status.json"}
    write_json(output / "metrics.json", result)
    progress("completed")


def clinical_plots(output, codes, records, examples, layers, condition):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(codes)
    values = frame.groupby("code")["mean_mass_all_patients"].mean().nlargest(20).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7)); values.plot.barh(ax=ax)
    ax.set_xlabel("Mean readout attention mass across sampled patients, heads and standard layers")
    ax.set_title(f"CKD {condition}: descriptive CCS attention, not feature importance")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / f"ccs_attention.{ext}", dpi=150)
    plt.close(fig)
    frame = pd.DataFrame(records).groupby("layer")
    fig, ax = plt.subplots(figsize=(8, 4))
    for q in range(1, 5):
        v = frame[f"input_quartile_{q}_mass"].mean(); ax.plot(v.index, v.values, label=f"Input quartile {q}")
    ax.set_xlabel("Physical layer"); ax.set_ylabel("Mean readout attention mass"); ax.legend()
    ax.set_title(f"CKD {condition}: input position (not elapsed clinical time)")
    fig.tight_layout(); fig.savefig(output / "input_position_attention.png", dpi=150); plt.close(fig)
    for i, (events, mass) in enumerate(examples):
        # Select by position, never by label or attention magnitude; aggregate long histories into <=80 bins.
        bins = np.array_split(np.arange(len(events)), min(len(events), 80))
        binned = np.stack([mass[:, b].sum(-1) for b in bins], -1)
        fig, ax = plt.subplots(figsize=(12, 4)); im = ax.imshow(binned, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(layers)), layers, fontsize=7)
        ax.set_ylabel("Physical layer"); ax.set_xlabel("Consecutive input-event bins")
        ax.set_title(f"CKD {condition}, fixed sample {i}; final-query attention, full history")
        fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output / f"sample_{i}.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    audit_clinical(args.run_root)
