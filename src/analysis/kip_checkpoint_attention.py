"""Offline inspection of existing KIP classifiers; no fitting or checkpoint writes."""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from src.analysis.kip_attention_core import (
    KEYS, METRICS, PREFIX, attention_statistics, counterfactuals, event_matrix,
    fixed_sample, letter_token_groups, validate_sequence,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "results/kip_checkpoint_attention_20260922"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def metadata_default(item):
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False,
                              default=metadata_default) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def discover_jobs():
    jobs = []
    efficiency = ROOT / "results/kip_m4_data_efficiency_20260920_seed9950"
    for arm in ("qwen27b_qlora", "qwen14b_qlora", "llama8b_lora"):
        for fraction in ("0.30", "0.50", "0.01", "0.10"):
            run = efficiency / arm / f"fraction_{fraction}/attempt_001"
            reference = run
            if arm == "qwen27b_qlora" and fraction == "0.50":
                reference = run.parent / "final_evaluation_20260921"
            checkpoint = run / "best_trainable.pt"
            if not checkpoint.is_file() or not (reference / "metrics.json").is_file():
                continue
            jobs.append(dict(key=f"{arm}_m4_{fraction}", kind="efficiency", run=str(run),
                             reference=str(reference), checkpoint=str(checkpoint), tag="kip_m4", arm=arm))
    for root_name in ("kip_llama1b_full_20260916_seed9950", "kip_encoders_20260916_seed9950"):
        base = ROOT / "results" / root_name / "runs"
        for checkpoint in sorted(base.glob("*/*/kip_m*/ordered/seed9950/best_model")):
            run = checkpoint.parent
            if not (run / "metrics.json").is_file():
                continue
            config = read_json(run / "manifest.json")["config"]
            model = "llama1b" if "llama" in config["model_name"].lower() else (
                "roberta" if "roberta" in config["model_name"].lower() else "bert")
            tuning = config.get("tuning", "full")
            jobs.append(dict(key=f"{model}_{tuning}_{config['tag']}", kind="hf", run=str(run),
                             reference=str(run), checkpoint=str(checkpoint), tag=config["tag"], arm=model))
    return jobs


def checkpoint_files(job):
    p = Path(job["checkpoint"])
    return [p] if p.is_file() else sorted(f for f in p.iterdir() if f.is_file())


def build_plan(run_root, samples=1000, perturbations=200):
    import pandas as pd
    if not 0 <= perturbations <= samples:
        raise ValueError("counterfactual sample must be within the attention sample")
    run_root = Path(run_root).resolve()
    if (run_root / "launch_plan.json").exists():
        raise FileExistsError("existing plan; resume the queue instead of rebuilding")
    jobs = discover_jobs()
    if not jobs or jobs[0]["arm"] != "qwen27b_qlora":
        raise RuntimeError("priority Qwen checkpoints missing")
    plan = dict(schema_version=1, task="checkpoint_only_kip_attention", samples=samples,
                perturbations=perturbations, seed=20260922, jobs=jobs,
                fitting=False, checkpoint_writes=False, data={}, created_at_unix=time.time(),
                exclusions={"GPC": "no locally available fine-tuned checkpoint found",
                            "qwen27b_kip_m6": "no fine-tuned checkpoint found",
                            "llama8b_full": "completed fits did not persist selected full weights"},
                priority="finish CKD, inspect saved KIP checkpoints, then resume Llama70",
                interpretation="attention is descriptive; swaps test input dependence, not a causal internal circuit",
                statistics="uniform-visible-token baseline; causal future positions excluded; all examples primary")
    run_root.mkdir(parents=True, exist_ok=True)
    for tag in sorted({j["tag"] for j in jobs}):
        d = ROOT / "data/simulation/kip_ordered_recovered"
        xp, yp = d / f"X_test_{tag}.csv", d / f"y_test_{tag}.csv"
        x = pd.read_csv(xp, dtype=str, keep_default_na=False)["Sequences"].tolist()
        y = pd.read_csv(yp)["Outcome"].to_numpy(dtype=int)
        if len(x) != 50000 or len(y) != len(x):
            raise ValueError("requires the fixed 50K held-out split")
        for s, label in zip(x, y):
            if validate_sequence(s, tag)[1] != label:
                raise ValueError("KIP rule does not match saved labels")
        indices = fixed_sample(y, samples, plan["seed"])
        cf = np.sort(np.random.default_rng(plan["seed"] + 1).choice(samples, perturbations, replace=False))
        plan["data"][tag] = dict(features=str(xp), labels=str(yp),
                                 features_sha256=digest(xp), labels_sha256=digest(yp),
                                 indices=indices.tolist(), perturbation_sample_positions=cf.tolist())
    for job in jobs:
        manifest = read_json(Path(job["run"]) / "manifest.json")
        cfg = manifest["config"]
        if cfg.get("is_smoke") or cfg.get("mode") != "ordered" or cfg["tag"] != job["tag"]:
            raise ValueError("only production ordered-trained checkpoints are in this analysis")
        files = checkpoint_files(job) + [Path(job["run"]) / "manifest.json",
                                        Path(job["reference"]) / "metrics.json",
                                        Path(job["reference"]) / "test_predictions.csv"]
        job["input_sha256"] = {str(p): digest(p) for p in files}
        source = cfg["model_source"]
        job["model_source"] = source["snapshot"] if isinstance(source, dict) else source
        job["source_manifest_sha256"] = digest(Path(job["run"]) / "manifest.json")
        job["status"] = "pending"
    sources = ["src/analysis/kip_attention_core.py", "src/analysis/kip_checkpoint_attention.py",
               "scripts/run_kip_checkpoint_attention.py", "docs/kip_checkpoint_attention_20260922.md",
               "src/experiments/kip_efficiency_models.py", "src/experiments/kip_encoder_training.py",
               "src/experiments/kip_data_efficiency.py", "src/experiments/kip_llama_training.py"]
    plan["source_sha256"] = {p: digest(ROOT / p) for p in sources}
    plan["plan_sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    write_json(run_root / "launch_plan.json", plan)
    return plan


def verify_job(plan, job):
    for p, h in plan["source_sha256"].items():
        if digest(ROOT / p) != h:
            raise ValueError(f"analysis/runtime source changed: {p}")
    for p, h in job["input_sha256"].items():
        if digest(p) != h:
            raise ValueError(f"checkpoint or reference changed: {p}")
    source = plan["data"][job["tag"]]
    for key in ("features", "labels"):
        if digest(source[key]) != source[key + "_sha256"]:
            raise ValueError("held-out data changed")
    cfg = read_json(Path(job["run"]) / "manifest.json")["config"]
    provenance = cfg["data"]["test"]
    provenance = provenance.get("ordered", provenance)
    for key in ("features", "labels"):
        if provenance[key]["sha256"] != source[key + "_sha256"]:
            raise ValueError("analysis split differs from checkpoint's held-out split")


def load_checkpoint(job, device="cuda"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    run, checkpoint = Path(job["run"]), Path(job["checkpoint"])
    manifest = read_json(run / "manifest.json")
    if job["kind"] == "efficiency":
        from src.experiments.kip_efficiency_models import load_classifier
        from src.experiments.kip_encoder_training import restore_trainable
        from src.experiments.kip_data_efficiency import model_provenance
        if model_provenance(Path(job["model_source"])) != manifest["config"]["model_source"]:
            raise ValueError("pretrained model identity differs from training")
        model, tokenizer, meta = load_classifier(job["arm"], job["model_source"],
                                                device=device, gradient_checkpointing=False)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("arm") != job["arm"]:
            raise ValueError("saved adapter arm mismatch")
        expected = set(manifest["model"]["trainable_parameter_names"])
        if set(saved["state_dict"]) != expected:
            raise ValueError("adapter/head checkpoint is incomplete")
        restore_trainable(model, saved["state_dict"])
        meta["selected_epoch"] = saved["epoch"]
    else:
        adapter = (checkpoint / "adapter_config.json").is_file()
        source = Path(job["model_source"]) if adapter else checkpoint
        if adapter:
            for name, h in manifest["config"]["source_file_sha256"].items():
                if digest(source / name) != h:
                    raise ValueError("encoder adapter base changed")
        model, info = AutoModelForSequenceClassification.from_pretrained(
            source, num_labels=2, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True)
        if not adapter and any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"incomplete full classifier: {info}")
        if adapter:
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
            expected = load_file(str(checkpoint / "adapter_model.safetensors"))
            actual = get_peft_model_state_dict(model)
            if set(expected) != set(actual) or any(not torch.equal(expected[k], actual[k].cpu()) for k in expected):
                raise ValueError("adapter and classifier-head restoration is not exact")
        tokenizer_path = checkpoint if (checkpoint / "tokenizer_config.json").exists() else run / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
        model.to(device)
        meta = dict(model_name=manifest["config"]["model_name"], loading_info=info)
    model.eval()
    return model, tokenizer, meta


def infer(model, inputs, efficiency, attention=False):
    import torch
    from src.experiments.kip_efficiency_models import last_unmasked_hidden
    with torch.inference_mode(), torch.autocast(device_type=inputs["input_ids"].device.type, dtype=torch.bfloat16):
        if efficiency and attention:
            mask = inputs["attention_mask"]
            out = model.backbone(**inputs, position_ids=(mask.long().cumsum(-1) - 1).clamp_min(0),
                                 use_cache=False, return_dict=True, output_attentions=True)
            logits = model.score(last_unmasked_hidden(out.last_hidden_state, mask).to(model.score.weight.dtype))
        elif efficiency:
            out = model(**inputs)
            logits = out.logits
        else:
            out = model(**inputs, output_attentions=attention, return_dict=True)
            logits = out.logits
        probabilities = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return probabilities, getattr(out, "attentions", None) if attention else None


@contextmanager
def capture_native_attention(captured, row_lengths):
    """Inspect selected unpadded rows without replacing the native SDPA output.

    Keys and queries are the exact projections used in that forward (after
    rotary embedding and any query/key normalization). Only their diagnostic
    softmax is materialized; subsequent model layers still receive native SDPA.
    ``captured[layer][batch_row]`` contains [heads, valid tokens, valid tokens].
    """
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
        if module.training or dropout or query.shape[-2] != key.shape[-2]:
            raise ValueError("attention capture requires uncached evaluation")
        if kwargs.get("position_bias") is not None:
            raise ValueError("position-biased attention capture is unsupported")
        layer = getattr(module, "layer_idx", None)
        if layer is None or layer in captured:
            raise ValueError("attention modules must have unique physical layer indices")
        layer_rows = {}
        for row, length in row_lengths.items():
            if not 0 <= row < query.shape[0] or not 0 < length <= query.shape[-2]:
                raise ValueError("invalid attention sample row/length")
            q = query[row:row + 1, :, :length]
            k = key[row:row + 1, :, :length]
            if q.shape[1] % k.shape[1]:
                raise ValueError("invalid grouped-query attention head counts")
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * (
                scaling if scaling is not None else query.shape[-1] ** -.5)
            if attention_mask is not None:
                if attention_mask.ndim != 4 or attention_mask.shape[0] not in (1, query.shape[0]):
                    raise ValueError("unsupported native attention mask")
                mask_row = 0 if attention_mask.shape[0] == 1 else row
                mask = attention_mask[mask_row:mask_row + 1, :, :length, :length]
                scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
            else:
                causal = kwargs.get("is_causal")
                if causal is None:
                    causal = getattr(module, "is_causal", True)
                if causal:
                    allowed = torch.ones((length, length), device=query.device, dtype=torch.bool).tril()
                    scores = scores.masked_fill(~allowed, float("-inf"))
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            layer_rows[int(row)] = weights[0].float().cpu().numpy()
        captured[int(layer)] = layer_rows
        return original(module, query, key, value, attention_mask,
                        dropout=dropout, scaling=scaling, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("sdpa", capture)
    try:
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original)


def attention_layers(config, arrays):
    types = getattr(config, "layer_types", None)
    full = [i for i, t in enumerate(types) if t == "full_attention"] if types else list(range(config.num_hidden_layers))
    if arrays is None:
        raise ValueError("backend returned no attention")
    if len(arrays) == config.num_hidden_layers:
        selected = [(i, arrays[i]) for i in full]
    elif len(arrays) == len(full):
        selected = list(zip(full, arrays))
    else:
        raise ValueError("attention count cannot be mapped to actual layer indices")
    if any(a is None for _, a in selected):
        raise ValueError("a standard-attention layer did not expose its matrix")
    return selected


def encode(tokenizer, sequence, device, max_length):
    import torch
    letters = sequence.split("\x1f")
    text = PREFIX + " ".join(letters)
    encoded = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=True)
    if len(encoded["input_ids"]) > max_length:
        raise ValueError("analysis requires truncation, forbidden")
    groups = letter_token_groups(encoded["offset_mapping"], letters)
    inputs = {k: torch.tensor([encoded[k]], device=device) for k in ("input_ids", "attention_mask")}
    return inputs, groups, letters


def analyze_job(run_root, key, limit=None):
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score, average_precision_score
    plan = read_json(Path(run_root) / "launch_plan.json")
    job = next(j for j in plan["jobs"] if j["key"] == key)
    output = Path(run_root) / "jobs" / key
    if limit:
        output = Path(run_root) / "smoke" / key
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("completed analysis cannot be overwritten")
    started = time.monotonic()

    def progress(status, **extra):
        write_json(output / "status.json", dict(status=status, pid=os.getpid(), key=key,
                    updated_at_unix=time.time(), wall_seconds=time.monotonic() - started, **extra))

    progress("verifying_inputs")
    verify_job(plan, job)
    source = plan["data"][job["tag"]]
    indices = np.array(source["indices"][:limit] if limit else source["indices"])
    sequences = pd.read_csv(source["features"], dtype=str, keep_default_na=False)["Sequences"].tolist()
    labels = pd.read_csv(source["labels"])["Outcome"].to_numpy(dtype=int)[indices]
    reference = pd.read_csv(Path(job["reference"]) / "test_predictions.csv")
    if "row_index" not in reference or "probability" not in reference:
        raise ValueError("saved test predictions lack explicit row/probability columns")
    evaluation_indices = reference["row_index"].to_numpy(dtype=int)
    if len(set(evaluation_indices)) != len(evaluation_indices):
        raise ValueError("reference evaluation rows must be unique")
    reference = reference.set_index("row_index").loc[indices]
    if not np.array_equal(reference["label"].to_numpy(), labels):
        raise ValueError("reference labels do not match sampled rows")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    progress("loading_checkpoint")
    model, tokenizer, model_meta = load_checkpoint(job)
    causal = model.config.model_type not in ("bert", "roberta")
    cfg = read_json(Path(job["run"]) / "manifest.json")["config"]
    from src.experiments.kip_encoder_training import DynamicPadding, TokenizedDataset
    dataset = TokenizedDataset([sequences[i] for i in evaluation_indices],
                               np.zeros(len(evaluation_indices), dtype=int), tokenizer, cfg["max_length"])
    collator = DynamicPadding(tokenizer.pad_token_id, cfg["max_length"])
    batch_size = int(cfg["eval_batch_size"])
    sample_positions = {int(row): n for n, row in enumerate(indices)}
    types = getattr(model.config, "layer_types", None)
    layer_ids = [i for i, typ in enumerate(types) if typ == "full_attention"] if types else list(
        range(model.config.num_hidden_layers))
    stats, baselines, predictions = ([None] * len(indices) for _ in range(3))
    perturbed, matrices = [], [None] * min(3, len(indices))
    cf_positions = set(source["perturbation_sample_positions"])
    delta, completed = None, 0

    def batch_inputs(rows):
        return {k: v.to("cuda") for k, v in collator(rows).items() if k != "labels"}

    for start in range(0, len(evaluation_indices), batch_size):
        batch_indices = evaluation_indices[start:start + batch_size]
        selected = [(offset, sample_positions[int(row)], int(row))
                    for offset, row in enumerate(batch_indices) if int(row) in sample_positions]
        if not selected:
            continue
        rows = [dataset[i] for i in range(start, start + len(batch_indices))]
        inputs = batch_inputs(rows)
        if delta is None:
            original, _ = infer(model, inputs, job["kind"] == "efficiency")
        captured = {}
        with capture_native_attention(captured, {offset: len(rows[offset][0]) for offset, _, _ in selected}):
            probs, _ = infer(model, inputs, job["kind"] == "efficiency")
        if sorted(captured) != layer_ids:
            raise ValueError("not all standard-attention layers were captured")
        if delta is None:
            delta = float(np.max(np.abs(original - probs)))
            if delta > 1e-6:
                raise ValueError(f"native attention capture changed prediction by {delta}")
            write_json(output / "manifest.json", dict(job=job, plan_sha256=plan["plan_sha256"],
                       model=model_meta, layer_ids=layer_ids, causal=causal,
                       missing_linear_attention_layers=[i for i in range(model.config.num_hidden_layers) if i not in layer_ids],
                       attention_scope="standard softmax attention only; hybrid recurrent blocks excluded",
                       capture="diagnostic Q/K softmax; native SDPA model outputs unchanged",
                       evaluation="original reference batch membership and DynamicPadding multiple=8",
                       eval_batch_size=batch_size, backend_probability_delta=delta,
                       no_training=True, sample_indices=indices.tolist()))
        variants = {}
        for offset, n, row in selected:
            encoded, groups, letters = encode(tokenizer, sequences[row], "cpu", cfg["max_length"])
            if encoded["input_ids"][0].tolist() != rows[offset][0]:
                raise ValueError("offset tokenizer differs from original evaluation tokenizer")
            predictions[n] = float(probs[offset])
            per_layer, base_layer, example_matrices = [], [], []
            for layer_id in layer_ids:
                attn = captured[layer_id][offset]
                mass, baseline = attention_statistics(attn, groups, letters, job["tag"], causal,
                                                       attn.shape[-1] - 1 if causal else 0)
                per_layer.append(mass)
                base_layer.append(baseline)
                if n < 3:
                    example_matrices.append(event_matrix(attn, groups))
            stats[n], baselines[n] = np.stack(per_layer), np.stack(base_layer)
            if n < 3:
                matrices[n] = np.stack(example_matrices)
            if n in cf_positions:
                for variant, (changed, expected_label) in counterfactuals(sequences[row], job["tag"]).items():
                    variants.setdefault(variant, []).append((offset, n, row, changed, expected_label))
        del captured
        for variant, changed_rows in variants.items():
            variant_rows = rows.copy()
            for offset, n, row, changed, expected_label in changed_rows:
                encoded, _, _ = encode(tokenizer, changed, "cpu", cfg["max_length"])
                variant_rows[offset] = (encoded["input_ids"][0].tolist(), 0)
            changed_inputs = batch_inputs(variant_rows)
            if changed_inputs["input_ids"].shape != inputs["input_ids"].shape:
                raise ValueError("counterfactual changed evaluation batch geometry")
            p, _ = infer(model, changed_inputs, job["kind"] == "efficiency")
            for offset, n, row, changed, expected_label in changed_rows:
                perturbed.append(dict(row_index=row, variant=variant, original_label=int(labels[n]),
                    expected_label=int(expected_label), original_probability=predictions[n], probability=float(p[offset]),
                    original_sequence=sequences[row], changed_sequence=changed))
        completed += len(selected)
        progress("extracting", samples_completed=completed, samples=len(indices))
    if completed != len(indices):
        raise ValueError("not all sampled reference rows were evaluated")
    probabilities = np.array(predictions)
    reference_error = np.abs(probabilities - reference["probability"].to_numpy())
    # Keep the original strict tolerance; reference batching/padding is now reproduced.
    if float(reference_error.max()) > .02:
        write_json(output / "reproduction_failure.json", dict(max_probability_error=float(reference_error.max()),
                     mean_probability_error=float(reference_error.mean()), tolerance=.02))
        raise ValueError("saved-checkpoint predictions were not reproduced within 0.02 probability tolerance")
    data = np.stack(stats).astype(np.float32)  # example, physical layer, metric, head
    baseline = np.stack(baselines).astype(np.float32)
    np.savez_compressed(output / "per_example_attention.npz", masses=data, uniform_baseline=baseline,
                        row_indices=indices, labels=labels, probabilities=probabilities,
                        layer_ids=np.array(layer_ids), metric_names=np.array(METRICS))
    pd.DataFrame(dict(row_index=indices, label=labels, probability=probabilities,
                     reference_probability=reference["probability"].to_numpy())).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(perturbed).to_csv(output / "counterfactuals.csv", index=False)
    summary = []
    conditions = {"all": np.ones(len(labels), bool), "correct_at_0.5": (probabilities >= .5) == labels,
                  "incorrect_at_0.5": (probabilities >= .5) != labels, "label0": labels == 0, "label1": labels == 1}
    for condition, mask in conditions.items():
        for l, layer_id in enumerate(layer_ids):
            for m, name in enumerate(METRICS):
                for h in range(data.shape[-1]):
                    values, expected = data[mask, l, m, h], baseline[mask, l, m]
                    valid = np.isfinite(values) & np.isfinite(expected) & (expected > 0)
                    if not valid.any():
                        continue
                    summary.append(dict(condition=condition, layer=layer_id, head=h, metric=name,
                        examples=int(valid.sum()), mean_mass=float(values[valid].mean()),
                        mean_uniform_baseline=float(expected[valid].mean()),
                        mean_enrichment=float((values[valid] / expected[valid]).mean())))
    pd.DataFrame(summary).to_csv(output / "attention_summary.csv", index=False)
    metrics = dict(status="completed", samples=len(indices), tag=job["tag"], key=key, is_smoke=bool(limit),
                   plan_sha256=plan["plan_sha256"],
                   capture="diagnostic Q/K softmax; native SDPA model outputs unchanged",
                   evaluation="original reference batch membership and DynamicPadding multiple=8",
                   sample_auc=float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
                   sample_ap=float(average_precision_score(labels, probabilities)) if labels.any() else None,
                   sample_accuracy_at_0_5=float(np.mean((probabilities >= .5) == labels)),
                   full_test_auc_reference=read_json(Path(job["reference"]) / "metrics.json")["test"]["auc"],
                   max_reference_probability_error=float(reference_error.max()),
                   mean_reference_probability_error=float(reference_error.mean()), causal=causal,
                   layer_ids=layer_ids, counterfactuals={}, wall_seconds=time.monotonic() - started)
    for variant in sorted({p["variant"] for p in perturbed}):
        rows = [p for p in perturbed if p["variant"] == variant]
        p, old = np.array([r["probability"] for r in rows]), np.array([r["original_probability"] for r in rows])
        target = np.array([r["expected_label"] for r in rows])
        metrics["counterfactuals"][variant] = dict(samples=len(rows),
            accuracy_at_0_5=float(np.mean((p >= .5) == target)),
            prediction_flip_rate=float(np.mean((p >= .5) != (old >= .5))),
            mean_absolute_probability_change=float(np.abs(p - old).mean()))
    progress("plotting", samples_completed=len(indices))
    plot_results(output, summary, matrices, [sequences[i].split("\x1f") for i in indices[:3]], layer_ids, key)
    verify_job(plan, job)
    metrics["artifact_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file() and p.name != "status.json"}
    write_json(output / "metrics.json", metrics)
    progress("completed", samples_completed=len(indices))
    return metrics


def plot_results(output, summary, matrices, letters, layer_ids, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    table = pd.DataFrame(summary)
    all_rows = table[table.condition == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("key_to_key", "key_to_inversion", "key_to_noninversion", "key_to_distractor"):
        grouped = all_rows[all_rows.metric == name].groupby("layer")
        for axis, column in zip(axes, ("mean_mass", "mean_enrichment")):
            values = grouped[column].mean()
            axis.plot(values.index, values.values, marker=".", label=name)
    axes[0].set_ylabel("Mean attention mass")
    axes[1].set_ylabel("Mean enrichment over uniform visible tokens")
    axes[1].axhline(1, color="gray", linestyle="--")
    for axis in axes:
        axis.set_xlabel("Physical layer (zero based)")
    axes[0].legend(fontsize=7)
    fig.suptitle(title + " | descriptive, standard attention layers only", fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"layer_profiles.{suffix}", dpi=150)
    plt.close(fig)
    selected = sorted(set([0, len(layer_ids) // 2, len(layer_ids) - 1]))
    for i, example in enumerate(matrices):
        fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4), squeeze=False)
        maximum = max(float(example[j].max()) for j in selected)
        for axis, j in zip(axes[0], selected):
            im = axis.imshow(example[j], vmin=0, vmax=maximum, cmap="viridis")
            axis.set_xticks(range(20), letters[i], fontsize=7)
            axis.set_yticks(range(20), letters[i], fontsize=7)
            axis.set_title(f"Layer {layer_ids[j]}")
            axis.set_xlabel("Target event")
            axis.set_ylabel("Query event")
        fig.colorbar(im, ax=list(axes[0]), shrink=.7, label="Head-mean mass; prompt mass excluded")
        fig.suptitle(f"{title}, fixed sample {i}; not selected by prediction", fontsize=10)
        fig.savefig(output / f"example_{i}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--job", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    analyze_job(args.run_root, args.job, limit=8 if args.smoke else None)


if __name__ == "__main__":
    main()
