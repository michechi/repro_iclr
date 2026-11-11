"""OC oracle and causal-mask-aware lag attention."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.paper_data import oc_label, validate_rule

METRICS = ("key_to_lag_key", "key_to_other_key", "key_to_distractor", "key_to_self",
           "key_to_prompt_special", "readout_to_key", "readout_to_distractor", "readout_to_prompt_special")


def resolve_oc_rule(plan, *, features_path=None, job=None):
    """Prefer an explicit plan rule; otherwise load the dataset manifest rule."""
    rule = plan.get("rule") or plan.get("oc_rule")
    if rule is None and features_path is not None:
        manifest = Path(features_path).resolve().parent / "paper_dataset_manifest.json"
        if manifest.is_file():
            rule = json.loads(manifest.read_text()).get("rule")
    if rule is None and job is not None and job.get("task") == "oc":
        source = plan["datasets"]["oc"][job["condition"]]["test"]
        manifest = Path(source["features"]).resolve().parent / "paper_dataset_manifest.json"
        if manifest.is_file():
            rule = json.loads(manifest.read_text()).get("rule")
    if rule is None:
        raise ValueError("OC attention requires plan['rule'] or a dataset paper_dataset_manifest.json")
    validate_rule(rule)
    if rule["task"] != "oc":
        raise ValueError("OC attention requires an OC rule")
    return rule


def label(sequence, keys, lag):
    letters = sequence.split("\x1f") if isinstance(sequence, str) else list(sequence)
    if len(letters) < 2:
        raise ValueError("OC requires a nonempty sequence of length at least 2")
    return oc_label(letters, keys, lag)


def oc_statistics(attention, groups, letters, keys, lag, causal=True, readout=None):
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        raise ValueError("expected finite [heads,tokens,tokens]")
    size = a.shape[-1]
    allowed = np.tri(size, dtype=bool) if causal else np.ones((size, size), bool)
    if (np.any(a < -1e-7) or np.max(np.abs(a.sum(-1) - 1)) > .025
            or np.any(np.abs(a[:, ~allowed]) > 1e-6)):
        raise ValueError("invalid attention probabilities or causal mask")
    if not 1 <= lag < len(letters):
        raise ValueError("OC lag must lie in [1, n-1]")
    key_set = set(keys)
    uniform = allowed / allowed.sum(-1, keepdims=True)
    ki = [i for i, c in enumerate(letters) if c in key_set]
    key_tokens = {t for i in ki for t in groups[i]}
    event_tokens = {t for g in groups for t in g}
    distractors, special = event_tokens - key_tokens, set(range(size)) - event_tokens
    masses, baselines = [], []
    for metric in METRICS:
        values, expected = [], []
        for i in ([None] if metric.startswith("readout_") else ki):
            queries = [size - 1 if readout is None else readout] if i is None else groups[i]
            if metric == "key_to_lag_key":
                targets = {t for j in ki if abs(i - j) == lag for t in groups[j]}
            elif metric == "key_to_other_key":
                targets = {t for j in ki if j != i and abs(i - j) != lag for t in groups[j]}
            elif metric.endswith("distractor"):
                targets = distractors
            elif metric.endswith("prompt_special"):
                targets = special
            elif metric.endswith("self"):
                targets = set(groups[i])
            else:
                targets = key_tokens
            token_mask = np.zeros(size, bool)
            token_mask[list(targets)] = True
            visible = allowed[queries] & token_mask[None, :]
            base = (uniform[queries] * visible).sum(-1).mean()
            if base > 0:
                values.append((a[:, queries] * visible[None]).sum(-1).mean(-1))
                expected.append(base)
        masses.append(np.mean(values, axis=0) if values else np.full(a.shape[0], np.nan))
        baselines.append(float(np.mean(expected)) if expected else np.nan)
    return np.stack(masses), np.array(baselines)


def perturbations(sequence, keys, lag, seed):
    """Generate count-preserving sequence perturbations and their clean OC labels."""
    letters = sequence.split("\x1f")
    old_label = label(letters, keys, lag)
    rng = np.random.default_rng(seed)
    shuffled = "\x1f".join(rng.permutation(letters).tolist())
    result = {"random_shuffle": (shuffled, label(shuffled, keys, lag))}
    for _ in range(64):
        changed = "\x1f".join(rng.permutation(letters).tolist())
        if label(changed, keys, lag) != old_label:
            result["count_matched_label_flip"] = (changed, 1 - old_label)
            break
    key_set = set(keys)
    positions = [i for i, c in enumerate(letters) if c not in key_set]
    pair = next(((i, j) for i in positions for j in positions if i < j and letters[i] != letters[j]), None)
    if pair:
        changed = letters.copy()
        i, j = pair
        changed[i], changed[j] = changed[j], changed[i]
        text = "\x1f".join(changed)
        if label(text, keys, lag) != old_label:
            raise AssertionError("distractor-only swap changed the OC mechanism")
        result["distractor_swap"] = (text, old_label)
    return result
