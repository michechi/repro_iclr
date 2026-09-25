"""OC oracle and causal-mask-aware lag attention, using canonical data parameters."""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
import typing

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORACLE_SOURCE = ROOT / "repro/src/analysis/mechanism_id/scripts/common.py"
KEYS = "WDQJXU"
LAG = 7
METRICS = ("key_to_lag_key", "key_to_other_key", "key_to_distractor", "key_to_self",
           "key_to_prompt_special", "readout_to_key", "readout_to_distractor", "readout_to_prompt_special")


@lru_cache(maxsize=1)
def canonical_rule():
    # Load the exact canonical function, without running legacy module-level mkdirs
    # or changing sys.path to the separate repro package.
    tree = ast.parse(ORACLE_SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "rule_greedy_monotone_impl")
    namespace = dict(vars(typing))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORACLE_SOURCE), "exec"), namespace)
    return namespace[function.name]


def label(sequence):
    letters = sequence.split("\x1f") if isinstance(sequence, str) else list(sequence)
    if len(letters) != 20 or any(len(c) != 1 or not "A" <= c <= "Z" for c in letters):
        raise ValueError("OC requires 20 uppercase events")
    return canonical_rule()(letters, LAG, frozenset(KEYS), {k: i for i, k in enumerate(KEYS)},
                            tolerance=False, min_chain_length=2)


def oc_statistics(attention, groups, letters, causal=True, readout=None):
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        raise ValueError("expected finite [heads,tokens,tokens]")
    size = a.shape[-1]
    allowed = np.tri(size, dtype=bool) if causal else np.ones((size, size), bool)
    if (np.any(a < -1e-7) or np.max(np.abs(a.sum(-1) - 1)) > .025
            or np.any(np.abs(a[:, ~allowed]) > 1e-6)):
        raise ValueError("invalid attention probabilities or causal mask")
    uniform = allowed / allowed.sum(-1, keepdims=True)
    ki = [i for i, c in enumerate(letters) if c in KEYS]
    key_tokens = {t for i in ki for t in groups[i]}
    event_tokens = {t for g in groups for t in g}
    distractors, special = event_tokens - key_tokens, set(range(size)) - event_tokens
    masses, baselines = [], []
    for metric in METRICS:
        values, expected = [], []
        for i in ([None] if metric.startswith("readout_") else ki):
            queries = [size - 1 if readout is None else readout] if i is None else groups[i]
            if metric == "key_to_lag_key":
                targets = {t for j in ki if abs(i - j) == LAG for t in groups[j]}
            elif metric == "key_to_other_key":
                targets = {t for j in ki if j != i and abs(i - j) != LAG for t in groups[j]}
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


def perturbations(sequence, seed):
    """Counts preserved. Report actual clean oracle labels, never assume a shuffle flips."""
    letters = sequence.split("\x1f")
    old_label = label(letters)
    rng = np.random.default_rng(seed)
    shuffled = "\x1f".join(rng.permutation(letters).tolist())
    result = {"random_shuffle": (shuffled, label(shuffled))}
    for _ in range(64):
        changed = "\x1f".join(rng.permutation(letters).tolist())
        if label(changed) != old_label:
            result["count_matched_label_flip"] = (changed, 1 - old_label)
            break
    positions = [i for i, c in enumerate(letters) if c not in KEYS]
    pair = next(((i, j) for i in positions for j in positions if i < j and letters[i] != letters[j]), None)
    if pair:
        changed = letters.copy()
        i, j = pair
        changed[i], changed[j] = changed[j], changed[i]
        text = "\x1f".join(changed)
        if label(text) != old_label:
            raise AssertionError("distractor-only swap changed the OC mechanism")
        result["distractor_swap"] = (text, old_label)
    return result
