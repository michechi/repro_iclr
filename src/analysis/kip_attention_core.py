"""Pure array operations for checkpoint-only KIP attention inspection."""
from __future__ import annotations

import numpy as np

KEYS = {"kip_m4": "COTA", "kip_m6": "GHLYIZ"}
PREFIX = "Sequential events: "
METRICS = ("key_to_key", "key_to_inversion", "key_to_noninversion",
           "key_to_distractor", "key_to_self", "key_to_prompt_special",
           "readout_to_key", "readout_to_distractor", "readout_to_prompt_special")


def validate_sequence(sequence, tag):
    letters = sequence.split("\x1f")
    if len(letters) != 20 or any(len(c) != 1 or not "A" <= c <= "Z" for c in letters):
        raise ValueError("expected 20 uppercase letters")
    keys = KEYS[tag]
    if any(letters.count(c) != 1 for c in keys):
        raise ValueError("each KIP key must occur exactly once")
    ranks = [keys.index(c) for c in letters if c in keys]
    inversions = sum(a > b for i, a in enumerate(ranks) for b in ranks[i + 1:])
    return letters, int(inversions % 2 == 0)


def letter_token_groups(offsets, letters):
    """Map character spans, including split tokens, without matching prompt letters."""
    groups = []
    for i, letter in enumerate(letters):
        start = len(PREFIX) + 2 * i
        group = [t for t, (a, b) in enumerate(offsets) if a < start + len(letter) and b > start]
        if not group:
            raise ValueError(f"unmapped event {i}")
        groups.append(group)
    flat = [t for g in groups for t in g]
    if len(set(flat)) != len(flat):
        raise ValueError("one token spans multiple events; event aggregation would be ambiguous")
    return groups


def counterfactuals(sequence, tag):
    letters, label = validate_sequence(sequence, tag)
    positions = [i for i, c in enumerate(letters) if c in KEYS[tag]]
    swapped = letters.copy()
    i, j = positions[:2]
    swapped[i], swapped[j] = swapped[j], swapped[i]
    key_swap = "\x1f".join(swapped)
    if validate_sequence(key_swap, tag)[1] != 1 - label:
        raise AssertionError("a transposition must flip inversion parity")
    distractors = [i for i, c in enumerate(letters) if c not in KEYS[tag]]
    pair = next(((i, j) for i in distractors for j in distractors if i < j and letters[i] != letters[j]), None)
    result = {"key_swap": (key_swap, 1 - label)}
    if pair:
        swapped = letters.copy()
        i, j = pair
        swapped[i], swapped[j] = swapped[j], swapped[i]
        sequence2 = "\x1f".join(swapped)
        if validate_sequence(sequence2, tag)[1] != label:
            raise AssertionError("distractor swap changed key parity")
        result["distractor_swap"] = (sequence2, label)
    return result


def attention_statistics(attention, groups, letters, tag, causal, readout):
    """Per-head mass and uniform-visible-token baseline, averaged over key events.

    Masked future tokens are excluded from both mass and baseline. Queries of
    split events are averaged; target subtoken masses are summed. Self attention
    means the complete event, not only an individual token's diagonal.
    """
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 3 or a.shape[1] != a.shape[2] or not np.isfinite(a).all():
        raise ValueError("expected finite [heads,tokens,tokens] attention")
    size = a.shape[-1]
    allowed = np.tri(size, dtype=bool) if causal else np.ones((size, size), dtype=bool)
    if np.any(a < -1e-7) or np.max(np.abs(a.sum(-1) - 1)) > .025:
        raise ValueError("attention rows must be probabilities")
    if np.any(np.abs(a[:, ~allowed]) > 1e-6):
        raise ValueError("causal attention contains future-token mass")
    uniform = allowed / allowed.sum(-1, keepdims=True)
    keys = KEYS[tag]
    ki = [i for i, c in enumerate(letters) if c in keys]
    key_tokens = {t for i in ki for t in groups[i]}
    event_tokens = {t for g in groups for t in g}
    distractor_tokens = event_tokens - key_tokens
    special = set(range(size)) - event_tokens
    masses, baselines = [], []
    for metric in METRICS:
        values, expected = [], []
        query_events = [None] if metric.startswith("readout_") else ki
        for i in query_events:
            queries = [readout] if i is None else groups[i]
            if metric.endswith("prompt_special"):
                targets = special
            elif metric.endswith("distractor"):
                targets = distractor_tokens
            elif metric.endswith("self"):
                targets = set(groups[i])
            elif metric in ("key_to_inversion", "key_to_noninversion"):
                targets = set()
                for j in ki:
                    if j == i:
                        continue
                    first, second = sorted((i, j))
                    inverted = keys.index(letters[first]) > keys.index(letters[second])
                    if inverted == (metric == "key_to_inversion"):
                        targets.update(groups[j])
            else:
                targets = key_tokens - (set(groups[i]) if i is not None else set())
            token_mask = np.zeros(size, dtype=bool)
            token_mask[list(targets)] = True
            visible = allowed[queries] & token_mask[None, :]
            expected_mass = (uniform[queries] * visible).sum(-1).mean()
            if expected_mass > 0:
                values.append((a[:, queries, :] * visible[None, :, :]).sum(-1).mean(-1))
                expected.append(expected_mass)
        masses.append(np.mean(values, axis=0) if values else np.full(a.shape[0], np.nan))
        baselines.append(float(np.mean(expected)) if expected else np.nan)
    return np.stack(masses), np.array(baselines)


def event_matrix(attention, groups):
    """Head-mean event attention; prompt/special mass is deliberately not renormalized."""
    a = np.asarray(attention).mean(0)
    return np.array([[a[np.ix_(q, t)].sum(-1).mean() for t in groups] for q in groups])


def fixed_sample(labels, count, seed):
    """Proportional label-stratified subset shared by every model on a task."""
    labels = np.asarray(labels)
    if count < 2 or count > len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("invalid binary sample size/labels")
    n1 = min(count - 1, max(1, int(round(count * float(labels.mean())))))
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == c), n, replace=False)
                                   for c, n in [(0, count - n1), (1, n1)]]))
