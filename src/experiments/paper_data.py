"""Task definitions, synthetic dataset generation, and oracle metrics for OC, KIP, and GPC."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import string
from pathlib import Path

ALPHABET = string.ascii_uppercase
SEP = "\x1f"


def tokens(sequence):
    return sequence.split(SEP) if isinstance(sequence, str) and SEP in sequence else list(sequence)


def parse_keys(keys):
    """Accept an ordered key sequence as a string or iterable of symbols."""
    if keys is None:
        raise ValueError("keys are required")
    if isinstance(keys, str):
        return list(keys)
    return list(keys)


def oc_label(sequence, keys, lag):
    """Definition 3.3: existence of an ordered, entire maximal key run."""
    row, rank = tokens(sequence), {c: i for i, c in enumerate(keys)}
    if not 1 <= lag < len(row):
        raise ValueError("OC lag must lie in [1,n-1]")
    for residue in range(lag):
        chain = []
        for i in range(residue, len(row), lag):
            if row[i] in rank:
                chain.append(rank[row[i]])
            else:
                if len(chain) >= 2 and all(a <= b for a, b in zip(chain, chain[1:])):
                    return 1
                chain = []
        if len(chain) >= 2 and all(a <= b for a, b in zip(chain, chain[1:])):
            return 1
    return 0


def kip_label(sequence, keys):
    """Definition 3.5 Eq.(4), positive for odd inversion count."""
    row = tokens(sequence)
    if any(row.count(c) != 1 for c in keys):
        raise ValueError("every KIP key must occur exactly once")
    rank = {c: i for i, c in enumerate(keys)}
    order = [rank[c] for c in row if c in rank]
    return sum(a > b for i, a in enumerate(order) for b in order[i + 1:]) % 2


def gpc_label(sequence, keys):
    """Definition 3.7; zero occurrences count as even for every key."""
    return int(sum(c in keys for c in tokens(sequence)) % 2 == len(keys) % 2)


def latent_label(sequence, rule):
    task = rule["task"]
    if task == "oc":
        return oc_label(sequence, rule["keys"], rule["lag"])
    if task == "kip":
        return kip_label(sequence, rule["keys"])
    if task == "gpc":
        return gpc_label(sequence, rule["keys"])
    raise ValueError(f"unknown task {task}")


def sample_rule(task, seed, *, n=20, m=6, alphabet=ALPHABET, lag=7, noise=0., keys=None):
    """Build a task rule. Keys are sampled from the alphabet unless provided explicitly."""
    alphabet = list(alphabet) if not isinstance(alphabet, str) else list(alphabet)
    if (task not in {"oc", "kip", "gpc"} or len(alphabet) < 2
            or len(set(alphabet)) != len(alphabet)
            or any(not isinstance(c, str) or not c for c in alphabet)):
        raise ValueError("invalid task/alphabet")
    if keys is not None:
        keys = parse_keys(keys)
        m = len(keys)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (n, m)):
        raise ValueError("sequence length and key count must be integers")
    if n < 2 or not 1 <= m <= len(alphabet) or (task != "oc" and m == len(alphabet)) or not 0 <= noise < .5:
        raise ValueError("require n>=2, 1<=m<=ell (strict for GPC/KIP), and 0<=noise<.5")
    if task == "kip" and not 3 <= m <= n:
        raise ValueError("KIP requires 3<=m<=n")
    if task == "oc" and not 1 <= lag < n:
        raise ValueError("invalid lag")
    if keys is None:
        keys = random.Random(seed).sample(list(alphabet), m)
    elif any(key not in alphabet for key in keys):
        raise ValueError("explicit keys must belong to the alphabet")
    elif len(set(keys)) != len(keys):
        raise ValueError("explicit keys must be distinct")
    rule = {"task": task, "rule_seed": seed, "n": n, "m": m, "alphabet": alphabet,
            "keys": keys, "lag": lag if task == "oc" else None,
            "noise_probability": noise, "label_convention": "odd_inversions" if task == "kip" else task}
    validate_rule(rule)
    return rule


def validate_rule(rule):
    if not isinstance(rule, dict):
        raise ValueError("rule must be an object")
    if rule.get("task") not in {"oc", "kip", "gpc"}:
        raise ValueError("unknown task in rule")
    alphabet, keys = rule.get("alphabet", []), rule.get("keys", [])
    if len(alphabet) < 2 or len(set(alphabet)) != len(alphabet) or any(not isinstance(c, str) or not c for c in alphabet):
        raise ValueError("invalid rule alphabet")
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= set(alphabet):
        raise ValueError("invalid rule keys")
    n = rule.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("invalid rule length")
    noise = rule.get("noise_probability")
    if not isinstance(noise, (int, float)) or not 0 <= noise < .5:
        raise ValueError("invalid rule noise")
    if "m" in rule and rule["m"] != len(keys):
        raise ValueError("rule key count mismatch")
    if rule["task"] == "kip" and (not 3 <= len(keys) <= n or len(keys) >= len(alphabet)):
        raise ValueError("invalid KIP key count")
    if rule["task"] == "gpc" and len(keys) >= len(alphabet):
        raise ValueError("GPC requires a proper key subset")
    if rule["task"] == "oc" and (not isinstance(rule.get("lag"), int)
            or isinstance(rule["lag"], bool) or not 1 <= rule["lag"] < n):
        raise ValueError("invalid OC lag")


def shuffle_sequences(sequences, seed):
    """Permute each sequence independently while preserving its token multiset C(X)."""
    rng = random.Random(seed)
    shuffled = []
    for seq in sequences:
        row = list(tokens(seq))
        rng.shuffle(row)
        shuffled.append(SEP.join(row) if isinstance(seq, str) else row)
    return shuffled


def generate_rows(rule, count, sequence_seed, noise_seed):
    validate_rule(rule)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("row count must be a nonnegative integer")
    rng, noise_rng = random.Random(sequence_seed), random.Random(noise_seed)
    alphabet, keys, n = rule["alphabet"], rule["keys"], rule["n"]
    distractors = [c for c in alphabet if c not in keys]
    for _ in range(count):
        if rule["task"] == "kip":
            row = rng.choices(distractors, k=n)
            for position, key in zip(rng.sample(range(n), len(keys)), keys):
                row[position] = key
        else:
            row = rng.choices(alphabet, k=n)
        latent = latent_label(row, rule)
        flip = int(noise_rng.random() < rule["noise_probability"])
        yield SEP.join(row), latent, latent ^ flip


def oracle_metrics(rho, noise):
    if not 0 < rho < 1 or not 0 <= noise < .5:
        raise ValueError("Theorem A.1 requires 0<rho<1 and 0<=noise<.5")
    q = noise + (1 - 2 * noise) * rho
    return {"auc": .5 + rho * (1 - rho) * (1 - 2 * noise) / (2 * q * (1 - q)),
            "precision": 1 - noise, "recall": (1 - noise) * rho / q,
            "f1": 2 * (1 - noise) * rho / (noise + 2 * (1 - noise) * rho), "observed_prevalence": q}


def oc_prevalence(n, m, lag, alphabet_size=26):
    """Exact finite-state probability of any ordered maximal run under P0."""
    if not 1 <= lag < n or not 1 <= m <= alphabet_size:
        raise ValueError("invalid OC configuration")
    no_hit = 1.
    for residue in range(lag):
        length = len(range(residue, n, lag))
        states = {(None, False, True): 1.}
        for _ in range(length):
            nxt = {}
            for (last, long, ordered), prob in states.items():
                if last is None or not (long and ordered):
                    state = (None, False, True)
                    nxt[state] = nxt.get(state, 0.) + prob * (alphabet_size - m) / alphabet_size
                for rank in range(m):
                    state = (rank, last is not None, ordered and (last is None or last <= rank))
                    nxt[state] = nxt.get(state, 0.) + prob / alphabet_size
            states = nxt
        no_hit *= sum(prob for (last, long, ordered), prob in states.items() if not (long and ordered))
    return 1 - no_hit


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_csv(path, column):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [column]:
            raise ValueError(f"{path}: expected only {column}")
        return [row[column] for row in reader]


def canonical_gpc_indices(labels, target_rows, seed=0):
    """Select a stratified subsample of binary GPC rows in source order."""
    import numpy as np
    labels = np.asarray(labels)
    if not 0 < target_rows <= len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("invalid binary GPC subsample")
    groups = [np.flatnonzero(labels == label) for label in (0, 1)]
    numerators = np.asarray([len(group) * target_rows for group in groups], dtype=np.int64)
    quotas = numerators // len(labels)
    remainder = target_rows - int(quotas.sum())
    for label in sorted((0, 1), key=lambda label: (-(numerators[label] % len(labels)), label))[:remainder]:
        quotas[label] += 1
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(group, size=int(quota), replace=False)
        for group, quota in zip(groups, quotas)])).astype(np.int64)


def read_paper_split(directory, tag, split, *, rule_override=None, subsample_rows=None, subsample_seed=0,
                     label_binding="oracle"):
    """Load sequences, binary labels, and split metadata for a synthetic task.

    The labeling rule must come from ``paper_dataset_manifest.json`` or an
    explicit ``rule_override``. No task-specific key sets are hardcoded.

    ``label_binding``:
      - ``oracle``: observed labels must match the rule on the loaded sequences
        (or latent files under noise).
      - ``preserved``: keep the stored labels even if sequences were shuffled
        (Section 5 order controls); still verifies alphabet/length against the rule.
    """
    import numpy as np
    if label_binding not in {"oracle", "preserved"}:
        raise ValueError("label_binding must be 'oracle' or 'preserved'")
    directory = Path(directory)
    if split not in {"train", "val", "test"}:
        raise ValueError("unknown split")
    manifest_path = directory / "paper_dataset_manifest.json"
    manifest_digest = sha256(manifest_path) if manifest_path.is_file() else None
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    xp, yp = (directory / f"{kind}_{split}_{tag}.csv" for kind in ("X", "y"))
    input_hashes = {p.name: sha256(p) for p in (xp, yp)}
    seqs = _read_csv(xp, "Sequences")
    raw = np.asarray([float(v) for v in _read_csv(yp, "Outcome")])
    if not len(seqs) or len(seqs) != len(raw) or not np.isin(raw, [0, 1]).all():
        raise ValueError("nonempty aligned binary labels required")
    raw = raw.astype(np.int64)
    if rule_override is not None:
        rule = dict(rule_override)
    elif manifest is not None:
        if manifest.get("tag") != tag or manifest.get("protocol") != "paper_data_v1":
            raise ValueError("dataset manifest identity mismatch")
        for name, digest in input_hashes.items():
            if manifest.get("files_sha256", {}).get(name) != digest:
                raise ValueError(f"dataset hash mismatch: {name}")
        rule = manifest.get("rule")
    else:
        raise ValueError("dataset requires paper_dataset_manifest.json or an explicit rule_override")
    validate_rule(rule)
    if any(len(tokens(s)) != rule["n"] or any(c not in rule["alphabet"] for c in tokens(s)) for s in seqs):
        raise ValueError("sequence shape/alphabet does not match the paper rule")
    sequence_latent = np.asarray([latent_label(s, rule) for s in seqs], dtype=np.int64)
    noise = rule["noise_probability"]
    labels = raw.copy()
    if label_binding == "preserved":
        # Order-control CSVs keep the pre-shuffle labels; do not re-derive them from sequences.
        latent = raw.copy() if not noise else None
        if noise:
            if manifest is None:
                raise ValueError("noisy preserved-label data requires a dataset manifest with latent labels")
            lp = directory / f"y_latent_{split}_{tag}.csv"
            if manifest.get("files_sha256", {}).get(lp.name) != sha256(lp):
                raise ValueError("latent-label hash mismatch")
            latent = np.asarray([float(v) for v in _read_csv(lp, "Outcome")], dtype=np.int64)
            if not np.isin(latent, [0, 1]).all():
                raise ValueError("latent labels must be binary")
            input_hashes[lp.name] = sha256(lp)
    else:
        latent = sequence_latent
        if rule["task"] == "kip" and not noise:
            if not np.array_equal(raw, latent):
                raise ValueError("KIP labels do not match the declared parity convention")
        elif not noise and not np.array_equal(raw, latent):
            raise ValueError("deterministic labels disagree with the paper oracle")
        if noise:
            if manifest is None:
                raise ValueError("noisy data requires a dataset manifest with latent labels")
            lp = directory / f"y_latent_{split}_{tag}.csv"
            if manifest.get("files_sha256", {}).get(lp.name) != sha256(lp):
                raise ValueError("latent-label hash mismatch")
            clean = np.asarray([float(v) for v in _read_csv(lp, "Outcome")])
            clean_evidence = {lp.name: sha256(lp)}
            if not np.isin(clean, [0, 1]).all() or not np.array_equal(clean, latent):
                raise ValueError("clean labels disagree with maximal-run oracle")
            input_hashes.update(clean_evidence)
    for name, digest in input_hashes.items():
        if sha256(directory / name) != digest:
            raise ValueError("input changed while loading")
    source_rows = len(labels)
    selection = {"strategy": "all_rows", "seed": None}
    indices = np.arange(source_rows, dtype=np.int64)
    if subsample_rows is not None:
        if not isinstance(subsample_rows, int) or isinstance(subsample_rows, bool) or subsample_rows < 1:
            raise ValueError("subsample_rows must be a positive integer")
        if source_rows == 2 * subsample_rows and rule["task"] == "gpc":
            # Remark A.4: stratify on latent Y*, not the (possibly noisy) observed label.
            indices = canonical_gpc_indices(latent, subsample_rows, seed=subsample_seed)
            selection = {"strategy": "stratified_without_replacement_largest_remainder_sorted_on_latent",
                         "seed": subsample_seed, "rng": "numpy.Generator.PCG64"}
        elif source_rows != subsample_rows:
            raise ValueError(f"unexpected split size {source_rows}; expected {subsample_rows}")
    selection["indices_sha256"] = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
    selection["indices_hash_encoding"] = "little-endian int64 source row indices, C order"
    seqs = [seqs[int(i)] for i in indices]
    labels, latent = labels[indices], latent[indices]
    evidence = {"features": {"path": str(xp.resolve()), "sha256": input_hashes[xp.name]},
                "labels": {"path": str(yp.resolve()), "sha256": input_hashes[yp.name]},
                "task": rule["task"], "tag": tag, "split": split, "rows": len(labels), "rule": rule,
                "source_rows": source_rows, "returned_rows": len(labels), "source_selection": selection,
                "source_sha256": input_hashes, "label_binding": label_binding,
                "label_convention": "odd_inversions" if rule["task"] == "kip" else rule["task"],
                "noise_probability": noise,
                "realized_flip_rate": float(np.mean(labels != latent)),
                "latent_prevalence": float(latent.mean())}
    if manifest is not None:
        if sha256(manifest_path) != manifest_digest:
            raise ValueError("dataset manifest changed while loading")
        evidence["dataset_manifest_sha256"] = manifest_digest
    return seqs, labels, evidence
