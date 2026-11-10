#!/usr/bin/env python3
"""Section 5 and Appendix E opening GPC helper ladder.

For the GPC regime (ell=26, m=6, n=20):
  (A) Hidden-K — raw letter input and 26-letter counts, label = parity-of-count rule
  (B) Known-K membership sequence — input = per-position 0/1 b_t = 1{X_t in S}
  (C) Known-K count vector — input = per-key integer counts (6-dim) and per-key parity bits
  (D) Total-count parity — input = scalar total key count

All trained with LogisticRegression and a 64-unit MLP, reporting AUC and F1.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.paper_data import read_paper_split, tokens, sha256

DATA_DIR = ROOT / "data/paper/test_just_pair"
RESULTS_DIR = ROOT / "results/paper_gpc_helper"
SEED = 20260421


def load_real_parity(rows_train: int = 400_000, rows_eval: int = 50_000, *, data_dir=DATA_DIR):
    splits = {}
    keys = None
    for split, rows in [("train", rows_train), ("val", rows_eval), ("test", rows_eval)]:
        X, y, evidence = read_paper_split(data_dir, "test_just_pair", split)
        if len(y) != rows:
            raise ValueError(f"GPC helper requires {rows} {split} rows")
        if (evidence["rule"]["task"] != "gpc" or evidence["rule"]["n"] != 20
                or "".join(evidence["rule"]["alphabet"]) != string.ascii_uppercase
                or evidence["rule"]["noise_probability"] != 0):
            raise ValueError("GPC helper requires noiseless length-20 data over A-Z")
        split_keys = tuple(evidence["rule"]["keys"])
        if len(split_keys) != 6:
            raise ValueError("GPC helper requires six keys")
        if keys is not None and set(split_keys) != set(keys):
            raise ValueError("GPC helper requires the same key set across splits")
        keys = split_keys
        splits[split] = (X, y, evidence)
    return splits


def _feat_letter_onehot_flat(toks, n: int, alphabet_size: int = 26):
    import numpy as np
    arr = np.zeros((n, alphabet_size), dtype=np.float32)
    for i, t in enumerate(toks):
        if i >= n or len(t) != 1:
            continue
        a = ord(t) - ord("A")
        if 0 <= a < alphabet_size:
            arr[i, a] = 1.0
    return arr.reshape(-1)


def _feat_count26(toks, alphabet_size: int = 26):
    import numpy as np
    c = np.zeros(alphabet_size, dtype=np.float32)
    for t in toks:
        if len(t) == 1:
            a = ord(t) - ord("A")
            if 0 <= a < alphabet_size:
                c[a] += 1
    return c


def _feat_key_count(toks, keys):
    import numpy as np
    idx = {k: i for i, k in enumerate(keys)}
    c = np.zeros(len(keys), dtype=np.float32)
    for t in toks:
        if t in idx:
            c[idx[t]] += 1
    return c


def _feat_key_count_parity(toks, keys):
    c = _feat_key_count(toks, keys)
    return (c % 2).astype(c.dtype)


def _feat_membership_bits(toks, keys, n: int):
    import numpy as np
    key_set = frozenset(keys)
    arr = np.zeros(n, dtype=np.float32)
    for i, t in enumerate(toks):
        if i >= n:
            break
        if t in key_set:
            arr[i] = 1.0
    return arr


def _feat_total_key_count(toks, keys):
    import numpy as np
    key_set = frozenset(keys)
    return np.array([sum(1 for t in toks if t in key_set)], dtype=np.float32)


def _eval(model, Xtr, ytr, Xv, yv, Xte, yte):
    import numpy as np
    from sklearn.metrics import f1_score, roc_auc_score
    model.fit(Xtr, ytr)
    if hasattr(model, "predict_proba"):
        pv = model.predict_proba(Xv)[:, 1]
        pt = model.predict_proba(Xte)[:, 1]
    else:
        pv = model.decision_function(Xv)
        pt = model.decision_function(Xte)
    if len(np.unique(yte)) >= 2:
        auc = float(roc_auc_score(yte, pt))
    else:
        auc = float("nan")
    thrs = np.arange(10, 90) / 100
    if len(np.unique(yv)) >= 2:
        scores = [f1_score(yv, pv >= t) for t in thrs]
        bt = thrs[int(np.argmax(scores))]
        f1 = float(f1_score(yte, pt >= bt))
    else:
        f1 = float("nan")
    return auc, f1


def run_paper_regime(data_dir=DATA_DIR, provenance=None):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    data = load_real_parity(data_dir=data_dir)
    if provenance is not None:
        provenance.update({split: values[2] for split, values in data.items()})
    ytr = data["train"][1]
    yv = data["val"][1]
    yte = data["test"][1]
    keys = tuple(data["train"][2]["rule"]["keys"])

    tok_tr = [tokens(s) for s in data["train"][0]]
    tok_v = [tokens(s) for s in data["val"][0]]
    tok_te = [tokens(s) for s in data["test"][0]]

    variants = {
        "A_hidden_letters_onehot": (lambda t: _feat_letter_onehot_flat(t, 20, 26)),
        "A_hidden_count26":        (lambda t: _feat_count26(t, 26)),
        "B_known_membership_bits": (lambda t: _feat_membership_bits(t, keys, 20)),
        "C_known_key_count":       (lambda t: _feat_key_count(t, keys)),
        "C_known_key_count_parity": (lambda t: _feat_key_count_parity(t, keys)),
        "D_total_key_count":       (lambda t: _feat_total_key_count(t, keys)),
    }

    rows = []
    for vname, ex in variants.items():
        Xtr = np.stack([ex(t) for t in tok_tr], 0)
        Xv = np.stack([ex(t) for t in tok_v], 0)
        Xte = np.stack([ex(t) for t in tok_te], 0)

        for mname, model in [
            ("logreg", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)),
            ("mlp_64", MLPClassifier(hidden_layer_sizes=(64,), random_state=SEED,
                                     max_iter=200, early_stopping=True)),
        ]:
            t0 = time.time()
            auc, f1 = _eval(model, Xtr, ytr, Xv, yv, Xte, yte)
            dt = time.time() - t0
            rows.append({
                "regime": "paper_m6_l26_n20",
                "variant": vname, "model": mname,
                "feat_dim": int(Xtr.shape[1]),
                "AUC": auc, "F1": f1, "train_sec": dt,
                "rho_test": float(np.mean(yte)),
            })
    return rows


def run_sweep(ell_list=(4, 8, 12, 16, 26), m_ratios=(0.5,), n_list=(20,), n_rows=50_000):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    alph_all = list(string.ascii_uppercase)
    rng_master = np.random.default_rng(SEED)
    rows = []
    for ell in ell_list:
        alph = alph_all[:ell]
        for r in m_ratios:
            m = max(1, int(r * ell))
            K = tuple(alph[:m])
            key_set = frozenset(K)
            for n in n_list:
                rng = np.random.default_rng(int(rng_master.integers(0, 2**31)))
                idx = rng.integers(0, ell, size=(n_rows, n))
                arr = np.array(alph)[idx]
                seqs = [list(row) for row in arr]
                tot = np.array([sum(1 for t in s if t in key_set) for s in seqs])
                y = (tot % 2 == m % 2).astype(int)

                def _onehot(s):
                    a = np.zeros((n, ell), dtype=np.float32)
                    for i, t in enumerate(s):
                        a[i, alph.index(t)] = 1.0
                    return a.reshape(-1)

                def _count(s):
                    a = np.zeros(ell, dtype=np.float32)
                    for t in s:
                        a[alph.index(t)] += 1
                    return a

                def _bits(s):
                    return np.array([1.0 if t in key_set else 0.0 for t in s], dtype=np.float32)

                variants = {
                    "A_hidden_onehot": _onehot,
                    "A_hidden_count": _count,
                    "B_known_membership_bits": _bits,
                }

                split = n_rows // 2
                split2 = (3 * n_rows) // 4
                for vname, ex in variants.items():
                    X_all = np.stack([ex(s) for s in seqs], 0)
                    Xtr, ytr = X_all[:split], y[:split]
                    Xv, yv = X_all[split:split2], y[split:split2]
                    Xte, yte = X_all[split2:], y[split2:]
                    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)
                    auc, f1 = _eval(model, Xtr, ytr, Xv, yv, Xte, yte)
                    rows.append({
                        "regime": f"sweep_ell{ell}_m{m}_n{n}",
                        "ell": ell, "m": m, "n": n, "p": m / ell,
                        "variant": vname, "model": "logreg",
                        "feat_dim": int(X_all.shape[1]),
                        "AUC": auc, "F1": f1, "rho_test": float(np.mean(yte)),
                    })
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Appendix E helper ladder. --execute fits the estimators.")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "gpc_helpers.csv")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--skip_paper", action="store_true")
    p.add_argument("--skip_sweep", action="store_true")
    p.add_argument("--sweep_rows", type=int, default=50_000)
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--execute", action="store_true")
    execution.add_argument("--dry-run", action="store_true", help="print the helper specification (the default)")
    args = p.parse_args(argv)
    if args.skip_paper and args.skip_sweep:
        p.error("at least one helper analysis must be selected")
    if args.sweep_rows < 8:
        p.error("--sweep_rows must be at least 8")
    return args


def build_plan(args):
    return {"protocol_id": "paper_gpc_helper_v1",
            "paper_reference": "Section 5 and Appendix E helper ladder",
            "paper_helper_enabled": not args.skip_paper, "sweep_enabled": not args.skip_sweep,
            "seed": SEED,
            "input_data": str(args.data_dir), "output_csv": str(args.out),
            "paper_split_rows": {"train": 400000, "val": 50000, "test": 50000},
            "keys": None, "n": 20, "ell": 26,
            "feature_dimensions": {"letter_onehot": 520, "letter_counts": 26, "membership_bits": 20,
                                   "key_counts": 6, "key_parity": 6, "total_key_count": 1},
            "models": {"logreg": {"solver": "lbfgs", "max_iter": 2000},
                       "mlp": {"hidden_layer_sizes": [64], "max_iter": 200, "early_stopping": True}},
            "threshold_selection": "validation F1 on .10,.11,...,.89",
            "sweep_rows": args.sweep_rows, "sweep_train_val_test_fraction": [.5, .25, .25],
            "sweep_ell": [4, 8, 12, 16, 26], "sweep_m": "floor(ell/2)",
            "label_rule": "Definition 3.7: sum(membership_bits) % 2 == len(keys) % 2"}


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    import pandas as pd
    manifest_path = args.out.with_suffix(".manifest.json")
    if args.out.exists() or manifest_path.exists():
        raise FileExistsError("output CSV or manifest path already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance = {}
    manifest = {**plan,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                    (Path(__file__).resolve(), ROOT / "src/experiments/paper_data.py")}}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    if not args.skip_paper:
        rows.extend(run_paper_regime(args.data_dir, provenance))
        manifest["keys"] = provenance["train"]["rule"]["keys"]
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if not args.skip_sweep:
        rows.extend(run_sweep(n_rows=args.sweep_rows))
        pd.DataFrame(rows).to_csv(args.out, index=False)
    manifest.update(status="completed", data=provenance, output_sha256=sha256(args.out))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
