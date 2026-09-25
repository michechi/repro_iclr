"""
Signal decomposition analysis: order-invariant content, local order and full-model AUC

Quantifies how much of each task is solvable from:
  1. Order-invariant content (Bag-of-Events = 26-dim letter-count vector)
  2. Short local order   (k-gram baselines, k = 1..7)
  3. Richer sequence modeling (full models — numbers from paper)

Produces:
  - Table 1: Signal decomposition (BoE / k-gram / full-model AUC)
  - Table 2: Matched-histogram counterfactual via key-letter stratification
              (groups sequences by key-letter counts to isolate order signal)
"""

import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/simulation/tested")
OUTPUT_DIR = Path("paper_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

DELIM = "\x1f"
ALPHABET = list(string.ascii_uppercase)        # 26 letters
SEED = 42
np.random.seed(SEED)

DATASETS = {
    "Tricky Det.": "6",
    "Tricky Rnd.": "9",
    "Parity":      "test_just_pair",
}

# Full-model AUC from paper (Table 1, test-set)
FULL_MODEL_AUC = {
    "Tricky Det.": {
        "BERT": 0.999, "LSTM": 0.999, "Transformer": 0.999,
        "Llama-1B": 0.846, "Llama-8B": 0.995,
    },
    "Tricky Rnd.": {
        "BERT": 0.669, "LSTM": 0.665, "Transformer": 0.671,
        "Llama-1B": 0.587, "Qwen-14B": 0.670,
    },
    "Parity": {
        "BERT": 0.503, "LSTM": 0.502, "Transformer": 0.501,
        "Llama-1B": 0.497, "Llama-8B": 0.500,
    },
}

AUC_STAR = {"Tricky Det.": 1.000, "Tricky Rnd.": 0.670, "Parity": None}

# Key letters used in the ordering rule (from simulation/test_simulation_det.py)
KEY_LETTERS = ["W", "D", "Q", "J", "X", "N"]
KEY_SET = set(KEY_LETTERS)


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────
def load_dataset(csv_id):
    def _load(split):
        X = pd.read_csv(DATA_DIR / f"X_{split}_{csv_id}.csv").fillna("")
        y = pd.read_csv(DATA_DIR / f"y_{split}_{csv_id}.csv").fillna("")
        return X, y
    return {s: _load(s) for s in ("train", "val", "test")}


# ──────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────────────
def count_vector(seq: str) -> list:
    """26-dim letter-count vector (completely order-invariant)."""
    letters = seq.split(DELIM)
    counts = Counter(letters)
    return [counts.get(c, 0) for c in ALPHABET]


def key_count_vector(seq: str) -> list:
    """6-dim count vector of key letters only (W, D, Q, J, X, N)."""
    letters = seq.split(DELIM)
    counts = Counter(letters)
    return [counts.get(c, 0) for c in KEY_LETTERS]


def total_key_count(seq: str) -> int:
    """Total number of key letters in the sequence."""
    return sum(1 for c in seq.split(DELIM) if c in KEY_SET)


def extract_ngrams(seq: str, n: int) -> list:
    letters = seq.split(DELIM)
    if len(letters) < n:
        return []
    return ["-".join(letters[i:i + n]) for i in range(len(letters) - n + 1)]


# ──────────────────────────────────────────────────────────────────────
# BAG-OF-EVENTS (BoE) BASELINE
# ──────────────────────────────────────────────────────────────────────
def boe_baseline(data):
    """LogisticRegression on 26-dim count vectors (order-invariant)."""
    X_tr = np.array([count_vector(s) for s in data["train"][0].Sequences])
    y_tr = data["train"][1].Outcome.values

    X_val = np.array([count_vector(s) for s in data["val"][0].Sequences])
    y_val = data["val"][1].Outcome.values

    X_te = np.array([count_vector(s) for s in data["test"][0].Sequences])
    y_te = data["test"][1].Outcome.values

    clf = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    # Tune threshold on val, report on test
    preds_val = clf.predict_proba(X_val)[:, 1]
    preds_te = clf.predict_proba(X_te)[:, 1]

    auc_te = roc_auc_score(y_te, preds_te)

    thrs = np.linspace(0, 1, 201)
    best_thr = thrs[np.argmax([f1_score(y_val, preds_val >= t) for t in thrs])]
    f1_te = f1_score(y_te, preds_te >= best_thr)

    return {"AUC": auc_te, "F1": f1_te}, clf


def boe_xgb_baseline(data):
    """XGBoost on 26-dim count vectors (nonlinear, order-invariant)."""
    X_tr = np.array([count_vector(s) for s in data["train"][0].Sequences])
    y_tr = data["train"][1].Outcome.values
    X_te = np.array([count_vector(s) for s in data["test"][0].Sequences])
    y_te = data["test"][1].Outcome.values

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, subsample=0.8,
        random_state=SEED, verbose=0,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_te)[:, 1]
    return {"AUC": roc_auc_score(y_te, preds)}


def boe_key_only_baseline(data):
    """LogReg on 6-dim key-letter count vector only."""
    X_tr = np.array([key_count_vector(s) for s in data["train"][0].Sequences])
    y_tr = data["train"][1].Outcome.values
    X_te = np.array([key_count_vector(s) for s in data["test"][0].Sequences])
    y_te = data["test"][1].Outcome.values

    clf = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    preds = clf.predict_proba(X_te)[:, 1]
    return {"AUC": roc_auc_score(y_te, preds)}


# ──────────────────────────────────────────────────────────────────────
# K-GRAM BASELINE
# ──────────────────────────────────────────────────────────────────────
def kgram_baseline(data, n):
    stats = defaultdict(lambda: [0, 0])
    for seq, label in zip(data["train"][0].Sequences, data["train"][1].Outcome):
        for ng in extract_ngrams(seq, n):
            stats[ng][int(label)] += 1

    probs = {
        ng: c[1] / sum(c) if sum(c) > 0 else 0.5
        for ng, c in stats.items()
    }

    def _predict(X):
        return [
            np.mean([probs.get(ng, 0.5) for ng in extract_ngrams(seq, n)] or [0.5])
            for seq in X.Sequences
        ]

    preds_val = _predict(data["val"][0])
    preds_te = _predict(data["test"][0])

    y_val = data["val"][1].Outcome.values
    y_te = data["test"][1].Outcome.values

    auc_te = roc_auc_score(y_te, preds_te)

    thrs = np.linspace(0, 1, 201)
    best_thr = thrs[np.argmax([f1_score(y_val, [p >= t for p in preds_val]) for t in thrs])]
    f1_te = f1_score(y_te, [p >= best_thr for p in preds_te])

    # Coverage: fraction of test n-grams seen in training
    all_te_ng = set()
    for seq in data["test"][0].Sequences:
        all_te_ng.update(extract_ngrams(seq, n))
    coverage = len(all_te_ng & set(probs)) / max(len(all_te_ng), 1)

    return {"AUC": auc_te, "F1": f1_te, "Coverage": coverage}


# ──────────────────────────────────────────────────────────────────────
# MATCHED-HISTOGRAM COUNTERFACTUAL
# ──────────────────────────────────────────────────────────────────────
def matched_histogram_analysis(data, boe_clf):
    """
    Two-level counterfactual:

    Level 1 — Full 26-dim count vectors.  With n=20 from |Σ|=26 these are
    nearly unique, so exact matching is uninformative.  We report this fact.

    Level 2 — Key-letter count stratification.  Group test sequences by
    the 6-dim count vector of key letters (W,D,Q,J,X,N).  Within each
    stratum, letter counts of the task-relevant letters are identical, so
    any label variation must arise from their ORDER.  We train a BoE on
    key-letter counts and measure how much signal remains unexplained.
    """
    X_te, y_te_df = data["test"]
    y_te = y_te_df.Outcome.values
    n_test = len(y_te)

    # --- Level 1: full 26-dim (for reporting) ---
    cvecs_full = [tuple(count_vector(s)) for s in X_te.Sequences]
    n_unique_full = len(set(cvecs_full))

    # --- Level 2: key-letter count stratification ---
    key_cvecs = [tuple(key_count_vector(s)) for s in X_te.Sequences]
    key_totals = [total_key_count(s) for s in X_te.Sequences]

    groups = defaultdict(lambda: {"idx": [], "labels": []})
    for i, (kv, lab) in enumerate(zip(key_cvecs, y_te)):
        groups[kv]["idx"].append(i)
        groups[kv]["labels"].append(int(lab))

    n_key_groups = len(groups)
    ambig = {kv: g for kv, g in groups.items()
             if len(set(g["labels"])) > 1}
    n_ambig_groups = len(ambig)

    ambig_idx = []
    for g in ambig.values():
        ambig_idx.extend(g["idx"])
    ambig_idx = np.array(ambig_idx) if ambig_idx else np.array([], dtype=int)
    n_ambig = len(ambig_idx)

    # BoE (key-only) on ambiguous subset
    X_te_key = np.array([key_count_vector(s) for s in X_te.Sequences])
    clf_key = LogisticRegression(max_iter=2000, random_state=SEED, solver="lbfgs")
    X_tr_key = np.array([key_count_vector(s) for s in data["train"][0].Sequences])
    y_tr = data["train"][1].Outcome.values
    clf_key.fit(X_tr_key, y_tr)
    preds_key = clf_key.predict_proba(X_te_key)[:, 1]

    if n_ambig > 20 and len(np.unique(y_te[ambig_idx])) == 2:
        boe_auc_ambig = roc_auc_score(y_te[ambig_idx], preds_key[ambig_idx])
    else:
        boe_auc_ambig = float("nan")

    # Full BoE on ambiguous subset
    X_te_cv = np.array([count_vector(s) for s in X_te.Sequences])
    preds_full = boe_clf.predict_proba(X_te_cv)[:, 1]
    if n_ambig > 20 and len(np.unique(y_te[ambig_idx])) == 2:
        boe_full_auc_ambig = roc_auc_score(y_te[ambig_idx], preds_full[ambig_idx])
    else:
        boe_full_auc_ambig = float("nan")

    base_rate = float(np.mean(y_te[ambig_idx])) if n_ambig > 0 else float("nan")

    # --- Total-key-count stratified table ---
    strata = defaultdict(lambda: {"n": 0, "pos": 0})
    for kt, lab in zip(key_totals, y_te):
        strata[kt]["n"] += 1
        strata[kt]["pos"] += int(lab)

    strata_rows = []
    for kt in sorted(strata):
        s = strata[kt]
        rate = s["pos"] / s["n"] if s["n"] > 0 else 0
        strata_rows.append({
            "total_key_count": kt,
            "n_sequences": s["n"],
            "n_positive": s["pos"],
            "base_rate": rate,
        })

    return {
        "n_test": n_test,
        # Level 1
        "n_unique_full_26d": n_unique_full,
        "pct_unique_full": 100.0 * n_unique_full / max(n_test, 1),
        # Level 2
        "n_key_groups": n_key_groups,
        "n_ambiguous_groups": n_ambig_groups,
        "pct_ambiguous_groups": 100.0 * n_ambig_groups / max(n_key_groups, 1),
        "n_ambiguous_seqs": n_ambig,
        "pct_ambiguous_seqs": 100.0 * n_ambig / max(n_test, 1),
        "base_rate_ambig": base_rate,
        "boe_key_auc_ambig": boe_auc_ambig,
        "boe_full_auc_ambig": boe_full_auc_ambig,
        "strata": strata_rows,
    }


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    decomp_rows = []   # Table 1
    matched_rows = []   # Table 2

    for task_name, csv_id in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  {task_name}  (csv_id = {csv_id})")
        print(f"{'='*60}")

        data = load_dataset(csv_id)
        n_train = len(data["train"][1])
        n_test  = len(data["test"][1])
        rho = data["test"][1].Outcome.mean()
        print(f"  Train: {n_train:,}   Test: {n_test:,}   ρ(test): {rho:.4f}")

        # --- BoE (linear) ---
        print("  BoE-linear (LogReg, 26-dim) ...", end=" ", flush=True)
        boe_res, boe_clf = boe_baseline(data)
        print(f"AUC = {boe_res['AUC']:.4f}")

        # --- BoE (XGBoost, nonlinear) ---
        print("  BoE-XGB   (XGBoost, 26-dim) ...", end=" ", flush=True)
        boe_xgb = boe_xgb_baseline(data)
        print(f"AUC = {boe_xgb['AUC']:.4f}")

        # --- BoE (key-only) ---
        print("  BoE-key   (LogReg, 6-dim)   ...", end=" ", flush=True)
        boe_key = boe_key_only_baseline(data)
        print(f"AUC = {boe_key['AUC']:.4f}")

        # --- k-grams ---
        kg_results = {}
        for k in range(1, 8):
            print(f"  {k}-gram ...", end=" ", flush=True)
            kg_results[k] = kgram_baseline(data, k)
            print(f"AUC = {kg_results[k]['AUC']:.4f}  "
                  f"(coverage {kg_results[k]['Coverage']:.1%})")

        # --- assemble decomposition row ---
        row = {"Task": task_name, "ρ": f"{rho:.3f}"}
        row["Chance"] = 0.500
        row["BoE-key (6d)"] = boe_key["AUC"]
        row["BoE-linear (26d)"] = boe_res["AUC"]
        row["BoE-XGB (26d)"] = boe_xgb["AUC"]
        row["1-gram"] = kg_results[1]["AUC"]
        row["2-gram"] = kg_results[2]["AUC"]
        row["3-gram"] = kg_results[3]["AUC"]

        # Best full model
        fm = FULL_MODEL_AUC[task_name]
        best_model = max(fm, key=fm.get)
        row["Best model"] = f"{best_model} ({fm[best_model]:.3f})"
        row["Best model AUC"] = fm[best_model]

        if AUC_STAR[task_name] is not None:
            row["AUC*"] = AUC_STAR[task_name]

        # Signal gaps (use XGB as content ceiling for cleaner decomposition)
        content_ceil = boe_xgb["AUC"]
        row["Δ content"]    = content_ceil - 0.500
        row["Δ local order"] = max(kg_results[2]["AUC"] - content_ceil, 0)
        row["Δ sequential"]  = fm[best_model] - max(content_ceil, kg_results[2]["AUC"])

        decomp_rows.append(row)

        # --- matched histogram ---
        print("  Matched-histogram analysis ...", end=" ", flush=True)
        mh = matched_histogram_analysis(data, boe_clf)
        mh["Task"] = task_name
        matched_rows.append(mh)
        print(f"{mh['pct_ambiguous_seqs']:.1f}% of test seqs in "
              f"ambiguous key-letter groups")

    # ──────────────────────────────────────────────────────────────────
    # OUTPUT
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TABLE 1: SIGNAL DECOMPOSITION (AUC on test set)")
    print("=" * 70)

    hdr = f"{'':22s} | {'Tricky Det.':>12s} | {'Tricky Rnd.':>12s} | {'Parity':>12s}"
    print(hdr)
    print("-" * len(hdr))

    labels = [
        ("Chance",            "Chance"),
        ("BoE-key (6d)",      "BoE-key (6d)"),
        ("BoE-linear (26d)",  "BoE-linear (26d)"),
        ("BoE-XGB (26d)",     "BoE-XGB (26d)"),
        ("1-gram",            "1-gram"),
        ("2-gram",            "2-gram"),
        ("3-gram",            "3-gram"),
        ("Best model AUC",    "Best model"),
    ]
    for key, display in labels:
        vals = []
        for r in decomp_rows:
            v = r.get(key, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        print(f"{display:22s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s}")

    # AUC* row
    vals = []
    for r in decomp_rows:
        v = r.get("AUC*", "—")
        vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
    print(f"{'AUC*':22s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s}")

    print()
    print("Signal gaps (ΔAUC):")
    for key, display in [("Δ content", "Content (XGB−0.5)"),
                         ("Δ local order", "Local order (2g−XGB)"),
                         ("Δ sequential", "Sequential (model−ceil)")]:
        vals = [f"{r[key]:+.3f}" for r in decomp_rows]
        print(f"  {display:28s} | {vals[0]:>12s} | {vals[1]:>12s} | {vals[2]:>12s}")

    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TABLE 2: MATCHED-HISTOGRAM COUNTERFACTUAL")
    print("=" * 70)

    for mh in matched_rows:
        print(f"\n  {mh['Task']}:")
        print(f"    Full 26-dim unique vectors:       {mh['n_unique_full_26d']:,} / "
              f"{mh['n_test']:,}  ({mh['pct_unique_full']:.1f}%)")
        print(f"    Key-letter groups (6-dim):         {mh['n_key_groups']:,}")
        print(f"    Ambiguous groups (both labels):    {mh['n_ambiguous_groups']:,}"
              f"  ({mh['pct_ambiguous_groups']:.1f}% of key groups)")
        print(f"    Sequences in ambiguous groups:     {mh['n_ambiguous_seqs']:,}"
              f"  ({mh['pct_ambiguous_seqs']:.1f}% of test)")
        if not np.isnan(mh["base_rate_ambig"]):
            print(f"    Base rate (ambiguous subset):      {mh['base_rate_ambig']:.3f}")
            print(f"    BoE-key AUC (ambiguous subset):    {mh['boe_key_auc_ambig']:.4f}")
            print(f"    BoE-full AUC (ambiguous subset):   {mh['boe_full_auc_ambig']:.4f}")

        # Print key-count strata
        if mh["strata"]:
            print(f"\n    Key-count strata (total key letters → base rate):")
            print(f"    {'#keys':>5s} | {'N':>7s} | {'ρ':>6s}")
            print(f"    {'-'*5}-+-{'-'*7}-+-{'-'*6}")
            for s in mh["strata"]:
                print(f"    {s['total_key_count']:5d} | "
                      f"{s['n_sequences']:7,d} | "
                      f"{s['base_rate']:.3f}")

    # ──────────────────────────────────────────────────────────────────
    # LaTeX output
    # ──────────────────────────────────────────────────────────────────
    # Table 1 — Signal decomposition
    tex_rows = []
    tex_rows.append(r"\begin{table}[t]")
    tex_rows.append(r"\centering")
    tex_rows.append(r"\small")
    tex_rows.append(r"\caption{Signal decomposition (AUC). "
                    r"\emph{BoE}: logistic regression or XGBoost on the "
                    r"order-invariant letter-count vector; "
                    r"\emph{$k$-gram}: position-aware $k$-gram classifier. "
                    r"$\Delta$ rows show the marginal AUC gain per signal layer.}")
    tex_rows.append(r"\label{tab:signal_decomposition}")
    tex_rows.append(r"\begin{tabular}{l c c c}")
    tex_rows.append(r"\toprule")
    tex_rows.append(r"Method & Tricky Det.\ & Tricky Rnd.\ & Parity \\")
    tex_rows.append(r"\midrule")

    tex_labels = [
        ("Chance",            "Chance"),
        ("BoE-key (6d)",      r"BoE-key (6\,dim)"),
        ("BoE-XGB (26d)",     r"BoE-XGB (26\,dim)"),
        ("2-gram",            "2-gram"),
        ("Best model AUC",    "Best model"),
    ]
    for key, display in tex_labels:
        vals = []
        for r in decomp_rows:
            v = r.get(key, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        tex_rows.append(f"{display} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    # AUC* row
    vals = []
    for r in decomp_rows:
        v = r.get("AUC*", "---")
        vals.append(f"{v:.3f}" if isinstance(v, float) else "---")
    tex_rows.append(r"\midrule")
    tex_rows.append(f"AUC$^*$ & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    # Delta rows
    tex_rows.append(r"\midrule")
    for key, display in [("Δ content", r"$\Delta_{\text{content}}$"),
                         ("Δ local order", r"$\Delta_{\text{local}}$"),
                         ("Δ sequential", r"$\Delta_{\text{seq}}$")]:
        vals = [f"{r[key]:+.3f}" for r in decomp_rows]
        tex_rows.append(f"{display} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    tex_rows.append(r"\bottomrule")
    tex_rows.append(r"\end{tabular}")
    tex_rows.append(r"\end{table}")

    tex1 = "\n".join(tex_rows)
    (OUTPUT_DIR / "signal_decomposition.tex").write_text(tex1)
    print(f"\n  Saved: {OUTPUT_DIR / 'signal_decomposition.tex'}")

    # Table 2 — Matched histogram counterfactual
    tex2_rows = []
    tex2_rows.append(r"\begin{table}[t]")
    tex2_rows.append(r"\centering")
    tex2_rows.append(r"\small")
    tex2_rows.append(r"\caption{Matched-histogram counterfactual. "
                     r"Sequences are grouped by their 6-dimensional key-letter "
                     r"count vector $(|W|,|D|,|Q|,|J|,|X|,|N|)$. "
                     r"\emph{Ambiguous} groups contain both labels, "
                     r"so any within-group discrimination must use sequential order. "
                     r"BoE AUC on the ambiguous subset shows the residual content "
                     r"signal from non-key letters.}")
    tex2_rows.append(r"\label{tab:matched_histogram}")
    tex2_rows.append(r"\begin{tabular}{l c c c}")
    tex2_rows.append(r"\toprule")
    tex2_rows.append(r" & Tricky Det.\ & Tricky Rnd.\ & Parity \\")
    tex2_rows.append(r"\midrule")

    mh_fields = [
        ("n_key_groups",        "Key-letter groups",         ",d"),
        ("n_ambiguous_groups",  "Ambiguous groups",          ",d"),
        ("pct_ambiguous_seqs",  r"Ambiguous seqs (\%)",      ".1f"),
        ("base_rate_ambig",     r"$\rho$ (ambig.\ subset)",  ".3f"),
        ("boe_key_auc_ambig",   "BoE-key AUC (ambig.)",      ".3f"),
        ("boe_full_auc_ambig",  "BoE-full AUC (ambig.)",     ".3f"),
    ]
    for field, display, fmt in mh_fields:
        vals = []
        for mh in matched_rows:
            v = mh.get(field, float("nan"))
            if isinstance(v, float) and not np.isnan(v):
                vals.append(f"{v:{fmt}}")
            elif isinstance(v, int):
                vals.append(f"{v:{fmt}}")
            else:
                vals.append("---")
        tex2_rows.append(f"{display} & {vals[0]} & {vals[1]} & {vals[2]} \\\\")

    tex2_rows.append(r"\bottomrule")
    tex2_rows.append(r"\end{tabular}")
    tex2_rows.append(r"\end{table}")

    tex2 = "\n".join(tex2_rows)
    (OUTPUT_DIR / "matched_histogram.tex").write_text(tex2)
    print(f"  Saved: {OUTPUT_DIR / 'matched_histogram.tex'}")

    # ──────────────────────────────────────────────────────────────────
    # Save raw numbers for downstream use
    # ──────────────────────────────────────────────────────────────────
    pd.DataFrame(decomp_rows).to_csv(
        OUTPUT_DIR / "signal_decomposition.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(
        OUTPUT_DIR / "matched_histogram.csv", index=False)
    print(f"  Saved CSVs to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
