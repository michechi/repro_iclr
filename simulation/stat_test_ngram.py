"""
Baseline analysis for ordered vs perturbed sequences
Reproducible n-gram baseline evaluation

Produces:
- n-gram baseline table (main paper)
- cumulative prefix table
- top discriminative bigrams (appendix)
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------

DATA_DIR = Path("data/simulation")
OUTPUT_DIR = Path("paper_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_ID = "test_just_pair"
DELIM = "\x1f"
MAX_NGRAM = 7
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------

def load_split(name):
    X = pd.read_csv(DATA_DIR / f"X_{name}_{CSV_ID}.csv").fillna("")
    y = pd.read_csv(DATA_DIR / f"y_{name}_{CSV_ID}.csv").fillna("")
    return X, y

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")

# ------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------

def extract_ngrams(seq, n):
    letters = seq.split(DELIM)
    return [
        "-".join(letters[i:i+n])
        for i in range(len(letters) - n + 1)
    ] if len(letters) >= n else []

def ngram_predict(seq, probs, n):
    ngrams = extract_ngrams(seq, n)
    return np.mean([probs.get(ng, 0.5) for ng in ngrams]) if ngrams else 0.5

# ------------------------------------------------------------------
# N-GRAM BASELINE
# ------------------------------------------------------------------

def evaluate_ngram(X_train, y_train, X_val, y_val, n):
    stats = defaultdict(lambda: [0, 0])

    for seq, label in zip(X_train.Sequences, y_train.Outcome):
        for ng in extract_ngrams(seq, n):
            stats[ng][int(label)] += 1

    probs = {
        ng: c[1] / sum(c) if sum(c) > 0 else 0.5
        for ng, c in stats.items()
    }

    preds = [
        ngram_predict(seq, probs, n)
        for seq in X_val.Sequences
    ]

    auc = roc_auc_score(y_val.Outcome, preds)

    thresholds = np.linspace(0, 1, 101)
    f1s = [
        f1_score(y_val.Outcome, [p >= t for p in preds])
        for t in thresholds
    ]

    return {
        "AUC": auc,
        "F1 (thr=0.5)": f1_score(y_val.Outcome, [p >= 0.5 for p in preds]),
        "Best F1": max(f1s),
        "Best threshold": thresholds[np.argmax(f1s)]
    }

# ------------------------------------------------------------------
# PREFIX ANALYSIS
# ------------------------------------------------------------------

def prefix_analysis(X_train, y_train, X_val, y_val, max_k=20):
    results = []

    for k in range(1, max_k + 1):
        train_pref = [
            DELIM.join(seq.split(DELIM)[:k])
            for seq in X_train.Sequences
        ]
        val_pref = [
            DELIM.join(seq.split(DELIM)[:k])
            for seq in X_val.Sequences
        ]

        stats = defaultdict(lambda: [0, 0])
        for p, y in zip(train_pref, y_train.Outcome):
            stats[p][int(y)] += 1

        preds = [
            stats[p][1] / sum(stats[p]) if sum(stats[p]) > 0 else 0.5
            for p in val_pref
        ]

        results.append({
            "Prefix length": k,
            "AUC": roc_auc_score(y_val.Outcome, preds),
            "Unique prefixes": len(set(train_pref)),
            "Coverage (%)": 100 * len(set(val_pref) & set(train_pref)) / len(set(val_pref))
        })

    return pd.DataFrame(results)

# ------------------------------------------------------------------
# BIGRAM ANALYSIS (APPENDIX)
# ------------------------------------------------------------------

def bigram_discriminative_analysis(X_train, y_train, top_k=10):
    bg0, bg1, tot = Counter(), Counter(), Counter()

    for seq, y in zip(X_train.Sequences, y_train.Outcome):
        bgs = extract_ngrams(seq, 2)
        (bg1 if y else bg0).update(bgs)
        tot.update(bgs)

    p0 = {k: v / sum(bg0.values()) for k, v in bg0.items()}
    p1 = {k: v / sum(bg1.values()) for k, v in bg1.items()}
    pt = {k: v / sum(tot.values()) for k, v in tot.items()}

    rows = []
    for bg in set(p0) | set(p1):
        rows.append([
            bg,
            pt.get(bg, 0),
            p0.get(bg, 0),
            p1.get(bg, 0),
            abs(p1.get(bg, 0) - p0.get(bg, 0))
        ])

    df = pd.DataFrame(
        rows,
        columns=["Bigram", "P(bg)", "P(bg|0)", "P(bg|1)", "Δ"]
    ).sort_values("Δ", ascending=False)

    return df.head(top_k)

# ------------------------------------------------------------------
# RUN + SAVE TABLES
# ------------------------------------------------------------------

# 1) N-gram table (MAIN) — includes 1-gram
ngram_results = {
    n: evaluate_ngram(X_train, y_train, X_val, y_val, n)
    for n in range(1, MAX_NGRAM + 1)
}

df_ngram = (
    pd.DataFrame(ngram_results)
      .T
      .reset_index()
      .rename(columns={"index": "n"})
)

df_ngram["Model"] = df_ngram["n"].apply(
    lambda n: "1-gram (unigram)" if n == 1 else f"{n}-gram"
)

df_ngram = df_ngram[
    ["Model", "AUC", "F1 (thr=0.5)", "Best F1", "Best threshold"]
]

df_ngram.to_latex(
    OUTPUT_DIR / f"ngram_baselines_{CSV_ID}.tex",
    index=False,
    float_format="%.4f",
    caption="Performance of n-gram baseline classifiers, including unigram baseline.",
    label="tab:ngram_baselines"
)

# 2) Prefix table
df_prefix = prefix_analysis(X_train, y_train, X_val, y_val)
df_prefix.to_latex(
    OUTPUT_DIR / f"prefix_baselines_{CSV_ID}.tex",
    index=False,
    float_format="%.4f",
    caption="Cumulative prefix baselines.",
    label="tab:prefix_baselines"
)

# 3) Bigram appendix
df_bigram = bigram_discriminative_analysis(X_train, y_train)
df_bigram.to_latex(
    OUTPUT_DIR / f"top_bigrams_{CSV_ID}.tex",
    index=False,
    float_format="%.4f",
    caption="Most discriminative bigrams (appendix).",
    label="tab:top_bigrams"
)
