"""
Shuffle test for CKD→ESRD sequential dependence.

Compares model performance on original (temporally ordered) sequences vs.
randomly permuted sequences within each patient. If performance is similar,
the task does not require sequential information.

Models tested:
  1. XGBoost on bag-of-codes (order-invariant control — should be identical)
  2. K-gram classifiers k=1,2,3 (order-sensitive — k=1 is invariant, k≥2 sensitive)
  3. Logistic regression on bag-of-codes (simple baseline control)

Multiple shuffle seeds are used for confidence intervals.
"""

import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data" / "processed" / "ckd_cohort_ccs.csv"
OUTPUT = ROOT / "results" / "mimic" / "shuffle_test.csv"

SEED = 42
N_SHUFFLE_SEEDS = 10


# ── features ──────────────────────────────────────────────────────────

def bag_of_codes_features(token_lists, vocab):
    """Binary bag-of-codes matrix."""
    vocab_idx = {v: i for i, v in enumerate(vocab)}
    X = np.zeros((len(token_lists), len(vocab)), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        for t in tokens:
            if t in vocab_idx:
                X[i, vocab_idx[t]] = 1
    return X


def extract_kgrams(tokens, k):
    if len(tokens) < k:
        return []
    return ["-".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def kgram_predict(tokens, probs, k):
    grams = extract_kgrams(tokens, k)
    if not grams:
        return 0.5
    return np.mean([probs.get(g, 0.5) for g in grams])


def train_kgram_classifier(train_tokens, train_labels, k):
    stats = defaultdict(lambda: [0, 0])
    for tokens, label in zip(train_tokens, train_labels):
        for g in extract_kgrams(tokens, k):
            stats[g][int(label)] += 1
    probs = {
        g: c[1] / (c[0] + c[1]) if (c[0] + c[1]) > 0 else 0.5
        for g, c in stats.items()
    }
    return probs


def eval_kgram(test_tokens, test_labels, probs, k):
    preds = [kgram_predict(t, probs, k) for t in test_tokens]
    auc = roc_auc_score(test_labels, preds)
    thresholds = np.linspace(0, 1, 101)
    f1s = [f1_score(test_labels, (np.array(preds) >= t).astype(int), zero_division=0)
           for t in thresholds]
    return auc, max(f1s)


# ── main ──────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(COHORT)
    df["tokens"] = df["codes"].apply(json.loads)

    # Fixed train/test split
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  (Y=1: {train_df['label'].sum()})")
    print(f"Test:  {len(test_df)}  (Y=1: {test_df['label'].sum()})")

    # Build vocabulary from train
    vocab = sorted({t for tokens in train_df["tokens"] for t in tokens})
    print(f"Vocabulary: {len(vocab)} codes")

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    results = []

    # ── Original (ordered) sequences ──────────────────────────────────
    print("\n=== ORIGINAL (ordered) sequences ===")

    # Bag-of-codes features (order-invariant)
    X_train_boc = bag_of_codes_features(train_df["tokens"].tolist(), vocab)
    X_test_boc = bag_of_codes_features(test_df["tokens"].tolist(), vocab)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=SEED, eval_metric="logloss", verbosity=0
    )
    xgb.fit(X_train_boc, y_train)
    xgb_preds = xgb.predict_proba(X_test_boc)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_preds)
    xgb_f1s = [f1_score(y_test, (xgb_preds >= t).astype(int), zero_division=0)
               for t in np.linspace(0, 1, 101)]
    xgb_f1 = max(xgb_f1s)
    print(f"  XGBoost (BoC):     AUC={xgb_auc:.4f}  F1={xgb_f1:.4f}")
    results.append({"model": "XGBoost_BoC", "condition": "original",
                    "shuffle_seed": -1, "AUC": xgb_auc, "F1": xgb_f1})

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED,
                            class_weight="balanced")
    lr.fit(X_train_boc, y_train)
    lr_preds = lr.predict_proba(X_test_boc)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_preds)
    lr_f1s = [f1_score(y_test, (lr_preds >= t).astype(int), zero_division=0)
              for t in np.linspace(0, 1, 101)]
    lr_f1 = max(lr_f1s)
    print(f"  LogReg (BoC):      AUC={lr_auc:.4f}  F1={lr_f1:.4f}")
    results.append({"model": "LogReg_BoC", "condition": "original",
                    "shuffle_seed": -1, "AUC": lr_auc, "F1": lr_f1})

    # K-gram classifiers on original
    for k in [1, 2, 3]:
        probs = train_kgram_classifier(train_df["tokens"].tolist(), y_train, k)
        auc, f1 = eval_kgram(test_df["tokens"].tolist(), y_test, probs, k)
        print(f"  {k}-gram classifier: AUC={auc:.4f}  F1={f1:.4f}")
        results.append({"model": f"{k}-gram", "condition": "original",
                        "shuffle_seed": -1, "AUC": auc, "F1": f1})

    # ── Shuffled sequences ────────────────────────────────────────────
    print(f"\n=== SHUFFLED sequences ({N_SHUFFLE_SEEDS} seeds) ===")

    for shuf_seed in range(N_SHUFFLE_SEEDS):
        rng = np.random.RandomState(shuf_seed)

        train_tokens_shuf = [list(rng.permutation(t)) for t in train_df["tokens"]]
        test_tokens_shuf = [list(rng.permutation(t)) for t in test_df["tokens"]]

        # BoC is invariant — XGBoost and LogReg don't need retraining
        # but we verify on first shuffle seed as sanity check
        if shuf_seed == 0:
            X_train_shuf = bag_of_codes_features(train_tokens_shuf, vocab)
            X_test_shuf = bag_of_codes_features(test_tokens_shuf, vocab)
            assert np.array_equal(X_train_boc, X_train_shuf), "BoC should be shuffle-invariant"
            assert np.array_equal(X_test_boc, X_test_shuf), "BoC should be shuffle-invariant"
            print("  [sanity check] BoC features identical after shuffle ✓")

        # Record BoC models as invariant (same result)
        results.append({"model": "XGBoost_BoC", "condition": "shuffled",
                        "shuffle_seed": shuf_seed, "AUC": xgb_auc, "F1": xgb_f1})
        results.append({"model": "LogReg_BoC", "condition": "shuffled",
                        "shuffle_seed": shuf_seed, "AUC": lr_auc, "F1": lr_f1})

        # K-gram classifiers on shuffled
        for k in [1, 2, 3]:
            probs = train_kgram_classifier(train_tokens_shuf, y_train, k)
            auc, f1 = eval_kgram(test_tokens_shuf, y_test, probs, k)
            results.append({"model": f"{k}-gram", "condition": "shuffled",
                            "shuffle_seed": shuf_seed, "AUC": auc, "F1": f1})

        print(f"  seed={shuf_seed}: ", end="")
        seed_res = [r for r in results
                    if r["shuffle_seed"] == shuf_seed and r["condition"] == "shuffled"]
        for r in seed_res:
            if r["model"] in ["2-gram", "3-gram"]:
                print(f"{r['model']} AUC={r['AUC']:.4f}  ", end="")
        print()

    # ── Summary ───────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 85)
    print("SHUFFLE TEST RESULTS  (CKD→ESRD, MIMIC-IV 3.1)")
    print("=" * 85)
    print(f"{'Model':<18s}  {'Original AUC':>14s}  {'Shuffled AUC':>20s}  {'Δ AUC':>14s}")
    print("-" * 85)

    for model in ["XGBoost_BoC", "LogReg_BoC", "1-gram", "2-gram", "3-gram"]:
        orig = res_df[(res_df["model"] == model) & (res_df["condition"] == "original")]
        shuf = res_df[(res_df["model"] == model) & (res_df["condition"] == "shuffled")]
        orig_auc = orig["AUC"].values[0]
        shuf_mean = shuf["AUC"].mean()
        shuf_std = shuf["AUC"].std()
        delta = orig_auc - shuf_mean
        print(f"{model:<18s}  {orig_auc:>14.4f}  "
              f"{shuf_mean:>10.4f}±{shuf_std:.4f}  "
              f"{delta:>+14.4f}")

    print("-" * 85)
    print(f"Shuffled results averaged over {N_SHUFFLE_SEEDS} random permutations.")
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
