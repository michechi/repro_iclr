"""
K-gram baseline analysis on MIMIC CKD→ESRD cohort.
Adapted from simulation/stat_test_ngram.py.

For each k in 1..K_MAX: learns P(Y=1|g) for every k-gram g on the training set,
predicts each test sequence by averaging its k-gram probabilities,
and reports AUC, F1, Precision, Recall, plus coverage metrics.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data" / "processed" / "ckd_cohort_ccs.csv"
OUTPUT = ROOT / "results" / "mimic" / "kgram_analysis.csv"

SEED = 42
MAX_K = 7
N_SEEDS = 5  # multiple seeds for confidence

np.random.seed(SEED)


# ── utilities ─────────────────────────────────────────────────────────

def extract_kgrams(tokens, k):
    """Extract k-grams from a list of tokens."""
    return [
        "-".join(tokens[i:i + k])
        for i in range(len(tokens) - k + 1)
    ] if len(tokens) >= k else []


def kgram_predict(tokens, probs, k):
    """Predict P(Y=1) for a sequence by averaging its k-gram probabilities."""
    kgrams = extract_kgrams(tokens, k)
    if not kgrams:
        return 0.5
    return np.mean([probs.get(g, 0.5) for g in kgrams])


# ── main ──────────────────────────────────────────────────────────────

def run_single_seed(df, seed, max_k=MAX_K):
    """Run k-gram analysis for a single train/test split."""
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )

    results = []
    for k in range(1, max_k + 1):
        # Learn P(Y=1|g) from training set
        stats = defaultdict(lambda: [0, 0])
        train_kgrams_set = set()
        for tokens, label in zip(train_df["tokens"], train_df["label"]):
            for g in extract_kgrams(tokens, k):
                stats[g][int(label)] += 1
                train_kgrams_set.add(g)

        probs = {
            g: c[1] / (c[0] + c[1]) if (c[0] + c[1]) > 0 else 0.5
            for g, c in stats.items()
        }

        # Coverage: what fraction of test k-gram tokens were seen in train
        test_total = 0
        test_covered = 0
        test_kgrams_set = set()
        for tokens in test_df["tokens"]:
            grams = extract_kgrams(tokens, k)
            for g in grams:
                test_total += 1
                test_kgrams_set.add(g)
                if g in train_kgrams_set:
                    test_covered += 1

        token_coverage = test_covered / test_total if test_total > 0 else 0.0
        type_coverage = len(test_kgrams_set & train_kgrams_set) / len(test_kgrams_set) if test_kgrams_set else 0.0

        # Predict on test set
        preds = [kgram_predict(tokens, probs, k) for tokens in test_df["tokens"]]
        y_true = test_df["label"].values

        auc = roc_auc_score(y_true, preds)

        # Find best F1 threshold
        thresholds = np.linspace(0, 1, 101)
        f1s = [f1_score(y_true, (np.array(preds) >= t).astype(int), zero_division=0) for t in thresholds]
        best_idx = int(np.argmax(f1s))
        best_thr = thresholds[best_idx]
        preds_bin = (np.array(preds) >= best_thr).astype(int)

        results.append({
            "k": k,
            "seed": seed,
            "n_unique_kgrams_train": len(train_kgrams_set),
            "n_unique_kgrams_test": len(test_kgrams_set),
            "token_coverage": token_coverage,
            "type_coverage": type_coverage,
            "AUC": auc,
            "F1": f1s[best_idx],
            "Precision": precision_score(y_true, preds_bin, zero_division=0),
            "Recall": recall_score(y_true, preds_bin, zero_division=0),
            "Best_threshold": best_thr,
        })

    return results


def main():
    df = pd.read_csv(COHORT)
    df["tokens"] = df["codes"].apply(json.loads)
    print(f"Cohort: {len(df)} patients  (Y=1: {df['label'].sum()}, "
          f"Y=0: {(df['label']==0).sum()})")

    all_results = []
    seeds = [SEED + i for i in range(N_SEEDS)]

    for s in seeds:
        print(f"\n--- Seed {s} ---")
        res = run_single_seed(df, s)
        all_results.extend(res)
        for r in res:
            print(f"  k={r['k']}  AUC={r['AUC']:.4f}  F1={r['F1']:.4f}  "
                  f"cov_token={r['token_coverage']:.4f}  "
                  f"cov_type={r['type_coverage']:.4f}  "
                  f"({r['n_unique_kgrams_train']:,} train k-grams)")

    # Aggregate across seeds
    raw_df = pd.DataFrame(all_results)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(OUTPUT.with_name("kgram_analysis_raw.csv"), index=False)

    agg = raw_df.groupby("k").agg(
        n_kgrams_train=("n_unique_kgrams_train", "mean"),
        token_coverage_mean=("token_coverage", "mean"),
        token_coverage_std=("token_coverage", "std"),
        type_coverage_mean=("type_coverage", "mean"),
        type_coverage_std=("type_coverage", "std"),
        AUC_mean=("AUC", "mean"),
        AUC_std=("AUC", "std"),
        F1_mean=("F1", "mean"),
        F1_std=("F1", "std"),
        Precision_mean=("Precision", "mean"),
        Recall_mean=("Recall", "mean"),
    ).reset_index()

    agg.to_csv(OUTPUT, index=False)

    print("\n" + "=" * 80)
    print("K-GRAM DIAGNOSTIC TABLE  (CKD→ESRD, MIMIC-IV 3.1)")
    print("=" * 80)
    print(f"{'k':>2s}  {'|V_k|':>10s}  {'Cov(token)':>12s}  {'Cov(type)':>12s}  "
          f"{'AUC':>14s}  {'F1':>14s}")
    print("-" * 80)
    for _, r in agg.iterrows():
        print(f"{int(r['k']):2d}  {r['n_kgrams_train']:>10,.0f}  "
              f"{r['token_coverage_mean']:>6.1%}±{r['token_coverage_std']:>4.1%}  "
              f"{r['type_coverage_mean']:>6.1%}±{r['type_coverage_std']:>4.1%}  "
              f"{r['AUC_mean']:.4f}±{r['AUC_std']:.4f}  "
              f"{r['F1_mean']:.4f}±{r['F1_std']:.4f}")
    print("-" * 80)
    print(f"Averaged over {N_SEEDS} random 80/20 splits.")
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()
