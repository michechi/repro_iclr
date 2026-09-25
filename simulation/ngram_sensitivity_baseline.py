"""
Run k-gram baseline classifier on all sensitivity datasets.

Computes n-gram baselines (1-gram through 7-gram) with AUC, F1, precision,
recall, and coverage for each sensitivity config. Outputs a summary CSV.

Usage:
    python -m simulation.ngram_sensitivity_baseline \
        --data_dir data/simulation/sensitivity/ \
        --output results_ngram_sensitivity.csv

    # Single dataset
    python -m simulation.ngram_sensitivity_baseline \
        --data_dir data/simulation/sensitivity/ \
        --only 102 \
        --output results_ngram_102.csv
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# All sensitivity dataset IDs
ALL_IDS = [100, 101, 102, 103, 110, 112, 120, 121, 122, 123, 124,
           130, 131, 132, 133, 134, 140, 141, 142, 143, 144]

# Config metadata for the results table
CONFIG_META = {
    100: (10, 6, 7), 101: (15, 6, 7), 102: (20, 6, 7), 103: (30, 6, 7),
    130: (20, 3, 1), 131: (20, 3, 3), 132: (20, 3, 5), 110: (20, 3, 7),
    133: (20, 3, 9), 134: (20, 3, 10),
    120: (20, 6, 1), 121: (20, 6, 3), 122: (20, 6, 5),
    123: (20, 6, 9), 124: (20, 6, 10),
    140: (20, 10, 1), 141: (20, 10, 3), 142: (20, 10, 5), 112: (20, 10, 7),
    143: (20, 10, 9), 144: (20, 10, 10),
}


def extract_ngrams(seq, n, sep='\x1f'):
    """Extract all n-grams from a sequence."""
    letters = seq.split(sep)
    if len(letters) < n:
        return []
    return ["-".join(letters[i:i+n]) for i in range(len(letters)-n+1)]


def ngram_baseline(X_train, y_train, X_test, y_test, n):
    """
    Run n-gram baseline classifier.

    Returns dict with AUC, best F1, precision, recall, coverage, threshold.
    """
    # Calculate P(label=1 | n-gram) from training data
    ngram_stats = defaultdict(lambda: [0, 0])

    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        ngrams = extract_ngrams(seq, n)
        for ng in ngrams:
            ngram_stats[ng][int(label)] += 1

    ngram_probs = {}
    for ng, counts in ngram_stats.items():
        total = sum(counts)
        ngram_probs[ng] = counts[1] / total if total > 0 else 0.5

    # Coverage: % of test n-grams seen in training
    train_ngrams = set(ngram_stats.keys())
    test_ngrams = set()
    for seq in X_test['Sequences']:
        test_ngrams.update(extract_ngrams(seq, n))
    coverage = len(test_ngrams & train_ngrams) / len(test_ngrams) * 100 if test_ngrams else 0

    # Predict on test set
    test_preds_prob = []
    for seq in X_test['Sequences']:
        ngrams = extract_ngrams(seq, n)
        probs = [ngram_probs.get(ng, 0.5) for ng in ngrams]
        avg_prob = np.mean(probs) if probs else 0.5
        test_preds_prob.append(avg_prob)

    # AUC
    try:
        auc = roc_auc_score(y_test['Outcome'], test_preds_prob)
    except ValueError:
        auc = 0.5

    # Find best threshold for F1
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [
        f1_score(y_test['Outcome'], [1 if p >= t else 0 for p in test_preds_prob], zero_division=0)
        for t in thresholds
    ]
    best_f1 = max(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    # Precision/recall at best threshold
    test_preds_binary = [1 if p >= best_threshold else 0 for p in test_preds_prob]
    precision = precision_score(y_test['Outcome'], test_preds_binary, zero_division=0)
    recall_val = recall_score(y_test['Outcome'], test_preds_binary, zero_division=0)

    return {
        'auc': auc,
        'f1': best_f1,
        'precision': precision,
        'recall': recall_val,
        'threshold': best_threshold,
        'coverage': coverage,
        'unique_ngrams_train': len(ngram_stats),
        'unique_ngrams_test': len(test_ngrams),
    }


def load_dataset(data_dir, dataset_id):
    """Load train/val/test CSVs for a given dataset ID."""
    na_vals = ['', 'None', 'NaN', 'na', 'nan']

    X_train = pd.read_csv(os.path.join(data_dir, f"X_train_{dataset_id}.csv"), na_values=na_vals).fillna('')
    y_train = pd.read_csv(os.path.join(data_dir, f"y_train_{dataset_id}.csv"), na_values=na_vals).fillna('')
    X_test = pd.read_csv(os.path.join(data_dir, f"X_test_{dataset_id}.csv"), na_values=na_vals).fillna('')
    y_test = pd.read_csv(os.path.join(data_dir, f"y_test_{dataset_id}.csv"), na_values=na_vals).fillna('')

    return X_train, y_train, X_test, y_test


def run_all_ngrams_for_dataset(data_dir, dataset_id, ngram_sizes=range(1, 8)):
    """Run all n-gram baselines for a single dataset. Returns list of result dicts."""
    logger.info(f"Dataset {dataset_id}: loading data...")
    X_train, y_train, X_test, y_test = load_dataset(data_dir, dataset_id)

    n, m, lag = CONFIG_META.get(dataset_id, (None, None, None))
    rho = y_train['Outcome'].mean()

    logger.info(f"Dataset {dataset_id}: n={n}, m={m}, λ={lag}, ρ={rho:.4f}, "
                f"train={len(X_train)}, test={len(X_test)}")

    results = []
    for k in ngram_sizes:
        metrics = ngram_baseline(X_train, y_train, X_test, y_test, k)

        result = {
            'dataset_id': dataset_id,
            'n': n,
            'm': m,
            'lag': lag,
            'rho': round(rho, 4),
            'ngram_size': k,
            **metrics
        }
        results.append(result)

        logger.info(f"  {k}-gram: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, "
                     f"Coverage={metrics['coverage']:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="N-gram baseline on sensitivity datasets")
    parser.add_argument("--data_dir", type=str, default="data/simulation/sensitivity/")
    parser.add_argument("--output", type=str, default="results_ngram_sensitivity.csv")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated dataset IDs to process (default: all)")
    parser.add_argument("--ngram_max", type=int, default=7,
                        help="Maximum n-gram size to test (default: 7)")

    args = parser.parse_args()

    if args.only:
        dataset_ids = [int(x) for x in args.only.split(',')]
    else:
        dataset_ids = ALL_IDS

    ngram_sizes = range(1, args.ngram_max + 1)

    all_results = []

    for dataset_id in dataset_ids:
        # Check if data exists
        train_path = os.path.join(args.data_dir, f"X_train_{dataset_id}.csv")
        if not os.path.exists(train_path):
            logger.warning(f"Dataset {dataset_id} not found at {train_path}, skipping")
            continue

        results = run_all_ngrams_for_dataset(args.data_dir, dataset_id, ngram_sizes)
        all_results.extend(results)

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(args.output, index=False)
    logger.info(f"\nResults saved to {args.output}")

    # Print summary: best n-gram AUC per dataset
    print(f"\n{'='*70}")
    print("Summary: Best n-gram AUC per dataset")
    print(f"{'='*70}")
    print(f"{'ID':<6} {'n':<5} {'m':<5} {'λ':<5} {'ρ':<8} {'Best k':<8} {'AUC':<8}")
    print("-" * 50)

    for did in dataset_ids:
        subset = df_results[df_results['dataset_id'] == did]
        if subset.empty:
            continue
        best_row = subset.loc[subset['auc'].idxmax()]
        print(f"{did:<6} {best_row['n']:<5} {best_row['m']:<5} {best_row['lag']:<5} "
              f"{best_row['rho']:<8} {int(best_row['ngram_size']):<8} {best_row['auc']:<8.4f}")


if __name__ == "__main__":
    main()
