import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Reading data - already splitted!
# 1: lags [7,5,4,2]
# 2: lags [9,8,7,6]

csv_number = 'alph'
X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

# Stratified 50% subsample if csv_number == "test_just_pair"
if csv_number == "test_just_pair":
    from sklearn.model_selection import train_test_split

    def stratified_subsample(X, y, fraction=0.5, random_state=42):
        """Subsample keeping the original class distribution"""
        _, X_sub, _, y_sub = train_test_split(
            X, y, test_size=fraction, stratify=y['Outcome'], random_state=random_state
        )
        return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)

    print(f"Stratified 50% subsample for csv_number='{csv_number}'")
    print(f"  Train: {len(X_train)} -> ", end="")
    X_train, y_train = stratified_subsample(X_train, y_train)
    print(f"{len(X_train)}")

    print(f"  Val:   {len(X_val)} -> ", end="")
    X_val, y_val = stratified_subsample(X_val, y_val)
    print(f"{len(X_val)}")

    print(f"  Test:  {len(X_test)} -> ", end="")
    X_test, y_test = stratified_subsample(X_test, y_test)
    print(f"{len(X_test)}")

df = pd.DataFrame({
    "Sequences":pd.concat([X_train.Sequences, X_val.Sequences, X_test.Sequences]),
    "Outcome":pd.concat([y_train.Outcome, y_val.Outcome, y_test.Outcome])
})
df['Outcome'].value_counts()/len(df)

# Extract first letter
train_first = [seq.split('\x1f')[0] for seq in X_train['Sequences']]
val_first = [seq.split('\x1f')[0] for seq in X_val['Sequences']]
test_first = [seq.split('\x1f')[0] for seq in X_test['Sequences']]


# Compute P(label=1 | letter) from training
letter_stats = defaultdict(lambda: [0, 0])  # [count_label_0, count_label_1]
for letter, label in zip(train_first, y_train['Outcome']):
    letter_stats[letter][int(label)] += 1

# Predict on test
test_preds = []
for letter in test_first:
    counts = letter_stats[letter]
    prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
    test_preds.append(prob)

baseline_auc = roc_auc_score(y_test['Outcome'], test_preds)
print(f"Baseline AUC (first letter only): {baseline_auc:.4f}")

# Test on all letters
for pos in range(len(X_train["Sequences"][0].split("\x1f"))):
    pos_letters = [seq.split('\x1f')[pos] for seq in X_train['Sequences']]
    
    letter_stats = defaultdict(lambda: [0, 0])
    for letter, label in zip(pos_letters, y_train['Outcome']):
        letter_stats[letter][int(label)] += 1
    
    test_pos_letters = [seq.split('\x1f')[pos] for seq in X_test['Sequences']]
    test_preds = []
    for letter in test_pos_letters:
        counts = letter_stats[letter]
        prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
        test_preds.append(prob)
    
    pos_auc = roc_auc_score(y_test['Outcome'], test_preds)
    print(f"Position {pos+1} AUC: {pos_auc:.4f}")

# Test on all letters cumulating
def cumulative_position_analysis(X_train, y_train, X_val, y_val):
    """Test cumulative information: first k letters"""
    
    results = []
    
    for k in range(1, 21):  # From 1 to 9 letters
        # Extract the first k letters from each sequence
        train_prefixes = ['\x1f'.join(seq.split('\x1f')[:k]) for seq in X_train['Sequences']]
        val_prefixes = ['\x1f'.join(seq.split('\x1f')[:k]) for seq in X_val['Sequences']]
        
        # Compute P(label=1 | prefix) from training
        prefix_stats = defaultdict(lambda: [0, 0])
        for prefix, label in zip(train_prefixes, y_train['Outcome']):
            prefix_stats[prefix][int(label)] += 1
        
        # Predict on validation
        val_preds = []
        for prefix in val_prefixes:
            counts = prefix_stats[prefix]
            prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
            val_preds.append(prob)
        
        auc = roc_auc_score(y_val['Outcome'], val_preds)
        
        # Also compute coverage: how many unique prefixes seen in train
        unique_train = len(set(train_prefixes))
        unique_val = len(set(val_prefixes))
        coverage = len(set(val_prefixes) & set(train_prefixes)) / unique_val * 100
        
        print(f"First {k} letters | AUC: {auc:.4f} | "
              f"Unique train: {unique_train:,} | "
              f"Coverage test: {coverage:.1f}%")
        
        results.append({
            'k': k,
            'auc': auc,
            'unique_train': unique_train,
            'coverage': coverage
        })
    
    return results

results = cumulative_position_analysis(X_train, y_train, X_test, y_test)

# Test on bigrams
def extract_bigrams(seq):
    letters = seq.split('\x1f')
    return [f"{letters[i]}-{letters[i+1]}" for i in range(len(letters)-1)]

def bigram_distribution_analysis(X_train, y_train, X_val, y_val):
    """Analyze whether ordered/unordered sequences have different bigram distributions"""
    
    # Count bigrams per class
    bigrams_class_0 = Counter()
    bigrams_class_1 = Counter()
    bigrams_tot = Counter()
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        if label == 0:
            bigrams_class_0.update(bigrams)
        else:
            bigrams_class_1.update(bigrams)
        bigrams_tot.update(bigrams)
    
    # Normalize to probabilities
    total_0 = sum(bigrams_class_0.values())
    total_1 = sum(bigrams_class_1.values())
    total_tot = sum(bigrams_tot.values())
    
    prob_0 = {bg: count/total_0 for bg, count in bigrams_class_0.items()}
    prob_1 = {bg: count/total_1 for bg, count in bigrams_class_1.items()}
    prob_tot={bg: count/total_tot for bg, count in bigrams_tot.items()}
    
    # Find the most discriminative bigrams
    all_bigrams = set(prob_0.keys()) | set(prob_1.keys())
    bigram_scores = []

    for bg in all_bigrams:
        p0 = prob_0.get(bg, 0)
        p1 = prob_1.get(bg, 0)
        ptot=prob_tot.get(bg, 0)
        diff = abs(p1 - p0)
        bigram_scores.append((bg, ptot, p0, p1, diff))
    
    # Sort by difference
    bigram_scores.sort(key=lambda x: x[4], reverse=True)
    
    print("Top 20 most discriminative bigrams:")
    print("Bigram | P(bg) | P(bg|label=0) | P(bg|label=1) | Difference")
    for bg, ptot, p0, p1, diff in bigram_scores[:40]:
        print(f"{bg:8} | {ptot:13.4f} | {p0:13.4f} | {p1:13.4f} | {diff:10.4f}")
    
    # Sort by probability
    bigram_scores.sort(key=lambda x: x[1], reverse=True)
    print("Top 20 most frequent bigrams:")
    print("Bigram | P(bg) | P(bg|label=0) | P(bg|label=1) | Difference")
    for bg, ptot, p0, p1, diff in bigram_scores[:20]:
        print(f"{bg:8} | {ptot:13.4f} | {p0:13.4f} | {p1:13.4f} | {diff:10.4f}")
    
    return bigram_scores

def bigram_baseline_classifier(X_train, y_train, X_val, y_val):
    """Classify based on the bigram distribution in the sequence"""
    
    # Compute P(label=1 | bigram) for each bigram
    bigram_stats = defaultdict(lambda: [0, 0])  # [count_0, count_1]
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        for bg in bigrams:
            bigram_stats[bg][label] += 1
    
    # Convert to probabilities
    bigram_probs = {}
    for bg, counts in bigram_stats.items():
        total = sum(counts)
        bigram_probs[bg] = counts[1] / total if total > 0 else 0.5
    
    # Predict: use the mean of the bigram probabilities in the sequence
    val_preds = []
    for seq in X_val['Sequences']:
        bigrams = extract_bigrams(seq)
        probs = [bigram_probs.get(bg, 0.5) for bg in bigrams]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds.append(avg_prob)
    
    auc = roc_auc_score(y_val['Outcome'], val_preds)
    print(f"\nBigram Baseline AUC: {auc:.4f}")
    
    return auc

# Run the analyses
bigram_scores = bigram_distribution_analysis(X_train, y_train, X_val, y_val)

bigram_auc = bigram_baseline_classifier(X_train, y_train, X_val, y_val)

# Let's use also F1 score
def bigram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val):
    """Classify based on the bigram distribution - with F1"""
    
    # Compute P(label=1 | bigram)
    bigram_stats = defaultdict(lambda: [0, 0])
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        bigrams = extract_bigrams(seq)
        for bg in bigrams:
            bigram_stats[bg][int(label)] += 1
    
    bigram_probs = {}
    for bg, counts in bigram_stats.items():
        total = sum(counts)
        bigram_probs[bg] = counts[1] / total if total > 0 else 0.5
    
    # Predict
    val_preds_prob = []
    for seq in X_val['Sequences']:
        bigrams = extract_bigrams(seq)
        probs = [bigram_probs.get(bg, 0.5) for bg in bigrams]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds_prob.append(avg_prob)
    
    # Compute metrics
    auc = roc_auc_score(y_val['Outcome'], val_preds_prob)
    
    # Binarize with threshold 0.5 for F1
    val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds_prob]
    f1 = f1_score(y_val['Outcome'], val_preds_binary)
    
    # Also try the optimal threshold (the one that maximizes F1)
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in val_preds_prob]
        f1_scores.append(f1_score(y_val['Outcome'], preds))
    
    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    
    print(f"Bigram Baseline AUC: {auc:.4f}")
    print(f"Bigram Baseline F1 (threshold=0.5): {f1:.4f}")
    print(f"Bigram Baseline F1 (best threshold={best_threshold:.2f}): {best_f1:.4f}")
    
    return auc, f1, best_f1

# Run
auc, f1, best_f1 = bigram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val)


def extract_ngrams(seq, n):
    """Extract all n-grams from a sequence"""
    letters = seq.split('\x1f')
    if len(letters) < n:
        return []
    return ["-".join(letters[i:i+n]) for i in range(len(letters)-n+1)]

def ngram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val, n=2):
    """
    Classify based on n-gram distribution - with F1
    
    Parameters:
    -----------
    n : int
        Size of n-gram (2=bigrams, 3=trigrams, etc.)
    """
    # Calculate P(label=1 | n-gram)
    ngram_stats = defaultdict(lambda: [0, 0])
    
    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        ngrams = extract_ngrams(seq, n)
        for ng in ngrams:
            ngram_stats[ng][int(label)] += 1
    
    ngram_probs = {}
    for ng, counts in ngram_stats.items():
        total = sum(counts)
        ngram_probs[ng] = counts[1] / total if total > 0 else 0.5
    
    # Predict
    val_preds_prob = []
    for seq in X_val['Sequences']:
        ngrams = extract_ngrams(seq, n)
        probs = [ngram_probs.get(ng, 0.5) for ng in ngrams]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds_prob.append(avg_prob)
    
    # Calculate metrics
    auc = roc_auc_score(y_val['Outcome'], val_preds_prob)

    # Binarize with threshold 0.5 for F1, precision, recall
    val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds_prob]
    f1 = f1_score(y_val['Outcome'], val_preds_binary)
    precision = precision_score(y_val['Outcome'], val_preds_binary)
    recall = recall_score(y_val['Outcome'], val_preds_binary)

    # Try with optimal threshold (that maximizes F1)
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    precisions = []
    recalls = []
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in val_preds_prob]
        f1_scores.append(f1_score(y_val['Outcome'], preds))
        precisions.append(precision_score(y_val['Outcome'], preds, zero_division=0))
        recalls.append(recall_score(y_val['Outcome'], preds, zero_division=0))

    best_f1 = max(f1_scores)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    ngram_name = {2: 'Bigram', 3: 'Trigram', 4: '4-gram', 5: '5-gram', 6: '6-gram', 7: '7-gram'}
    name = ngram_name.get(n, f'{n}-gram')

    print(f"{name} Baseline AUC: {auc:.4f}")
    print(f"{name} Baseline F1 (threshold=0.5): {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"{name} Baseline F1 (best threshold={best_threshold:.2f}): {best_f1:.4f} | Precision: {best_precision:.4f} | Recall: {best_recall:.4f}")

    return auc, f1, best_f1, best_threshold, precision, recall, best_precision, best_recall

# Run for all n-gram sizes
print("N-gram Baseline Classifier Results")


results = {}
for n in range(2, 8):  # 2 to 7
    print(f"\n{n}-gram results:")
    print("-"*70)
    auc, f1, best_f1, best_thresh, precision, recall, best_precision, best_recall = ngram_baseline_classifier_with_f1(
        X_train, y_train, X_val, y_val, n=n
    )
    results[n] = {
        'auc': auc,
        'f1': f1,
        'best_f1': best_f1,
        'best_threshold': best_thresh,
        'precision': precision,
        'recall': recall,
        'best_precision': best_precision,
        'best_recall': best_recall
    }

# N-gram cumulative analysis (similar to cumulative_position_analysis)
def ngram_cumulative_analysis(X_train, y_train, X_test, y_test, max_n=7):
    """
    N-gram analysis with coverage, similar to cumulative_position_analysis.
    Computes AUC, unique train, unique test and coverage for n from 1 to max_n.
    Coverage = % of test n-grams that were seen in training.
    """

    results = []

    print(f"\n{'='*80}")
    print("N-gram Cumulative Analysis (with coverage)")
    print(f"{'='*80}")
    print(f"{'n':>3} | {'AUC':>8} | {'Unique Train':>14} | {'Unique Test':>12} | {'Coverage Test':>14}")
    print("-"*65)

    for n in range(1, max_n + 1):
        # Extract all n-grams from each sequence
        train_ngrams_per_seq = [extract_ngrams(seq, n) for seq in X_train['Sequences']]
        test_ngrams_per_seq = [extract_ngrams(seq, n) for seq in X_test['Sequences']]

        # Compute P(label=1 | n-gram) from training
        ngram_stats = defaultdict(lambda: [0, 0])
        for ngrams, label in zip(train_ngrams_per_seq, y_train['Outcome']):
            for ng in ngrams:
                ngram_stats[ng][int(label)] += 1

        # Predict on test using the mean of the n-gram probabilities
        test_preds = []
        for ngrams in test_ngrams_per_seq:
            if not ngrams:
                test_preds.append(0.5)
                continue
            probs = []
            for ng in ngrams:
                counts = ngram_stats[ng]
                prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
                probs.append(prob)
            test_preds.append(np.mean(probs))

        auc = roc_auc_score(y_test['Outcome'], test_preds)

        # Compute unique counts and coverage
        unique_train_ngrams = set()
        for ngrams in train_ngrams_per_seq:
            unique_train_ngrams.update(ngrams)

        unique_test_ngrams = set()
        for ngrams in test_ngrams_per_seq:
            unique_test_ngrams.update(ngrams)

        unique_train = len(unique_train_ngrams)
        unique_test = len(unique_test_ngrams)
        # Coverage: % of test n-grams seen in training
        coverage = len(unique_test_ngrams & unique_train_ngrams) / unique_test * 100 if unique_test > 0 else 0

        ngram_name = {1: 'Unigram', 2: 'Bigram', 3: 'Trigram'}.get(n, f'{n}-gram')
        print(f"{n:>3} | {auc:>8.4f} | {unique_train:>14,} | {unique_test:>12,} | {coverage:>13.1f}%")

        results.append({
            'n': n,
            'name': ngram_name,
            'auc': auc,
            'unique_train': unique_train,
            'unique_test': unique_test,
            'coverage': coverage
        })

    return results

# Run the cumulative n-gram analysis
ngram_results = ngram_cumulative_analysis(X_train, y_train, X_test, y_test)


# =============================================================================
# N-gram Fraction Experiment (similar to XGBoost_fraction_experiment.py)
# =============================================================================

def find_best_ngram(X_train, y_train, X_test, y_test, ngram_sizes=[1, 2, 3, 4, 5, 6, 7]):
    """
    Find the best n-gram size using 100% of training data.

    Returns:
    --------
    tuple: (best_n, results_dict)
    """
    print(f"\n{'='*70}")
    print("Finding Best N-gram Size (100% training data)")
    print(f"{'='*70}")
    print(f"{'N':>3} | {'AUC':>8} | {'Best F1':>8} | {'Unique N-grams':>15}")
    print("-"*45)

    best_auc = 0
    best_n = 2
    results = {}

    for n in ngram_sizes:
        # Calculate P(label=1 | n-gram) from full training data
        ngram_stats = defaultdict(lambda: [0, 0])

        for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
            ngrams = extract_ngrams(seq, n)
            for ng in ngrams:
                ngram_stats[ng][int(label)] += 1

        ngram_probs = {}
        for ng, counts in ngram_stats.items():
            total = sum(counts)
            ngram_probs[ng] = counts[1] / total if total > 0 else 0.5

        # Predict on test set
        test_preds_prob = []
        for seq in X_test['Sequences']:
            ngrams = extract_ngrams(seq, n)
            probs = [ngram_probs.get(ng, 0.5) for ng in ngrams]
            avg_prob = np.mean(probs) if probs else 0.5
            test_preds_prob.append(avg_prob)

        # Calculate AUC
        auc = roc_auc_score(y_test['Outcome'], test_preds_prob)

        # Find best F1
        thresholds = np.linspace(0, 1, 101)
        f1_scores = [f1_score(y_test['Outcome'], [1 if p >= t else 0 for p in test_preds_prob], zero_division=0)
                     for t in thresholds]
        best_f1 = max(f1_scores)

        results[n] = {'auc': auc, 'best_f1': best_f1, 'unique_ngrams': len(ngram_stats)}

        print(f"{n:>3} | {auc:>8.4f} | {best_f1:>8.4f} | {len(ngram_stats):>15,}")

        if auc > best_auc:
            best_auc = auc
            best_n = n

    print(f"\n>>> Best N-gram: {best_n} (AUC: {best_auc:.4f})")
    return best_n, results


def ngram_fraction_experiment(X_train, y_train, X_test, y_test,
                               fractions=[0.01, 0.10, 0.30, 0.50, 0.75, 1.00],
                               ngram_sizes=[1, 2, 3, 4, 5, 6, 7],
                               random_state=42):
    """
    Test n-gram baseline classifier across different training data fractions.
    First finds the best n-gram on 100% data, then tests that n-gram on all fractions.

    Parameters:
    -----------
    X_train, y_train : pd.DataFrame
        Training data
    X_test, y_test : pd.DataFrame
        Test data (kept fixed)
    fractions : list
        Fractions of training data to test
    ngram_sizes : list
        N-gram sizes to consider for finding the best
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    tuple: (best_n, fraction_results_df, all_ngram_results)
    """
    from sklearn.model_selection import train_test_split

    # Step 1: Find best n-gram on 100% data
    best_n, all_ngram_results = find_best_ngram(X_train, y_train, X_test, y_test, ngram_sizes)

    # Step 2: Test best n-gram on different fractions
    print(f"\n{'='*90}")
    print(f"Fraction Experiment with Best N-gram (n={best_n})")
    print(f"{'='*90}")
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"{'='*90}")
    print(f"{'Fraction':>10} | {'Train Size':>12} | {'AUC':>8} | {'Best F1':>8} | {'Precision':>10} | {'Recall':>8} | {'Best Thresh':>11} | {'Coverage':>10}")
    print("-"*95)

    fraction_results = []

    for fraction in fractions:
        # Stratified subsample of training data
        if fraction < 1.0:
            _, X_train_sub, _, y_train_sub = train_test_split(
                X_train, y_train,
                test_size=fraction,
                stratify=y_train['Outcome'],
                random_state=random_state
            )
            X_train_sub = X_train_sub.reset_index(drop=True)
            y_train_sub = y_train_sub.reset_index(drop=True)
        else:
            X_train_sub = X_train.copy()
            y_train_sub = y_train.copy()

        # Calculate P(label=1 | n-gram) from subsampled training data
        ngram_stats = defaultdict(lambda: [0, 0])

        for seq, label in zip(X_train_sub['Sequences'], y_train_sub['Outcome']):
            ngrams = extract_ngrams(seq, best_n)
            for ng in ngrams:
                ngram_stats[ng][int(label)] += 1

        ngram_probs = {}
        for ng, counts in ngram_stats.items():
            total = sum(counts)
            ngram_probs[ng] = counts[1] / total if total > 0 else 0.5

        # Calculate coverage (% of test n-grams seen in training)
        train_ngrams = set(ngram_stats.keys())
        test_ngrams = set()
        for seq in X_test['Sequences']:
            test_ngrams.update(extract_ngrams(seq, best_n))
        coverage = len(test_ngrams & train_ngrams) / len(test_ngrams) * 100 if test_ngrams else 0

        # Predict on test set
        test_preds_prob = []
        for seq in X_test['Sequences']:
            ngrams = extract_ngrams(seq, best_n)
            probs = [ngram_probs.get(ng, 0.5) for ng in ngrams]
            avg_prob = np.mean(probs) if probs else 0.5
            test_preds_prob.append(avg_prob)

        # Calculate metrics
        auc = roc_auc_score(y_test['Outcome'], test_preds_prob)

        # Find best threshold for F1
        thresholds = np.linspace(0, 1, 101)
        f1_scores = [f1_score(y_test['Outcome'], [1 if p >= t else 0 for p in test_preds_prob], zero_division=0)
                     for t in thresholds]
        best_f1 = max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)]

        # Use best threshold for precision and recall (not 0.5, which often gives all zeros)
        test_preds_binary = [1 if p >= best_threshold else 0 for p in test_preds_prob]
        precision = precision_score(y_test['Outcome'], test_preds_binary, zero_division=0)
        recall = recall_score(y_test['Outcome'], test_preds_binary, zero_division=0)

        # Store results
        result = {
            'ngram_size': best_n,
            'fraction': fraction,
            'train_samples': len(X_train_sub),
            'test_samples': len(X_test),
            'unique_ngrams': len(ngram_stats),
            'coverage': coverage,
            'auc': auc,
            'best_f1': best_f1,
            'precision': precision,
            'recall': recall,
            'best_threshold': best_threshold
        }
        fraction_results.append(result)

        print(f"{fraction*100:>9.0f}% | {len(X_train_sub):>12,} | {auc:>8.4f} | {best_f1:>8.4f} | {precision:>10.4f} | {recall:>8.4f} | {best_threshold:>11.2f} | {coverage:>9.1f}%")

    # Convert to DataFrame
    results_df = pd.DataFrame(fraction_results)

    # Print summary
    print(f"\n{'='*90}")
    print(f"SUMMARY: Best {best_n}-gram performance by fraction")
    print(f"{'='*90}")
    print(f"\nAUC range: {results_df['auc'].min():.4f} - {results_df['auc'].max():.4f}")
    print(f"Best F1 range: {results_df['best_f1'].min():.4f} - {results_df['best_f1'].max():.4f}")
    print(f"Coverage range: {results_df['coverage'].min():.1f}% - {results_df['coverage'].max():.1f}%")

    return best_n, results_df, all_ngram_results


# Run the fraction experiment
best_n, fraction_results, all_ngram_results = ngram_fraction_experiment(
    X_train, y_train, X_test, y_test,
    fractions=[0.01, 0.10, 0.30, 0.50, 0.75, 1.00],
    ngram_sizes=[1, 2, 3, 4, 5, 6, 7]
)
