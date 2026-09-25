import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


# =============================================================================
# Configuration
# =============================================================================
# Path to the holes simulation data
DATA_PATH = "data/simulation/holes"
SEP = "\x1f"  # Separator used in sequences

# =============================================================================
# Load Data
# =============================================================================
print("Loading data...")
df = pd.read_csv(f"{DATA_PATH}/sequences_with_holes.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

print(f"Total sequences: {len(df)}")
print(f"Ordered sequences: {df['is_ordered'].sum()} ({df['is_ordered'].mean()*100:.1f}%)")
print(f"Unordered sequences: {len(df) - df['is_ordered'].sum()} ({(1-df['is_ordered'].mean())*100:.1f}%)")

# Split into train/val/test (80/10/10)
X = df['original_seq'].values
y = df['is_ordered'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.80, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.50, random_state=42, stratify=y_temp)

# Convert to DataFrames for consistency
X_train = pd.DataFrame({'Sequences': X_train})
X_val = pd.DataFrame({'Sequences': X_val})
X_test = pd.DataFrame({'Sequences': X_test})
y_train = pd.DataFrame({'Outcome': y_train})
y_val = pd.DataFrame({'Outcome': y_val})
y_test = pd.DataFrame({'Outcome': y_test})

print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# =============================================================================
# Single Position Analysis (Bigram Elements)
# =============================================================================
print("\n" + "="*80)
print("SINGLE POSITION ANALYSIS (Bigram Elements)")
print("="*80)

# Get sequence length
seq_length = len(X_train['Sequences'].iloc[0].split(SEP))
print(f"Sequence length: {seq_length} bigrams")

# Analyze each position
print("\nAUC by position (single bigram element):")
print("-"*50)

position_results = []
for pos in range(seq_length):
    # Extract element at position pos
    train_elements = [seq.split(SEP)[pos] for seq in X_train['Sequences']]

    # Calculate P(label=1 | element) from training
    element_stats = defaultdict(lambda: [0, 0])
    for elem, label in zip(train_elements, y_train['Outcome']):
        element_stats[elem][int(label)] += 1

    # Predict on test
    test_elements = [seq.split(SEP)[pos] for seq in X_test['Sequences']]
    test_preds = []
    for elem in test_elements:
        counts = element_stats[elem]
        prob = counts[1] / sum(counts) if sum(counts) > 0 else 0.5
        test_preds.append(prob)

    pos_auc = roc_auc_score(y_test['Outcome'], test_preds)
    position_results.append({'position': pos + 1, 'auc': pos_auc})
    print(f"Position {pos+1:2d} AUC: {pos_auc:.4f}")

# =============================================================================
# Cumulative Position Analysis
# =============================================================================
print("\n" + "="*80)
print("CUMULATIVE POSITION ANALYSIS")
print("="*80)


def cumulative_position_analysis(X_train, y_train, X_val, y_val, max_k=None):
    """Test cumulative information: first k bigrams"""

    seq_length = len(X_train['Sequences'].iloc[0].split(SEP))
    if max_k is None:
        max_k = seq_length

    results = []

    for k in range(1, max_k + 1):
        # Extract first k elements from each sequence
        train_prefixes = [SEP.join(seq.split(SEP)[:k]) for seq in X_train['Sequences']]
        val_prefixes = [SEP.join(seq.split(SEP)[:k]) for seq in X_val['Sequences']]

        # Calculate P(label=1 | prefix) from training
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

        # Calculate coverage: how many unique prefixes seen in train
        unique_train = len(set(train_prefixes))
        unique_val = len(set(val_prefixes))
        coverage = len(set(val_prefixes) & set(train_prefixes)) / unique_val * 100 if unique_val > 0 else 0

        print(f"First {k:2d} bigrams | AUC: {auc:.4f} | "
              f"Unique train: {unique_train:,} | "
              f"Coverage: {coverage:.1f}%")

        results.append({
            'k': k,
            'auc': auc,
            'unique_train': unique_train,
            'coverage': coverage
        })

    return results


cumulative_results = cumulative_position_analysis(X_train, y_train, X_test, y_test, max_k=min(20, seq_length))

# =============================================================================
# Element Pair Analysis (Pairs of consecutive bigrams)
# =============================================================================
print("\n" + "="*80)
print("ELEMENT PAIR ANALYSIS (Pairs of Consecutive Bigrams)")
print("="*80)


def extract_element_pairs(seq, sep=SEP):
    """Extract pairs of consecutive bigram elements"""
    elements = seq.split(sep)
    return [f"{elements[i]}_{elements[i+1]}" for i in range(len(elements)-1)]


def element_pair_distribution_analysis(X_train, y_train):
    """Analyze if ordered/unordered sequences have different pair distributions"""

    # Count pairs per class
    pairs_class_0 = Counter()
    pairs_class_1 = Counter()
    pairs_tot = Counter()

    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        pairs = extract_element_pairs(seq)
        if label == 0:
            pairs_class_0.update(pairs)
        else:
            pairs_class_1.update(pairs)
        pairs_tot.update(pairs)

    # Normalize to probabilities
    total_0 = sum(pairs_class_0.values())
    total_1 = sum(pairs_class_1.values())
    total_tot = sum(pairs_tot.values())

    prob_0 = {pair: count/total_0 for pair, count in pairs_class_0.items()}
    prob_1 = {pair: count/total_1 for pair, count in pairs_class_1.items()}
    prob_tot = {pair: count/total_tot for pair, count in pairs_tot.items()}

    # Find most discriminative pairs
    all_pairs = set(prob_0.keys()) | set(prob_1.keys())
    pair_scores = []

    for pair in all_pairs:
        p0 = prob_0.get(pair, 0)
        p1 = prob_1.get(pair, 0)
        ptot = prob_tot.get(pair, 0)
        diff = abs(p1 - p0)
        pair_scores.append((pair, ptot, p0, p1, diff))

    # Sort by difference
    pair_scores.sort(key=lambda x: x[4], reverse=True)

    print("\nTop 20 most discriminative element pairs:")
    print(f"{'Pair':<15} | {'P(pair)':<10} | {'P(pair|0)':<10} | {'P(pair|1)':<10} | {'Diff':<10}")
    print("-"*70)
    for pair, ptot, p0, p1, diff in pair_scores[:20]:
        print(f"{pair:<15} | {ptot:10.6f} | {p0:10.6f} | {p1:10.6f} | {diff:10.6f}")

    # Sort by frequency
    pair_scores.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 20 most frequent element pairs:")
    print(f"{'Pair':<15} | {'P(pair)':<10} | {'P(pair|0)':<10} | {'P(pair|1)':<10} | {'Diff':<10}")
    print("-"*70)
    for pair, ptot, p0, p1, diff in pair_scores[:20]:
        print(f"{pair:<15} | {ptot:10.6f} | {p0:10.6f} | {p1:10.6f} | {diff:10.6f}")

    return pair_scores


pair_scores = element_pair_distribution_analysis(X_train, y_train)

# =============================================================================
# Element Pair Baseline Classifier
# =============================================================================
print("\n" + "="*80)
print("ELEMENT PAIR BASELINE CLASSIFIER")
print("="*80)


def element_pair_baseline_classifier(X_train, y_train, X_val, y_val):
    """Classify based on element pair distribution"""

    # Calculate P(label=1 | pair) for each pair
    pair_stats = defaultdict(lambda: [0, 0])

    for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
        pairs = extract_element_pairs(seq)
        for pair in pairs:
            pair_stats[pair][int(label)] += 1

    # Convert to probabilities
    pair_probs = {}
    for pair, counts in pair_stats.items():
        total = sum(counts)
        pair_probs[pair] = counts[1] / total if total > 0 else 0.5

    # Predict: use mean probability of pairs in sequence
    val_preds = []
    for seq in X_val['Sequences']:
        pairs = extract_element_pairs(seq)
        probs = [pair_probs.get(pair, 0.5) for pair in pairs]
        avg_prob = np.mean(probs) if probs else 0.5
        val_preds.append(avg_prob)

    auc = roc_auc_score(y_val['Outcome'], val_preds)

    # Binarize with threshold 0.5 for F1
    val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds]
    f1 = f1_score(y_val['Outcome'], val_preds_binary)

    # Find optimal threshold
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in val_preds]
        f1_scores.append(f1_score(y_val['Outcome'], preds))

    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    print(f"Element Pair Baseline AUC: {auc:.4f}")
    print(f"Element Pair Baseline F1 (threshold=0.5): {f1:.4f}")
    print(f"Element Pair Baseline F1 (best threshold={best_threshold:.2f}): {best_f1:.4f}")

    return auc, f1, best_f1


pair_auc, pair_f1, pair_best_f1 = element_pair_baseline_classifier(X_train, y_train, X_test, y_test)

# =============================================================================
# N-gram Analysis (N consecutive bigram elements)
# =============================================================================
print("\n" + "="*80)
print("N-GRAM ANALYSIS (N Consecutive Bigram Elements)")
print("="*80)


def extract_ngrams(seq, n, sep=SEP):
    """Extract all n-grams (n consecutive elements) from a sequence"""
    elements = seq.split(sep)
    if len(elements) < n:
        return []
    return ["_".join(elements[i:i+n]) for i in range(len(elements)-n+1)]


def ngram_baseline_classifier_with_f1(X_train, y_train, X_val, y_val, n=2):
    """
    Classify based on n-gram distribution - with F1

    Parameters:
    -----------
    n : int
        Size of n-gram (2=pairs, 3=triplets, etc.)
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

    # Binarize with threshold 0.5 for F1
    val_preds_binary = [1 if p >= 0.5 else 0 for p in val_preds_prob]
    f1 = f1_score(y_val['Outcome'], val_preds_binary)

    # Find optimal threshold
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    for thresh in thresholds:
        preds = [1 if p >= thresh else 0 for p in val_preds_prob]
        f1_scores.append(f1_score(y_val['Outcome'], preds))

    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    ngram_name = {1: '1-gram (unigram)', 2: '2-gram (pairs)', 3: '3-gram (triplets)',
                  4: '4-gram', 5: '5-gram', 6: '6-gram', 7: '7-gram'}
    name = ngram_name.get(n, f'{n}-gram')

    print(f"{name} Baseline AUC: {auc:.4f}")
    print(f"{name} Baseline F1 (threshold=0.5): {f1:.4f}")
    print(f"{name} Baseline F1 (best threshold={best_threshold:.2f}): {best_f1:.4f}")

    return auc, f1, best_f1, best_threshold


# Run for all n-gram sizes
print("\nN-gram Baseline Classifier Results")
print("-"*70)

ngram_results = {}
for n in range(1, 8):  # 1 to 7 (includes 1-gram/unigram)
    print(f"\n{n}-gram results:")
    auc, f1, best_f1, best_thresh = ngram_baseline_classifier_with_f1(
        X_train, y_train, X_test, y_test, n=n
    )
    ngram_results[n] = {
        'auc': auc,
        'f1': f1,
        'best_f1': best_f1,
        'best_threshold': best_thresh
    }

# =============================================================================
# Ordered Chain Analysis
# =============================================================================
print("\n" + "="*80)
print("ORDERED CHAIN ANALYSIS")
print("="*80)

# Check if ordered_chain column exists
if 'ordered_chain' in df.columns:
    # Analyze the ordered chains
    ordered_df = df[df['is_ordered'] == 1]

    # Get chain lengths
    chain_lengths = ordered_df['ordered_chain'].apply(
        lambda x: len(x.split(',')) if x else 0
    )

    print(f"\nOrdered chain statistics:")
    print(f"  Mean length: {chain_lengths.mean():.2f}")
    print(f"  Min length: {chain_lengths.min()}")
    print(f"  Max length: {chain_lengths.max()}")
    print(f"  Std: {chain_lengths.std():.2f}")

    # Distribution of chain lengths
    print(f"\nChain length distribution:")
    for length, count in sorted(chain_lengths.value_counts().items()):
        pct = count / len(chain_lengths) * 100
        print(f"  Length {length}: {count} ({pct:.1f}%)")

    # Most common chain elements
    all_chain_elements = []
    for chain in ordered_df['ordered_chain']:
        if chain:
            all_chain_elements.extend(chain.split(','))

    element_counts = Counter(all_chain_elements)
    print(f"\nTop 20 most common elements in ordered chains:")
    for elem, count in element_counts.most_common(20):
        print(f"  {elem}: {count}")
else:
    print("No 'ordered_chain' column found in data.")

# =============================================================================
# Visualization
# =============================================================================
print("\n" + "="*80)
print("GENERATING PLOTS")
print("="*80)


def plot_ngram_vignette(X_train, y_train, max_n=7, top_n=100, min_n=1):
    """
    Create a vignette plot showing n-gram distributions.
    Each subplot shows the top_n most discriminative n-grams.
    """
    n_grams = list(range(min_n, max_n + 1))
    n_plots = len(n_grams)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, n in enumerate(n_grams):
        ax = axes[idx]

        # Count n-grams per class
        ngrams_0 = Counter()
        ngrams_1 = Counter()

        for seq, label in zip(X_train['Sequences'], y_train['Outcome']):
            ngrams = extract_ngrams(seq, n)
            if label == 0:
                ngrams_0.update(ngrams)
            else:
                ngrams_1.update(ngrams)

        # Find most discriminative n-grams
        all_ngrams = set(ngrams_0.keys()) | set(ngrams_1.keys())
        ngram_diff = {}

        for ng in all_ngrams:
            freq_0 = ngrams_0.get(ng, 0)
            freq_1 = ngrams_1.get(ng, 0)
            diff = abs(freq_0 - freq_1)
            ngram_diff[ng] = (freq_0, freq_1, diff)

        # Sort and take top N
        top_ngrams = sorted(ngram_diff.items(), key=lambda x: x[1][2], reverse=True)[:top_n]

        # Prepare data
        counts_0 = [data[0] for _, data in top_ngrams]
        counts_1 = [data[1] for _, data in top_ngrams]

        # Plot
        x = np.arange(len(top_ngrams))
        width = 0.35

        ax.bar(x - width/2, counts_1, width, label='Ordered (1)', alpha=0.8, color='C0')
        ax.bar(x + width/2, counts_0, width, label='Unordered (0)', alpha=0.8, color='C1')

        ngram_name = {1: '1-grams (unigrams)', 2: '2-grams', 3: '3-grams', 4: '4-grams',
                      5: '5-grams', 6: '6-grams', 7: '7-grams'}
        ax.set_title(f'Top {len(top_ngrams)} {ngram_name.get(n, f"{n}-grams")}',
                     fontsize=12, fontweight='bold')

        ax.set_xlabel(f'{ngram_name.get(n, f"{n}-grams")}', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_xticks([])

        if idx == 0:
            ax.legend(fontsize=10)

        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'N-gram Distribution Analysis (Top {top_n} Most Discriminative)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{DATA_PATH}/ngram_vignette.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {DATA_PATH}/ngram_vignette.png")
    plt.close()


def plot_position_auc(position_results):
    """Plot AUC by position"""
    positions = [r['position'] for r in position_results]
    aucs = [r['auc'] for r in position_results]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(positions, aucs, color='steelblue', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('AUC by Single Bigram Position', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DATA_PATH}/position_auc.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {DATA_PATH}/position_auc.png")
    plt.close()


def plot_cumulative_auc(cumulative_results):
    """Plot cumulative AUC and coverage"""
    k_vals = [r['k'] for r in cumulative_results]
    aucs = [r['auc'] for r in cumulative_results]
    coverage = [r['coverage'] for r in cumulative_results]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = 'steelblue'
    ax1.set_xlabel('Number of bigrams (k)', fontsize=12)
    ax1.set_ylabel('AUC', color=color1, fontsize=12)
    ax1.plot(k_vals, aucs, 'o-', color=color1, linewidth=2, markersize=6, label='AUC')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')

    ax2 = ax1.twinx()
    color2 = 'darkorange'
    ax2.set_ylabel('Coverage (%)', color=color2, fontsize=12)
    ax2.plot(k_vals, coverage, 's--', color=color2, linewidth=2, markersize=6, label='Coverage')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('Cumulative Position Analysis: AUC and Coverage', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    plt.savefig(f'{DATA_PATH}/cumulative_auc.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {DATA_PATH}/cumulative_auc.png")
    plt.close()


def plot_ngram_comparison(ngram_results):
    """Plot comparison of n-gram baseline classifiers"""
    n_vals = list(ngram_results.keys())
    aucs = [ngram_results[n]['auc'] for n in n_vals]
    f1s = [ngram_results[n]['f1'] for n in n_vals]
    best_f1s = [ngram_results[n]['best_f1'] for n in n_vals]

    x = np.arange(len(n_vals))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, aucs, width, label='AUC', alpha=0.8)
    bars2 = ax.bar(x, f1s, width, label='F1 (thresh=0.5)', alpha=0.8)
    bars3 = ax.bar(x + width, best_f1s, width, label='F1 (best thresh)', alpha=0.8)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')

    ax.set_xlabel('N-gram size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('N-gram Baseline Classifier Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}-gram' for n in n_vals])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{DATA_PATH}/ngram_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {DATA_PATH}/ngram_comparison.png")
    plt.close()


# Generate plots
plot_position_auc(position_results)
plot_cumulative_auc(cumulative_results)
plot_ngram_comparison(ngram_results)
plot_ngram_vignette(X_train, y_train, max_n=6, top_n=100)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nDataset: {len(df)} sequences")
print(f"  - Ordered: {df['is_ordered'].sum()} ({df['is_ordered'].mean()*100:.1f}%)")
print(f"  - Unordered: {len(df) - df['is_ordered'].sum()} ({(1-df['is_ordered'].mean())*100:.1f}%)")
print(f"  - Sequence length: {seq_length} bigram elements")

print(f"\nBest single position AUC: {max(r['auc'] for r in position_results):.4f} "
      f"(position {max(position_results, key=lambda x: x['auc'])['position']})")

print(f"\nBest cumulative AUC: {max(r['auc'] for r in cumulative_results):.4f} "
      f"(first {max(cumulative_results, key=lambda x: x['auc'])['k']} bigrams)")

best_ngram = max(ngram_results.items(), key=lambda x: x[1]['auc'])
print(f"\nBest n-gram baseline: {best_ngram[0]}-gram with AUC={best_ngram[1]['auc']:.4f}")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

# =============================================================================
# Generate LaTeX Tables
# =============================================================================

def generate_latex_tables(
    df,
    seq_length,
    position_results,
    cumulative_results,
    ngram_results,
    pair_auc,
    chain_lengths=None,
    lags=None,
    key_elements=None,
    save_to_file=True,
    output_path=None
):
    """Generate LaTeX tables from analysis results."""

    if output_path is None:
        output_path = f"{DATA_PATH}/latex_tables.tex"

    latex = []
    latex.append("% " + "="*70)
    latex.append("% AUTO-GENERATED LATEX TABLES FROM stat_test_holes.py")
    latex.append("% " + "="*70)
    latex.append("")

    # Table 1: Dataset Configuration
    latex.append("% Table 1: Dataset Configuration")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Synthetic Dataset Configuration}")
    latex.append("\\label{tab:dataset_config}")
    latex.append("\\begin{tabular}{ll}")
    latex.append("\\toprule")
    latex.append("\\textbf{Parameter} & \\textbf{Value} \\\\")
    latex.append("\\midrule")
    latex.append(f"Total sequences & {len(df):,} \\\\")
    latex.append(f"Sequence length & {seq_length} bigrams \\\\")
    latex.append("Alphabet size & 676 (AA--ZZ) \\\\")
    if key_elements:
        latex.append(f"Key elements & {key_elements} bigrams \\\\")
    if lags:
        latex.append(f"Lags & {lags} \\\\")
    n_ordered = int(df['is_ordered'].sum())
    n_unordered = len(df) - n_ordered
    pct_ordered = df['is_ordered'].mean() * 100
    latex.append(f"Ordered sequences & {n_ordered:,} ({pct_ordered:.0f}\\%) \\\\")
    latex.append(f"Unordered sequences & {n_unordered:,} ({100-pct_ordered:.0f}\\%) \\\\")
    latex.append("\\midrule")
    latex.append(f"Train / Val / Test & {len(X_train):,} / {len(X_val):,} / {len(X_test):,} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 2: Top 10 Single Position AUC
    latex.append("% Table 2: Single Position AUC Analysis")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Single Position AUC Analysis (Top 10 Positions)}")
    latex.append("\\label{tab:position_auc}")
    latex.append("\\begin{tabular}{cc|cc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Position} & \\textbf{AUC} & \\textbf{Position} & \\textbf{AUC} \\\\")
    latex.append("\\midrule")

    # Sort by AUC and get top 10
    sorted_positions = sorted(position_results, key=lambda x: x['auc'], reverse=True)[:10]
    for i in range(5):
        p1 = sorted_positions[i]
        p2 = sorted_positions[i + 5]
        latex.append(f"{p1['position']} & {p1['auc']:.4f} & {p2['position']} & {p2['auc']:.4f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 3: Cumulative Position Analysis
    latex.append("% Table 3: Cumulative Position Analysis")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Cumulative Position Analysis}")
    latex.append("\\label{tab:cumulative_auc}")
    latex.append("\\begin{tabular}{cccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{First $k$ Bigrams} & \\textbf{AUC} & \\textbf{Unique Prefixes} & \\textbf{Coverage (\\%)} \\\\")
    latex.append("\\midrule")

    # Show first 5 rows, then summarize if more
    for r in cumulative_results[:5]:
        latex.append(f"{r['k']} & {r['auc']:.4f} & {r['unique_train']:,} & {r['coverage']:.1f} \\\\")
    if len(cumulative_results) > 5:
        last = cumulative_results[-1]
        latex.append(f"{last['k']}+ & {last['auc']:.4f} & {last['unique_train']:,} & {last['coverage']:.1f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 4: N-gram Baseline Classifier
    latex.append("% Table 4: N-gram Baseline Classifier")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{N-gram Baseline Classifier Performance}")
    latex.append("\\label{tab:ngram_baseline}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{N-gram} & \\textbf{AUC} & \\textbf{F1 ($\\tau=0.5$)} & \\textbf{F1 (Best $\\tau$)} \\\\")
    latex.append("\\midrule")

    ngram_names = {1: '1-gram (unigram)', 2: '2-gram (pairs)', 3: '3-gram (triplets)',
                   4: '4-gram', 5: '5-gram', 6: '6-gram', 7: '7-gram'}
    best_auc = max(r['auc'] for r in ngram_results.values())

    for n, r in ngram_results.items():
        name = ngram_names.get(n, f'{n}-gram')
        bold = "\\textbf{" if r['auc'] == best_auc else ""
        bold_end = "}" if r['auc'] == best_auc else ""
        latex.append(f"{name} & {bold}{r['auc']:.4f}{bold_end} & {r['f1']:.4f} & {r['best_f1']:.4f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")

    # Table 5: Chain Length Distribution (if available)
    if chain_lengths is not None and len(chain_lengths) > 0:
        latex.append("% Table 5: Ordered Chain Statistics")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Ordered Chain Length Distribution}")
        latex.append("\\label{tab:chain_stats}")
        latex.append("\\begin{tabular}{ccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Chain Length} & \\textbf{Count} & \\textbf{Percentage} \\\\")
        latex.append("\\midrule")

        for length, count in sorted(chain_lengths.value_counts().items()):
            pct = count / len(chain_lengths) * 100
            latex.append(f"{length} & {count:,} & {pct:.1f}\\% \\\\")

        latex.append("\\midrule")
        latex.append(f"\\textbf{{Mean}} & \\multicolumn{{2}}{{c}}{{{chain_lengths.mean():.2f} $\\pm$ {chain_lengths.std():.2f}}} \\\\")
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        latex.append("")

    # Table 6: Summary
    latex.append("% Table 6: Summary")
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Summary of Statistical Baseline Analysis}")
    latex.append("\\label{tab:summary}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Method} & \\textbf{Best AUC} & \\textbf{Notes} \\\\")
    latex.append("\\midrule")

    best_pos = max(position_results, key=lambda x: x['auc'])
    best_cum = max(cumulative_results, key=lambda x: x['auc'])
    best_ng = max(ngram_results.items(), key=lambda x: x[1]['auc'])

    latex.append(f"Single position & {best_pos['auc']:.4f} & Position {best_pos['position']} \\\\")
    latex.append(f"Cumulative positions & {best_cum['auc']:.4f} & First {best_cum['k']} bigram(s) \\\\")
    latex.append(f"Element pairs & {pair_auc:.4f} & 2-gram baseline \\\\")
    latex.append(f"Best n-gram & {best_ng[1]['auc']:.4f} & {best_ng[0]}-gram \\\\")
    latex.append("\\midrule")
    latex.append("Random baseline & 0.5000 & --- \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_str = "\n".join(latex)

    # Print to console
    print("\n" + "="*80)
    print("LATEX TABLES")
    print("="*80)
    print(latex_str)

    # Save to file
    if save_to_file:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"\nLaTeX tables saved to: {output_path}")

    return latex_str


# Get chain lengths for the table
chain_lengths = None
if 'ordered_chain' in df.columns:
    ordered_df = df[df['is_ordered'] == 1]
    chain_lengths = ordered_df['ordered_chain'].apply(
        lambda x: len(x.split(',')) if x else 0
    )

# Generate LaTeX tables
generate_latex_tables(
    df=df,
    seq_length=seq_length,
    position_results=position_results,
    cumulative_results=cumulative_results,
    ngram_results=ngram_results,
    pair_auc=pair_auc,
    chain_lengths=chain_lengths,
    lags="[4, 3, 2]",  # Update this based on your configuration
    key_elements=200,   # Update this based on your configuration
    save_to_file=True
)
