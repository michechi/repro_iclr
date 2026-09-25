import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

csv_number = 'odd_3'
X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

df = pd.DataFrame({
    "Sequences":pd.concat([X_train.Sequences, X_val.Sequences, X_test.Sequences]),
    "Outcome":pd.concat([y_train.Outcome, y_val.Outcome, y_test.Outcome])
})
df['Outcome'].value_counts()/len(df)


def plot_ngram_vignette(X_train, y_train, max_n=7, top_n=700):
    """
    Create a vignette (faceted) plot showing n-gram distributions from 2-grams to max_n-grams.
    Each subplot shows the top_n most discriminative n-grams.
    """
    
    def extract_ngrams(seq, n):
        """Extract n-grams from a sequence"""
        letters = seq.split('\x1f')
        if len(letters) < n:
            return []
        return ["-".join(letters[i:i+n]) for i in range(len(letters)-n+1)]
    
    # Prepare subplots
    n_grams = list(range(2, max_n + 1))
    n_plots = len(n_grams)
    n_cols = 2  # 2 columns
    n_rows = int(np.ceil(n_plots / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    for idx, n in enumerate(n_grams):
        ax = axes[idx]
        
        # Count n-grams per class
        ngrams_0 = Counter()
        ngrams_1 = Counter()
        
        for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome']):
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
        ngram_labels = [ng for ng, _ in top_ngrams]
        counts_0 = [data[0] for _, data in top_ngrams]
        counts_1 = [data[1] for _, data in top_ngrams]
        
        # Plot
        x = np.arange(len(ngram_labels))
        width = 0.35
        
        ax.bar(x - width/2, counts_1, width, label='Label=1', alpha=0.8, color='C0')
        ax.bar(x + width/2, counts_0, width, label='Label=0', alpha=0.8, color='C1')
        
        ngram_name = {2: 'Bigrams', 3: 'Trigrams', 4: '4-grams', 5: '5-grams', 6: '6-grams', 7: '7-grams'}
        ax.set_title(f'Top {len(ngram_labels)} {ngram_name.get(n, f"{n}-grams")}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel(f'{ngram_name.get(n, f"{n}-grams")}', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        # Don't show x-tick labels for top_n=100 (too crowded)
        ax.set_xticks([])
        
        if idx == 0:  # Legend only on first plot
            ax.legend(fontsize=10)
        
        ax.grid(axis='y', alpha=0.3)
        
        # Print top 5 for each n-gram
        print(f"\nTop 10 most discriminative {ngram_name.get(n, f'{n}-grams')}:")
        print(f"{f'{n}-gram':<30} {'Count(0)':<12} {'Count(1)':<12} {'Diff':<10}")
        print("-" * 70)
        for ng, (c0, c1, diff) in top_ngrams[:10]:
            print(f"{ng:<30} {c0:<12} {c1:<12} {diff:<10.0f}")
    
    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'N-gram Distribution Analysis (Top {top_n} Most Discriminative)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('ngram_vignette.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
plot_ngram_vignette(X_train, y_train, max_n=7, top_n=100)

def plot_bigram_distribution(X_train, y_train, from_n=0, top_n=30):
    """
    Create a plot showing the most discriminative bigrams
    between the two classes
    """

    # Function to extract bigrams
    def extract_bigrams(seq):
        letters = seq.split('\x1f')
        return [f"{letters[i]}-{letters[i+1]}" for i in range(len(letters)-1)]

    # Count bigrams per class
    bigrams_0 = Counter()
    bigrams_1 = Counter()

    for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome']):
        bigrams = extract_bigrams(seq)
        if label == 0:
            bigrams_0.update(bigrams)
        else:
            bigrams_1.update(bigrams)

    # Compute normalized frequencies
    total_0 = sum(bigrams_0.values())
    total_1 = sum(bigrams_1.values())

    # Find the most discriminative bigrams
    all_bigrams = set(bigrams_0.keys()) | set(bigrams_1.keys())
    bigram_diff = {}

    for bg in all_bigrams:
        freq_0 = bigrams_0.get(bg, 0)
        freq_1 = bigrams_1.get(bg, 0)
        diff = abs(freq_0 - freq_1)
        bigram_diff[bg] = (freq_0, freq_1, diff)

    # Sort by difference and take the top N
    top_bigrams = sorted(bigram_diff.items(), key=lambda x: x[1][2], reverse=True)[from_n:top_n]

    # Prepare data for the plot
    bigram_labels = [bg for bg, _ in top_bigrams]
    counts_0 = [data[0] for _, data in top_bigrams]
    counts_1 = [data[1] for _, data in top_bigrams]

    # Create the plot
    x = np.arange(len(bigram_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 8))
    bars1 = ax.bar(x - width/2, counts_1, width, label='Label=1 (Ordered)', alpha=0.8)
    bars0 = ax.bar(x + width/2, counts_0, width, label='Label=0 (Unordered)', alpha=0.8)

    ax.set_xlabel('Bigrams', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Top {top_n} Most Discriminative Bigrams by Label', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bigram_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('bigram_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print(f"\nTop {min(10, top_n)} most discriminative bigrams:")
    print(f"{'Bigram':<10} {'Count(Label=0)':<15} {'Count(Label=1)':<15} {'Difference':<12}")
    print("-" * 60)
    for bg, (c0, c1, diff) in top_bigrams[:10]:
        print(f"{bg:<10} {c0:<15} {c1:<15} {diff:<12.0f}")

# Run
plot_bigram_distribution(X_train, y_train, from_n=1, top_n=100)
