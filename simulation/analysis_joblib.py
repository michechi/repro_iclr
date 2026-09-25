import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections
import sys

from simulation.do_check_lag import check_lag
from pathlib import Path
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from collections import Counter
from scipy.stats import bernoulli
from itertools import product
from typing import List, Union
from tqdm import tqdm

obj = joblib.load("./simulation_results_deterministic.joblib")
check = obj['check']
df_2_monitor = obj['df_2_monitor']
sequences = obj['sequences']
letters = list(string.ascii_uppercase)

def assign_outcome_positional(
    seq:str, 
    c_ord:dict, 
    lags:Union[int, List], 
    rnd:bool=False, 
    sep:str="\x1f", 
    already_splitted=False, 
    pr_1 = 0.7, 
    debugging=True,
    min_chain_length=2,
    tolerance=True
    ) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    # np.random.seed(seed=123456)

    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = seq.split(sep) # avoiding noising letters
    else: 
        test_seq_splt = seq
    
    # Check if there are key letters separated by lags.
    are_lagged_keys = check_lag(test_seq_splt, l_keys, lags)

    if (are_lagged_keys and all(isinstance(item, list) for item in are_lagged_keys)):
        n_seq=len(are_lagged_keys)
        all_ordered = [True]*n_seq
        tol=tolerance # For the cyclic ordering
        for pos, subseq in enumerate(are_lagged_keys):
            if len(subseq) < min_chain_length:
                all_ordered[pos] = False
                continue
            # Check whether are ordered
            for x,y in zip(subseq[:-1], subseq[1:]):
                if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
                        if tol:
                            tol=False
                        else:
                            all_ordered[pos]=False
        if any(all_ordered):
            # If there is at least one true, then there is one ordered sequence=> high probabilities of 1
            pr_to_simulate = pr_1
        else:
            # If there is no true, then there are no one ordered sequence=> low probabilities of 1
            pr_to_simulate = 1-pr_1
    
    elif (are_lagged_keys and not all(isinstance(item, list) for item in are_lagged_keys)):
        all_ordered = True
        tol=tolerance # For the cyclic ordering
        
        if len(are_lagged_keys) < min_chain_length:
            all_ordered = False
        
        else:
            # Check whether are ordered
            for x,y in zip(are_lagged_keys[:-1], are_lagged_keys[1:]):
                if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
                        if tol:
                            tol=False
                        else:
                            all_ordered=False
        if all_ordered:
            # If there is at least one true, then there is one ordered sequence=> high probabilities of 1
            pr_to_simulate = pr_1
        else:
            # If there is no true, then there are no one ordered sequence=> low probabilities of 1
            pr_to_simulate = 1-pr_1
    
    else:
        # If there no keys, then there are no one ordered sequence=> low probabilities of 1
        all_ordered=False
        pr_to_simulate = 1-pr_1 # if test_seq_splt is void then there are no keys so not ordered
        
    if rnd:
        # Stochastics outcome
        outcome = bernoulli.rvs(pr_to_simulate)
    else:
        # Deterministic outcome
        outcome = int(np.where(pr_to_simulate==pr_1, 1, 0))
    
    # Returning more, to be able to inspect results
    results_2_debug = {
        'outcome':[outcome],
        'seq':seq,
        'pr_2_sim':[pr_to_simulate],
        'lagged_keys':[are_lagged_keys],
        'all_ordered':[all_ordered]
    }
    return(results_2_debug)


#-------------------------------
letters_4_key = ["W", "D", "Q", "J", "X", "N"]
letters_4_odds = ["B", "S", "X", "U"]
df_2_monitor_test = df_2_monitor.loc[df_2_monitor['outcome']==1, ]
def get_key_letters_parity(seq, key_letters, sep='\x1f', already_splitted=False):
    """
    Returns only the parity (even/odd) for each key letter.
    
    Returns:
    --------
    dict : {letter: 'even'/'odd'}
    """
    
    if not already_splitted:
        seq_list = seq.split(sep)
    else:
        seq_list = seq
    
    # 1 even, 0 odd
    dict_4_letters = {key: (1 if seq_list.count(key) % 2 == 0 else 0) for key in key_letters}
    return 1 if sum(list(dict_4_letters.values())) %2 == 0 else 0

df_2_monitor_test['Parity'] = df_2_monitor_test['seq'].apply(lambda x: get_key_letters_parity(x, letters_4_odds))

def extract_characters(seq:str, sep:str="\x1f") -> str:
    seq_char = "-".join([x[0] for x in seq.split(sep)])
    return(seq_char)


# GRAPHICAL STUFF
# check.keys()
# chr_seq_1s=list(map(extract_characters, check['valid_sequences']))
# chr_seq_0s=list(map(extract_characters, check['invalid_sequences']))

chr_seq_1s=list(map(extract_characters, df_2_monitor_test.loc[df_2_monitor_test['Parity']==1, 'seq']))
chr_seq_0s=list(map(extract_characters, df_2_monitor_test.loc[df_2_monitor_test['Parity']==0, 'seq']))

def plot_all_positions_compact(chr_seq_1s, chr_seq_0s, n_positions=40):
    """
    Compact version with smaller, denser plots.
    """
    letters_ord = string.ascii_uppercase
    
    n_cols = 8  # More columns for compact view
    n_rows = int(np.ceil(n_positions / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3*n_rows))
    axes = axes.flatten()
    
    for i in range(n_positions):
        ax = axes[i]
        
        stats_1s = collections.Counter(x[i*2] for x in chr_seq_1s if len(x) > i*2)
        stats_0s = collections.Counter(x[i*2] for x in chr_seq_0s if len(x) > i*2)
        
        counts_1s = [stats_1s.get(letter, 0) for letter in letters_ord]
        counts_0s = [stats_0s.get(letter, 0) for letter in letters_ord]
        
        x_pos = np.arange(len(letters_ord))
        width = 0.35
        
        ax.bar(x_pos - width/2, counts_1s, width, label='Label=1', alpha=0.8, color='C0')
        ax.bar(x_pos + width/2, counts_0s, width, label='Label=0', alpha=0.8, color='C1')
        
        ax.set_title(f'Pos {i+1}', fontsize=8)
        ax.set_xticks([])  # Remove x-ticks for compactness
        ax.tick_params(labelsize=6)
        
        # Add legend only once
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')
    
    # Hide unused subplots
    for j in range(n_positions, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Letter Frequency by Position and Label', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('all_positions_frequency_compact.png', dpi=300, bbox_inches='tight')
    plt.show()

# Usage
plot_all_positions_compact(chr_seq_1s, chr_seq_0s, n_positions=20)

# Create Dataset
df = pd.DataFrame({
    "Sequences":df_2_monitor_test['seq'],
    "Outcome":df_2_monitor_test['Parity']
})

df_1s = df.loc[df["Outcome"]==1, ]
df_0s = df.loc[df["Outcome"]==0, ]
df_full = pd.concat([df_1s, df_0s], ignore_index=True)

def sample_with_proportion(df, n, pi, label_col='Outcome', random_state=9550):
    """
    Sample n rows from df with specified proportion of positive class.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    n : int
        Total number of samples to return
    pi : float
        Proportion of positive class (label=1), between 0 and 1
    label_col : str
        Name of the label column
    random_state : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Sampled dataframe with desired proportion
    """
    # Calculate number of samples per class
    n_pos = int(n * pi)
    n_neg = n - n_pos
    
    # Separate by class
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0]
    
    # Check if we have enough samples
    if len(df_pos) < n_pos:
        raise ValueError(f"Not enough positive samples: need {n_pos}, have {len(df_pos)}")
    if len(df_neg) < n_neg:
        raise ValueError(f"Not enough negative samples: need {n_neg}, have {len(df_neg)}")
    
    # Sample from each class
    sampled_pos = df_pos.sample(n=n_pos, random_state=random_state)
    sampled_neg = df_neg.sample(n=n_neg, random_state=random_state)
    
    # Combine and shuffle
    sampled_df = pd.concat([sampled_pos, sampled_neg], axis=0)
    sampled_df = sampled_df.sample(frac=1, random_state=random_state)  # shuffle
    
    return sampled_df.reset_index(drop=True)

# Ratio of Label:
# df_full.Outcome.value_counts()/len(df_full)
# # df_full = df_full.groupby('Outcome')[['Sequences', 'Outcome']].apply(lambda x: x.sample(frac=0.2))
# # df_full = pd.concat([df_1s.sample(n_1s), df_0s.sample(n_0s*2)])
# df_full = sample_with_proportion(df_full, 22439*3, 0.3)
df_full.Outcome.value_counts()/len(df_full)
len(df_full)

df_full["parity"] = df_full['Sequences'].apply(lambda x: get_key_letters_parity(x, letters_4_odds))
df_full['Outcome2'] = ((df_full['Outcome'] == 1) & (df_full['parity'] == 1)).astype(int)

df_full_class_0 = df_full[df_full['Outcome2'] == 1]
df_full_other   = df_full[df_full['Outcome2'] != 1]

# keep only a fraction (e.g. 40%)
df_class_1_down = df_full_class_0.sample(frac=0.9, random_state=42)

df_full = pd.concat([df_class_1_down, df_full_other])

# Splitting Train Val Test
# Now train_val_test split:
X, y = df_full["Sequences"], df_full["Outcome2"]
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.80, random_state=999)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.50, random_state=999)

# Uncomment just if you want to save data!
number_csv = "parity"
for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], [f"X_train_{number_csv}", f"X_val_{number_csv}", f"X_test_{number_csv}", f"y_train_{number_csv}", f"y_val_{number_csv}", f"y_test_{number_csv}"]):
    df.to_csv(f"data/simulation/{name}.csv", index=False)

# GRAPHICAL STUFF
# Let's visualize bigrams
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
    
    for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome2']):
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
plot_bigram_distribution(X_train, y_train, from_n=1, top_n=500)



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
        
        for seq, label in zip(pd.DataFrame(X_train)['Sequences'], pd.DataFrame(y_train)['Outcome2']):
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
        print(f"\nTop 5 most discriminative {ngram_name.get(n, f'{n}-grams')}:")
        print(f"{f'{n}-gram':<30} {'Count(0)':<12} {'Count(1)':<12} {'Diff':<10}")
        print("-" * 70)
        for ng, (c0, c1, diff) in top_ngrams[:5]:
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
plot_ngram_vignette(X_train, y_train, max_n=7, top_n=700)


# Now we want to create more ordered sequences:
c_vocab = {w:p for p,w in enumerate(letters_4_key,start=0)}

def generate_ordered_sequence_diverse(letters, key_letters, key_ordering, lag, n_events, 
                                     min_chain_length=3, seed=None):
    """
    Generate an ordered sequence with maximum diversity.
    
    Parameters:
    -----------
    letters : list
        All available letters (26)
    key_letters : list
        The key letters (6)
    key_ordering : dict
        {letter: rank} for the key letters
    lag : int
        Distance between consecutive letters in the chain (e.g. 7)
    n_events : int
        Length of the sequence (e.g. 20)
    min_chain_length : int
        Minimum length of the ordered chain (e.g. 3)
    seed : int, optional
        Seed for reproducibility
    
    Returns:
    --------
    str : ordered sequence
    """
    rng = np.random.default_rng(seed)
    
    # Start with a completely random sequence
    seq = [rng.choice(letters) for _ in range(n_events)]
    
    # Compute the maximum possible chain length given n_events and lag
    # If lag=7 and n_events=20: positions can be 0,7,14 -> max 3 letters
    # Formula: max_length = floor((n_events - 1) / lag) + 1
    max_possible_chain_length = (n_events - 1) // lag + 1
    
    # The chain length must be between min and min(max_possible, num_key_letters)
    max_chain_length = min(max_possible_chain_length, len(key_letters))
    
    if max_chain_length < min_chain_length:
        raise ValueError(
            f"Cannot create a chain of length {min_chain_length}: "
            f"with n_events={n_events} and lag={lag}, max possible is {max_possible_chain_length}"
        )
    
    # Choose the chain length (at least min_chain_length)
    chain_length = rng.integers(min_chain_length, max_chain_length + 1)
    
    # Compute the maximum start position
    # If chain_length=3 and lag=7: the last 3 elements must be at pos i, i+7, i+14
    # So i_max = n_events - 1 - (chain_length-1)*lag
    max_start = n_events - (chain_length - 1) * lag - 1
    
    if max_start < 0:
        raise ValueError(
            f"Cannot place chain: chain_length={chain_length}, "
            f"lag={lag}, n_events={n_events}"
        )
    
    # Choose a random start position
    start_pos = rng.integers(0, max_start + 1)
    
    # Select which key letters to use (in increasing rank order)
    available_keys = sorted(key_ordering.keys(), key=lambda x: key_ordering[x])
    
    # Choose chain_length letters keeping the order
    chosen_indices = sorted(rng.choice(len(available_keys), size=chain_length, replace=False))
    chosen_keys = [available_keys[i] for i in chosen_indices]
    
    # Place the chain letters at the correct positions
    for i, key in enumerate(chosen_keys):
        position = start_pos + i * lag
        seq[position] = key
    
    return '\x1f'.join(seq)


def generate_more_ordered_sequences(
    existing_ordered, 
    letters,
    key_letters,
    key_ordering,
    lag,
    n_events,
    min_chain_length,
    n_new,
    sep='\x1f',
    max_attempts_multiplier=10
):
    """
    Generate n_new new unique ordered sequences.
    
    Parameters:
    -----------
    existing_ordered : list
        Already existing ordered sequences
    letters : list
        All 26 letters
    key_letters : list
        The 6 key letters
    key_ordering : dict
        Mapping {letter: rank}
    lag : int
        Distance between consecutive letters (e.g. 7)
    n_events : int
        Sequence length (e.g. 20)
    min_chain_length : int
        Minimum chain length (e.g. 3)
    n_new : int
        How many new sequences to generate
    """
    
    existing_set = set(existing_ordered)
    new_sequences = []
    
    max_attempts = n_new * max_attempts_multiplier
    attempts = 0
    
    # Check that ordered sequences can be generated
    max_possible = (n_events - 1) // lag + 1
    if max_possible < min_chain_length:
        raise ValueError(
            f"Impossible: with n_events={n_events}, lag={lag}, "
            f"max chain length={max_possible} < min_chain_length={min_chain_length}"
        )
    
    print(f"Generating {n_new} new ordered sequences...")
    print(f"Parameters: n_events={n_events}, lag={lag}, min_chain={min_chain_length}")
    print(f"Max possible chain length: {max_possible}")
    
    with tqdm(total=n_new) as pbar:
        while len(new_sequences) < n_new and attempts < max_attempts:
            attempts += 1
            
            # Generate a new ordered sequence
            new_seq = generate_ordered_sequence_diverse(
                letters, key_letters, key_ordering, lag, n_events,
                min_chain_length=min_chain_length,
                seed=np.random.randint(0, 2**31)
            )
            
            # Check uniqueness
            if new_seq not in existing_set:
                new_sequences.append(new_seq)
                existing_set.add(new_seq)
                pbar.update(1)
    
    success_rate = len(new_sequences) / attempts if attempts > 0 else 0
    print(f"\nGenerated {len(new_sequences)}/{n_new} new unique sequences")
    print(f"Success rate: {success_rate:.2%} ({attempts} attempts)")
    
    if len(new_sequences) < n_new:
        print(f"Warning: Could not generate all {n_new} sequences. Got {len(new_sequences)}")
    
    return new_sequences


    # From the current code
letters = list(string.ascii_uppercase)
key_letters = list(c_vocab.keys())
lag = 7
n_events = 40
min_chain_length = 5

# Assume ordered sequences already exist
existing_ordered = check['valid_sequences']  # The current sequences

# Generate 10,000 new unique ordered sequences
new_ordered = generate_more_ordered_sequences(
    existing_ordered=existing_ordered,
    letters=letters,
    key_letters=key_letters,
    key_ordering=c_vocab,
    lag=lag,
    n_events=n_events,
    min_chain_length=min_chain_length,
    n_new=200000,
    sep='\x1f'
)

# Check that they are all ordered
print(f"\nVerifying new sequences are ordered...")
for seq in tqdm(new_ordered):  # Check a sample
    result = assign_outcome_positional(
        seq, c_vocab, lags=lag, 
        min_chain_length=min_chain_length,
        rnd=False
    )
    if result['outcome'][0] != 1:
        print(f"ERROR: Generated unordered sequence!")
else:
    print("All the other sampled sequences are correctly ordered!")

TTT = existing_ordered+new_ordered
len(TTT)



def plot_letter_frequency_by_position(sequences, sep='\x1f', figsize=(24, 16)):
    """
    Create a faceted plot showing the frequency of each letter at each position.
    
    Parameters:
    -----------
    sequences : list
        List of sequences (strings with separator)
    sep : str
        Separator in the sequences
    figsize : tuple
        Figure size
    """
    
    # Convert to list of lists
    seqs_lists = [seq.split(sep) for seq in sequences]
    n_positions = len(seqs_lists[0])
    
    # Setup subplot grid
    n_cols = 5
    n_rows = int(np.ceil(n_positions / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    letters_ord = string.ascii_uppercase
    
    for pos in range(n_positions):
        ax = axes[pos]
        
        # Count frequencies for this position
        letter_counts = collections.Counter()
        for seq in seqs_lists:
            if pos < len(seq):
                letter_counts[seq[pos]] += 1
        
        # Build the list of counts for all letters
        counts = [letter_counts.get(letter, 0) for letter in letters_ord]
        
        # Plot
        x = np.arange(len(letters_ord))
        ax.bar(x, counts, alpha=0.7, color='steelblue')
        
        ax.set_title(f'Position {pos+1}', fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(letters_ord, fontsize=6)
        ax.set_ylabel('Frequency', fontsize=8)
        
        # Grid
        ax.grid(axis='y', alpha=0.3)
        
        # Add line for uniform frequency
        uniform_freq = len(sequences) / 26
        ax.axhline(y=uniform_freq, color='red', linestyle='--', 
                   alpha=0.5, linewidth=1, label='Uniform')
        
        if pos == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for j in range(n_positions, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f'Letter Frequency by Position ({len(sequences)} sequences)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('letter_frequency_by_position.png', dpi=300, bbox_inches='tight')
    plt.show()


# Usage
plot_letter_frequency_by_position(new_ordered, sep='\x1f')

def calculate_position_uniformity(sequences, sep='\x1f'):
    """
    Compute how uniform the distribution is at each position.
    Uses Shannon entropy as the measure.
    """
    seqs_lists = [seq.split(sep) for seq in sequences]
    n_positions = len(seqs_lists[0])
    
    uniformity_scores = []
    
    for pos in range(n_positions):
        letter_counts = collections.Counter()
        for seq in seqs_lists:
            if pos < len(seq):
                letter_counts[seq[pos]] += 1
        
        # Compute entropy
        total = sum(letter_counts.values())
        probs = [count / total for count in letter_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Normalize (max entropy for 26 letters = log2(26) ≈ 4.7)
        max_entropy = np.log2(26)
        normalized_entropy = entropy / max_entropy
        
        uniformity_scores.append(normalized_entropy)
    
    return uniformity_scores


def subsample_uniform_distribution(sequences, target_size, sep='\x1f', 
                                   max_iterations=1000, tolerance=0.05):
    """
    Create a subsample trying to obtain a uniform distribution at each position.
    
    Strategy:
    1. Start with a random subsample
    2. For each position, identify over/under-represented letters
    3. Iteratively replace sequences to rebalance
    
    Parameters:
    -----------
    sequences : list
        Sequences to sample from
    target_size : int
        Desired subsample size
    tolerance : float
        How much we may deviate from uniform (0.05 = 5%)
    
    Returns:
    --------
    list : subsample of sequences
    """
    
    if target_size > len(sequences):
        print(f"Warning: target_size ({target_size}) > available sequences ({len(sequences)})")
        return sequences.copy()
    
    seqs_lists = [seq.split(sep) for seq in sequences]
    n_positions = len(seqs_lists[0])
    n_letters = 26
    
    # Target frequency per letter per position
    target_freq = target_size / n_letters
    
    print(f"Target frequency per letter per position: {target_freq:.1f}")
    print(f"Tolerance: ±{tolerance * 100}% = ±{target_freq * tolerance:.1f}")
    
    # Start with a random sample
    rng = np.random.default_rng(42)
    available_indices = list(range(len(sequences)))
    selected_indices = set(rng.choice(available_indices, size=target_size, replace=False))
    
    best_score = float('inf')
    best_sample = selected_indices.copy()
    
    for iteration in range(max_iterations):
        # Compute how far we are from uniform
        position_freqs = []
        total_deviation = 0
        
        for pos in range(n_positions):
            letter_counts = collections.Counter()
            for idx in selected_indices:
                seq = seqs_lists[idx]
                if pos < len(seq):
                    letter_counts[seq[pos]] += 1
            
            position_freqs.append(letter_counts)
            
            # Compute deviation from uniform
            for letter in string.ascii_uppercase:
                count = letter_counts.get(letter, 0)
                deviation = abs(count - target_freq)
                total_deviation += deviation
        
        # Update best
        if total_deviation < best_score:
            best_score = total_deviation
            best_sample = selected_indices.copy()
        
        # Check whether we are good enough
        avg_deviation = total_deviation / (n_positions * n_letters)
        if avg_deviation < tolerance * target_freq:
            print(f"Converged at iteration {iteration}")
            print(f"Average deviation: {avg_deviation:.2f}")
            break
        
        # Improve the sample
        # Find the most unbalanced position and letter
        max_over = 0
        max_under = 0
        over_pos, over_letter = None, None
        under_pos, under_letter = None, None
        
        for pos in range(n_positions):
            for letter in string.ascii_uppercase:
                count = position_freqs[pos].get(letter, 0)
                diff = count - target_freq
                
                if diff > max_over:
                    max_over = diff
                    over_pos = pos
                    over_letter = letter
                
                if diff < -max_under:
                    max_under = -diff
                    under_pos = pos
                    under_letter = letter
        
        # Find a sequence to remove (with the over-represented letter)
        to_remove = None
        for idx in selected_indices:
            seq = seqs_lists[idx]
            if over_pos < len(seq) and seq[over_pos] == over_letter:
                to_remove = idx
                break
        
        # Find a sequence to add (with the under-represented letter)
        to_add = None
        available = set(available_indices) - selected_indices
        for idx in available:
            seq = seqs_lists[idx]
            if under_pos < len(seq) and seq[under_pos] == under_letter:
                to_add = idx
                break
        
        # Swap if possible
        if to_remove is not None and to_add is not None:
            selected_indices.remove(to_remove)
            selected_indices.add(to_add)
        else:
            # Random swap
            to_remove = rng.choice(list(selected_indices))
            to_add = rng.choice(list(set(available_indices) - selected_indices))
            selected_indices.remove(to_remove)
            selected_indices.add(to_add)
    
    else:
        print(f"Reached max iterations ({max_iterations})")
        print(f"Best score: {best_score / (n_positions * n_letters):.2f}")
        selected_indices = best_sample
    
    # Return the selected sequences
    subsample = [sequences[i] for i in sorted(selected_indices)]
    
    # Final statistics
    print("\nFinal uniformity statistics:")
    uniformity = calculate_position_uniformity(subsample, sep)
    avg_uniformity = np.mean(uniformity)
    print(f"Average uniformity: {avg_uniformity:.3f} (1.0 = perfect uniform)")
    print(f"Min uniformity: {min(uniformity):.3f}")
    print(f"Max uniformity: {max(uniformity):.3f}")
    
    return subsample


# Usage
uniform_subsample = subsample_uniform_distribution(
    sequences=new_ordered,
    target_size=10000,
    sep='\x1f',
    max_iterations=1000,
    tolerance=0.05
)

# Plot for comparison
print("\nOriginal sequences:")
plot_letter_frequency_by_position(new_ordered[:10000], sep='\x1f')

print("\nUniform subsample:")
plot_letter_frequency_by_position(uniform_subsample, sep='\x1f')
