import pandas as pd
import numpy as np

def is_alphabetically_ordered(seq, strict=False):
    """
    Check whether a sequence is alphabetically ordered.

    Args:
        seq: string with letters separated by '\x1f'
        strict: if True, requires strict order (A < B < C)
                if False, allows equalities (A <= B <= C)

    Returns:
        bool: True if the sequence is ordered
    """
    letters = seq.split('\x1f')
    for i in range(len(letters) - 1):
        if strict:
            if letters[i] >= letters[i+1]:
                return False
        else:
            if letters[i] > letters[i+1]:
                return False
    return True


def analyze_alphabetical_order(csv_number='9'):
    """Analyze the percentage of alphabetically ordered sequences."""

    print(f"=== Alphabetical order analysis for csv_number={csv_number} ===\n")

    # Load the data
    X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv",
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv",
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv",
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv",
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv",
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv",
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    datasets = {
        'Train': (X_train, y_train),
        'Val': (X_val, y_val),
        'Test': (X_test, y_test)
    }

    for name, (X, y) in datasets.items():
        print(f"--- {name} (n={len(X)}) ---")

        # Non-strict order (<=)
        ordered_nonstrict = X['Sequences'].apply(lambda s: is_alphabetically_ordered(s, strict=False))
        # Strict order (<)
        ordered_strict = X['Sequences'].apply(lambda s: is_alphabetically_ordered(s, strict=True))

        pct_nonstrict = ordered_nonstrict.mean() * 100
        pct_strict = ordered_strict.mean() * 100

        print(f"  Non-strict order (A <= B): {ordered_nonstrict.sum():,} / {len(X):,} = {pct_nonstrict:.4f}%")
        print(f"  Strict order     (A < B):  {ordered_strict.sum():,} / {len(X):,} = {pct_strict:.4f}%")

        # Breakdown per class (label)
        for label in [0, 1]:
            mask = y['Outcome'] == label
            n_label = mask.sum()

            ordered_ns_label = ordered_nonstrict[mask].sum()
            ordered_s_label = ordered_strict[mask].sum()

            pct_ns = (ordered_ns_label / n_label * 100) if n_label > 0 else 0
            pct_s = (ordered_s_label / n_label * 100) if n_label > 0 else 0

            print(f"    Label={label}: non-strict {ordered_ns_label:,}/{n_label:,} = {pct_ns:.4f}% | "
                  f"strict {ordered_s_label:,}/{n_label:,} = {pct_s:.4f}%")

        print()

    # Additional analysis: maximum length of an ordered subsequence
    print("--- Mean length of the longest ordered subsequence ---")

    def longest_ordered_subsequence(seq):
        """Find the length of the longest ordered subsequence."""
        letters = seq.split('\x1f')
        max_len = 1
        current_len = 1
        for i in range(1, len(letters)):
            if letters[i] >= letters[i-1]:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 1
        return max_len

    for name, (X, y) in datasets.items():
        lengths = X['Sequences'].apply(longest_ordered_subsequence)
        print(f"  {name}: mean={lengths.mean():.2f}, median={lengths.median():.1f}, max={lengths.max()}")

        for label in [0, 1]:
            mask = y['Outcome'] == label
            lengths_label = lengths[mask]
            print(f"    Label={label}: mean={lengths_label.mean():.2f}, median={lengths_label.median():.1f}")


if __name__ == "__main__":
    analyze_alphabetical_order('9')
