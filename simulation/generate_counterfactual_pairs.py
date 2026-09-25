"""
Generate matched-histogram counterfactual pairs for Tricky Det / Tricky Rnd.

For each test sequence, shuffle letters (preserving the histogram exactly)
and re-label with the deterministic rule.  If the label flips, store the
(original, shuffled) pair.

Output:
  - paper_tables/counterfactual_pairs_{dataset}.csv
    Columns: pair_id, seq_orig, seq_shuf, label_orig, label_shuf, label_latent_orig

Usage:
    python -m simulation.generate_counterfactual_pairs
"""

import random
import string
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from simulation.do_check_lag import check_lag

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data/simulation/tested")
OUTPUT_DIR = Path("paper_tables")
OUTPUT_DIR.mkdir(exist_ok=True)

SEP = "\x1f"
SEED = 42
MAX_SHUFFLES = 50     # max attempts per sequence to find a label-flipping shuffle
KEY_LETTERS_9 = ["W", "D", "Q", "J", "X", "U"]
C_VOCAB_9 = {letter: rank for rank, letter in enumerate(KEY_LETTERS_9)}
LAG = 7
TOLERANCE = False
MIN_CHAIN = 2

# Only dataset 9 — labeling function verified:
# P(obs=1|lat=1)=0.703, P(obs=1|lat=0)=0.302
DATASETS = {
    "tricky_rnd": ("9", C_VOCAB_9),
}


# ──────────────────────────────────────────────────────────────────────
# LABELING (deterministic rule — matches generate_sensitivity_datasets.py)
# ──────────────────────────────────────────────────────────────────────
def _check_chain(chain, c_ord, min_chain_length, tolerance):
    if len(chain) < min_chain_length:
        return False
    tol = tolerance
    for x, y in zip(chain[:-1], chain[1:]):
        if c_ord[x[0]] > c_ord[y[0]]:
            if tol:
                tol = False
            else:
                return False
    return True


def label_sequence(seq_str, c_vocab):
    """Return deterministic (latent) label for a single sequence string."""
    tokens = seq_str.split(SEP)
    lagged = check_lag(tokens, c_vocab.keys(), LAG)
    if not lagged:
        return 0
    if all(isinstance(item, list) for item in lagged):
        return int(any(
            _check_chain(chain, c_vocab, MIN_CHAIN, TOLERANCE)
            for chain in lagged
        ))
    return int(_check_chain(lagged, c_vocab, MIN_CHAIN, TOLERANCE))


# ──────────────────────────────────────────────────────────────────────
# PAIR GENERATION
# ──────────────────────────────────────────────────────────────────────
def generate_pairs(X_test, y_test, c_vocab, max_shuffles=MAX_SHUFFLES, seed=SEED):
    """
    For each test sequence, try random shuffles until the *latent* label
    flips.  Returns a DataFrame of matched pairs.
    """
    rng = random.Random(seed)
    pairs = []
    n_tried = 0
    n_found = 0

    for idx in tqdm(range(len(X_test)), desc="Mining pairs"):
        seq_orig = X_test.Sequences.iloc[idx]
        label_orig_obs = int(y_test.Outcome.iloc[idx])
        label_orig_latent = label_sequence(seq_orig, c_vocab)

        tokens = seq_orig.split(SEP)

        for _ in range(max_shuffles):
            shuffled = tokens.copy()
            rng.shuffle(shuffled)
            seq_shuf = SEP.join(shuffled)
            label_shuf = label_sequence(seq_shuf, c_vocab)

            if label_shuf != label_orig_latent:
                # Found a label-flipping shuffle
                # Ensure the positive is first for consistent pair orientation
                if label_orig_latent == 1:
                    pairs.append({
                        "pair_id": n_found,
                        "seq_pos": seq_orig,
                        "seq_neg": seq_shuf,
                        "label_pos_obs": label_orig_obs,
                    })
                else:
                    pairs.append({
                        "pair_id": n_found,
                        "seq_pos": seq_shuf,
                        "seq_neg": seq_orig,
                        "label_neg_obs": label_orig_obs,
                    })
                n_found += 1
                break

        n_tried += 1

    print(f"  Tried {n_tried:,} sequences, found {n_found:,} pairs "
          f"({100*n_found/max(n_tried,1):.1f}%)")
    return pd.DataFrame(pairs)


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    for name, (csv_id, c_vocab) in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  {name} (csv_id={csv_id})")
        print(f"{'='*60}")

        X_test = pd.read_csv(DATA_DIR / f"X_test_{csv_id}.csv").fillna("")
        y_test = pd.read_csv(DATA_DIR / f"y_test_{csv_id}.csv").fillna("")
        print(f"  Test set: {len(X_test):,} sequences")
        print(f"  Key letters: {list(c_vocab.keys())}")

        # Verify labeling function via noise model consistency
        sample_size = min(2000, len(X_test))
        latent = np.array([label_sequence(s, c_vocab)
                           for s in X_test.Sequences.iloc[:sample_size]])
        observed = y_test.Outcome.iloc[:sample_size].values
        p1_lat1 = observed[latent == 1].mean() if (latent == 1).sum() > 0 else 0
        p1_lat0 = observed[latent == 0].mean() if (latent == 0).sum() > 0 else 0
        print(f"  Label verification: P(obs=1|lat=1)={p1_lat1:.3f} "
              f"P(obs=1|lat=0)={p1_lat0:.3f}")

        pairs_df = generate_pairs(X_test, y_test, c_vocab, seed=SEED)

        out_path = OUTPUT_DIR / f"counterfactual_pairs_{name}.csv"
        pairs_df.to_csv(out_path, index=False)
        print(f"  Saved {len(pairs_df):,} pairs to {out_path}")

        # Summary stats
        if len(pairs_df) > 0:
            print(f"\n  Summary:")
            print(f"    Total pairs:          {len(pairs_df):,}")
            print(f"    Coverage:             {100*len(pairs_df)/len(X_test):.1f}% "
                  f"of test sequences yielded a pair")


if __name__ == "__main__":
    main()
