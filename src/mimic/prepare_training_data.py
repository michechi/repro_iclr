"""
Prepare MIMIC CKD cohort data for ordered vs shuffled training experiments.

Reads ckd_cohort_ccs.csv, builds vocabulary, creates stratified 60/20/20 splits,
and saves ordered + shuffled versions with identical patient assignments.

Usage:
    python src/mimic/prepare_training_data.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
INPUT_PATH = "data/processed/ckd_cohort_ccs.csv"
OUTPUT_DIR = "data/processed/mimic_training"
DELIMITER = "\x1f"


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load cohort
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} patients")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Parse JSON code lists
    df["codes_list"] = df["codes"].apply(json.loads)

    # Build vocabulary: 0 = padding, 1..N = CCS codes
    all_codes = sorted({code for codes in df["codes_list"] for code in codes})
    vocab = {code: idx + 1 for idx, code in enumerate(all_codes)}
    vocab_size = len(vocab) + 1  # +1 for padding at index 0
    print(f"Vocabulary size: {vocab_size} ({len(vocab)} unique codes + padding)")

    vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({"code_to_idx": vocab, "vocab_size": vocab_size}, f, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    # Create ordered sequences
    df["seq_ordered"] = df["codes_list"].apply(lambda codes: DELIMITER.join(codes))

    # Create shuffled sequences (same codes, random order)
    rng = random.Random(SEED)
    def shuffle_codes(codes):
        shuffled = list(codes)
        rng.shuffle(shuffled)
        return DELIMITER.join(shuffled)

    df["seq_shuffled"] = df["codes_list"].apply(shuffle_codes)

    # Stratified split: 60/20/20
    # First split: 60% train, 40% temp
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.4, stratify=df["label"], random_state=SEED
    )
    # Second split: 50/50 of temp -> 20% val, 20% test
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df.loc[temp_idx, "label"], random_state=SEED
    )

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    # Print split info
    for name, idx in splits.items():
        n = len(idx)
        pos = df.loc[idx, "label"].sum()
        print(f"  {name}: {n} patients, {pos} positive ({100*pos/n:.1f}%)")

    # Save files for both ordered and shuffled
    for data_type in ["ordered", "shuffled"]:
        seq_col = f"seq_{data_type}"
        for split_name, idx in splits.items():
            subset = df.loc[idx]

            x_df = pd.DataFrame({"Sequences": subset[seq_col].values})
            y_df = pd.DataFrame({"Outcome": subset["label"].values})

            x_path = os.path.join(OUTPUT_DIR, f"X_{split_name}_{data_type}.csv")
            y_path = os.path.join(OUTPUT_DIR, f"y_{split_name}_{data_type}.csv")
            x_df.to_csv(x_path, index=False)
            y_df.to_csv(y_path, index=False)

    print(f"\nSaved 12 CSV files to {OUTPUT_DIR}")

    # Verification
    print("\n--- Verification ---")
    x_train_ord = pd.read_csv(os.path.join(OUTPUT_DIR, "X_train_ordered.csv"))
    x_train_shuf = pd.read_csv(os.path.join(OUTPUT_DIR, "X_train_shuffled.csv"))
    print(f"Train ordered rows: {len(x_train_ord)}, shuffled rows: {len(x_train_shuf)}")

    # Check same codes in different order
    ord_codes = set(x_train_ord.iloc[0]["Sequences"].split(DELIMITER))
    shuf_codes = set(x_train_shuf.iloc[0]["Sequences"].split(DELIMITER))
    print(f"First patient same codes: {ord_codes == shuf_codes}")


if __name__ == "__main__":
    main()
