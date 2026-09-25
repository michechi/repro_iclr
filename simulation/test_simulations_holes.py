import numpy as np
import pandas as pd
import string
import random
import json
import itertools
from typing import List, Tuple, Optional, Union

from simulation.do_check_lag import check_lag


def generate_bigram_alphabet(shuffle: bool = False, seed: int = None) -> List[str]:
    """
    Generate alphabet of 26*26 = 676 bigrams: AA, AB, ..., AZ, BA, ..., ZZ
    """
    single_letters = list(string.ascii_uppercase)
    bigrams = [a + b for a, b in itertools.product(single_letters, single_letters)]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(bigrams)

    return bigrams  # 676 elements

def generate_sequences(
    letters: list, n: int = 10, m: int = 10_000, replacement: bool = True, seed: int = None
) -> list:
    """
    Generate m sequences of length n from the given letters.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(letters, dtype='<U8')
    Llen = len(L)

    if not replacement and n > Llen:
        raise ValueError("n must be <= len(letters) when replacement=False")

    SEP = "\x1f"
    seen = set()
    out_rows = []
    batch_size = min(10_000, m // 10) if m > 100 else m

    def _gen_batch(k):
        if replacement:
            li = rng.integers(0, Llen, size=(k, n))
        else:
            l_scores = rng.random((k, Llen))
            li = np.argsort(l_scores, axis=1)[:, :n]
        seq = L[li]
        return seq

    while len(out_rows) < m:
        need = m - len(out_rows)
        k = max(batch_size, need)
        seq = _gen_batch(k)
        keys = [SEP.join(key) for key in seq.tolist()]
        mask_new = np.fromiter((key not in seen for key in keys), count=k, dtype=bool)
        if not mask_new.any():
            continue
        new_keys = [key for key, to_keep in zip(keys, mask_new) if to_keep]
        out_rows += new_keys
        for key in new_keys:
            seen.add(key)

    return out_rows[:m]


def is_ordered(seq: str, c_ord: dict, sep: str = "\x1f") -> bool:
    """
    Check if a sequence is ordered according to the given ordering dictionary.
    Only considers letters that are in c_ord keys.
    """
    l_keys = c_ord.keys()
    seq_splt = [x for x in seq.split(sep) if x in l_keys]

    if len(seq_splt) < 2:
        return True  # Single element or empty is considered ordered

    for x, y in zip(seq_splt[:-1], seq_splt[1:]):
        if c_ord[x] > c_ord[y]:
            return False
    return True


def is_ordered_with_lag(
    seq: str,
    c_ord: dict,
    lags: Union[int, List[int]],
    sep: str = "\x1f",
    min_chain_length: int = 2,
    tolerance: bool = False
) -> bool:
    """
    Check if a sequence is ordered according to the given ordering dictionary,
    considering only key elements that appear at specific lag distances.

    Args:
        seq: The sequence string (elements separated by sep)
        c_ord: Ordering dictionary {element: position}
        lags: Distance(s) between key elements. Can be:
            - int: fixed distance between all elements
            - List[int]: variable distances [d1, d2, d3, ...]
        sep: Separator between elements
        min_chain_length: Minimum number of lagged keys required to consider ordering
        tolerance: If True, allows one out-of-order pair (cyclic tolerance)

    Returns:
        True if the lagged key elements are in order, False otherwise
    """
    l_keys = list(c_ord.keys())
    seq_splt = seq.split(sep)

    # Find key elements at specified lags
    are_lagged_keys = check_lag(seq_splt, l_keys, lags)

    if not are_lagged_keys:
        return False  # No lagged keys found

    # Handle case where check_lag returns multiple chains
    if all(isinstance(item, list) for item in are_lagged_keys):
        # Multiple chains - check if any chain is ordered
        for subseq in are_lagged_keys:
            if len(subseq) < min_chain_length:
                continue

            # Check if this chain is ordered
            chain_ordered = True
            tol = tolerance
            for x, y in zip(subseq[:-1], subseq[1:]):
                if c_ord[x] > c_ord[y]:
                    if tol:
                        tol = False  # Use up tolerance
                    else:
                        chain_ordered = False
                        break

            if chain_ordered:
                return True  # At least one chain is ordered

        return False  # No chain is ordered

    else:
        # Single chain
        if len(are_lagged_keys) < min_chain_length:
            return False

        tol = tolerance
        for x, y in zip(are_lagged_keys[:-1], are_lagged_keys[1:]):
            if c_ord[x] > c_ord[y]:
                if tol:
                    tol = False
                else:
                    return False

        return True


def create_hole(
    seq: str,
    n_holes: int = 1,
    hole_token: str = "[MASK]",
    sep: str = "\x1f",
    seed: int = None
) -> Tuple[str, List[str], List[int]]:
    """
    Create holes in a sequence by replacing random positions with hole_token.

    Returns:
        - sequence with holes
        - list of removed elements (ground truth)
        - list of positions where holes were created
    """
    rng = np.random.default_rng(seed)
    elements = seq.split(sep)
    n = len(elements)

    n_holes = min(n_holes, n)  # Can't have more holes than elements
    hole_positions = sorted(rng.choice(n, size=n_holes, replace=False).tolist())

    removed = [elements[i] for i in hole_positions]

    for pos in hole_positions:
        elements[pos] = hole_token

    return sep.join(elements), removed, hole_positions


def create_prompt(
    seq_with_holes: str,
    is_seq_ordered: bool,
    hole_positions: List[int],
    key_letters: List[str],
    sep: str = "\x1f",
    hole_token: str = "[MASK]",
    prompt_type: str = "binary"
) -> str:
    """
    Create a prompt for LLM. Does NOT reveal the key letters.

    prompt_type:
        - "binary": Simple classification - is sequence ordered? (Yes/No)
        - "generation": Fill in the hole (free-form, tells if ordered or not)
    """
    readable_seq = seq_with_holes.replace(sep, " -> ")

    if prompt_type == "binary":
        # Binary classification: is this sequence ordered?
        # NO key letters revealed - model must learn from data
        prompt = f"Sequence: {readable_seq}\nIs this sequence ordered? Answer: Yes or No"

    elif prompt_type == "generation":
        # Generation task: fill the hole (still tells if ordered)
        order_info = "ORDERED" if is_seq_ordered else "NOT ORDERED"
        prompt = f"Sequence: {readable_seq}\nThis sequence is {order_info}.\nFill {hole_token}. Answer with ONLY the element."

    else:
        prompt = f"Sequence: {readable_seq}"

    return prompt


def create_prompt_binary_no_holes(seq: str, is_ordered: bool, sep: str = "\x1f") -> Tuple[str, int]:
    """
    Simplest binary classification: sequence -> ordered/not.
    No holes, no hints. Model must learn the hidden pattern.

    Returns: (prompt, label)
    """
    readable_seq = seq.replace(sep, " -> ")
    prompt = f"Sequence: {readable_seq}\nIs this sequence ordered?"
    return prompt, int(is_ordered)


def generate_dataset_with_holes(
    n_sequences: int = 1000,
    seq_length: int = 9,
    n_holes: int = 1,
    hole_token: str = "[MASK]",
    key_letters: List[str] = None,
    all_letters: List[str] = None,
    balance_ordered: bool = True,
    ordered_ratio: float = 0.5,
    seed: int = 42,
    prompt_type: str = "binary",  # "binary" or "generation"
    lags: Union[int, List[int]] = None,
    min_chain_length: int = 2,
    tolerance: bool = False
) -> pd.DataFrame:
    """
    Generate a dataset of sequences with holes for LLM completion task.

    Args:
        n_sequences: Number of sequences to generate
        seq_length: Length of each sequence
        n_holes: Number of holes per sequence
        hole_token: Token to represent holes
        key_letters: Letters that determine ordering (subset of all_letters)
        all_letters: All possible letters in sequences
        balance_ordered: If True, try to balance ordered/unordered sequences
        ordered_ratio: Ratio of ordered sequences (0.0 to 1.0), only used if balance_ordered=True
        seed: Random seed
        prompt_type: "binary" (is it ordered?) or "generation" (fill the hole)
        lags: Distance(s) between key elements for lag-based ordering. If None, uses simple ordering.
            - int: fixed distance between all elements
            - List[int]: variable distances [d1, d2, d3, ...]
        min_chain_length: Minimum number of lagged keys required (only used when lags is set)
        tolerance: If True, allows one out-of-order pair (only used when lags is set)

    Returns:
        DataFrame with columns: original_seq, seq_with_holes, is_ordered,
                               ground_truth, hole_positions, prompt
    """
    random.seed(seed)
    np.random.seed(seed)

    if all_letters is None:
        all_letters = list(string.ascii_uppercase)

    if key_letters is None:
        key_letters = ["W", "D", "Q", "J", "U", "H"]  # Default from test_simulations

    # Create ordering vocabulary
    c_vocab = {w: p for p, w in enumerate(key_letters, start=0)}

    # Generate more sequences than needed to allow filtering
    n_generate = n_sequences * 3 if balance_ordered else n_sequences
    sequences = generate_sequences(
        letters=all_letters,
        n=seq_length,
        m=n_generate,
        replacement=True,
        seed=seed
    )

    # Classify sequences
    ordered_seqs = []
    unordered_seqs = []

    # Choose ordering function based on whether lags are specified
    if lags is not None:
        def check_order(seq):
            return is_ordered_with_lag(seq, c_vocab, lags, min_chain_length=min_chain_length, tolerance=tolerance)
    else:
        def check_order(seq):
            return is_ordered(seq, c_vocab)

    for seq in sequences:
        if check_order(seq):
            ordered_seqs.append(seq)
        else:
            unordered_seqs.append(seq)

    print(f"Generated {len(ordered_seqs)} ordered and {len(unordered_seqs)} unordered sequences")

    # Balance if requested
    if balance_ordered:
        n_ordered = int(n_sequences * ordered_ratio)
        n_unordered = n_sequences - n_ordered
        selected_ordered = ordered_seqs[:n_ordered] if len(ordered_seqs) >= n_ordered else ordered_seqs
        selected_unordered = unordered_seqs[:n_unordered] if len(unordered_seqs) >= n_unordered else unordered_seqs
        final_sequences = selected_ordered + selected_unordered
        random.shuffle(final_sequences)
    else:
        final_sequences = sequences[:n_sequences]

    # Create dataset with holes
    data = []
    for i, seq in enumerate(final_sequences):
        seq_ordered = check_order(seq)

        if n_holes > 0:
            seq_with_holes, ground_truth, hole_positions = create_hole(
                seq, n_holes=n_holes, hole_token=hole_token, seed=seed + i
            )
        else:
            # No holes - use original sequence (for pure binary classification)
            seq_with_holes = seq
            ground_truth = []
            hole_positions = []

        prompt = create_prompt(
            seq_with_holes, seq_ordered, hole_positions, key_letters,
            hole_token=hole_token, prompt_type=prompt_type
        )

        data.append({
            "original_seq": seq,
            "seq_with_holes": seq_with_holes,
            "is_ordered": int(seq_ordered),
            "ground_truth": ",".join(ground_truth) if ground_truth else "",
            "hole_positions": ",".join(map(str, hole_positions)) if hole_positions else "",
            "prompt": prompt
        })

    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame, base_path: str = "data/simulation/holes"):
    """
    Save dataset in multiple formats.
    """
    import os
    os.makedirs(base_path, exist_ok=True)

    # CSV format
    df.to_csv(f"{base_path}/sequences_with_holes.csv", index=False)

    # JSONL format (useful for LLM fine-tuning)
    with open(f"{base_path}/sequences_with_holes.jsonl", "w") as f:
        for _, row in df.iterrows():
            json_obj = {
                "prompt": row["prompt"],
                "completion": row["ground_truth"],
                "metadata": {
                    "original_seq": row["original_seq"],
                    "is_ordered": row["is_ordered"],
                    "hole_positions": row["hole_positions"]
                }
            }
            f.write(json.dumps(json_obj) + "\n")

    print(f"Saved dataset to {base_path}/")
    print(f"  - CSV: sequences_with_holes.csv")
    print(f"  - JSONL: sequences_with_holes.jsonl")


if __name__ == "__main__":
    # Configuration
    N_SEQUENCES = 200_000
    SEQ_LENGTH = 50
    N_HOLES = 1  # Number of holes per sequence
    HOLE_TOKEN = "[MASK]"
    SEED = 959693
    ORDERED_RATIO = 0.4  # 40% ordered, 60% unordered

    # Lag configuration: distances between key elements
    # Set to None for simple ordering (any distance), or specify lags:
    # - int: fixed distance (e.g., 3 means keys must be 3 positions apart)
    # - List[int]: variable distances (e.g., [4, 3, 2] means 1st-2nd: 4, 2nd-3rd: 3, 3rd-4th: 2)
    LAGS = 7
    MIN_CHAIN_LENGTH = 3  # Minimum number of lagged keys required
    TOLERANCE = False  # Allow one out-of-order pair

    # Use bigram alphabet (676 elements) instead of single letters (26)
    # This reduces n-gram pattern leakage between train/test
    USE_BIGRAMS = True

    # Prompt type: "binary" (is it ordered?) or "generation" (fill the hole)
    PROMPT_TYPE = "binary"

    if USE_BIGRAMS:
        # Generate 676 bigrams: AA, AB, ..., ZZ
        ALL_LETTERS = generate_bigram_alphabet(shuffle=False, seed=SEED)

        # Select key bigrams for ordering (e.g., 6-10 bigrams spread across the alphabet)
        # These are the bigrams that determine if sequence is "ordered"
        rng = random.Random(SEED)
        KEY_LETTERS = sorted(rng.sample(ALL_LETTERS, k=200))  # 8 key bigrams
    else:
        ALL_LETTERS = list(string.ascii_uppercase)
        KEY_LETTERS = ["W", "D", "Q", "J", "U", "H"]

    print("Generating sequences dataset...")
    print(f"  - {N_SEQUENCES} sequences")
    print(f"  - {SEQ_LENGTH} elements per sequence")
    print(f"  - {N_HOLES} hole(s) per sequence")
    print(f"  - Alphabet size: {len(ALL_LETTERS)} ({'bigrams' if USE_BIGRAMS else 'single letters'})")
    print(f"  - Key elements for ordering ({len(KEY_LETTERS)}): {KEY_LETTERS}")
    print(f"  - Ordered/Unordered ratio: {ORDERED_RATIO*100:.0f}% / {(1-ORDERED_RATIO)*100:.0f}%")
    if LAGS is not None:
        print(f"  - Lags: {LAGS} (min chain: {MIN_CHAIN_LENGTH}, tolerance: {TOLERANCE})")
    else:
        print(f"  - Lags: None (simple ordering, any distance)")
    print(f"  - Prompt type: {PROMPT_TYPE}")
    print()

    df = generate_dataset_with_holes(
        n_sequences=N_SEQUENCES,
        seq_length=SEQ_LENGTH,
        n_holes=N_HOLES,
        hole_token=HOLE_TOKEN,
        key_letters=KEY_LETTERS,
        all_letters=ALL_LETTERS,
        balance_ordered=True,
        ordered_ratio=ORDERED_RATIO,
        seed=SEED,
        prompt_type=PROMPT_TYPE,
        lags=LAGS,
        min_chain_length=MIN_CHAIN_LENGTH,
        tolerance=TOLERANCE
    )

    # Show sample
    print("\nSample entries:")
    print("-" * 80)
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\nEntry {i+1}:")
        print(f"  Original: {row['original_seq'].replace(chr(31), ' -> ')}")
        print(f"  With holes: {row['seq_with_holes'].replace(chr(31), ' -> ')}")
        print(f"  Is ordered: {bool(row['is_ordered'])}")
        print(f"  Ground truth: {row['ground_truth']}")
        print(f"  Hole positions: {row['hole_positions']}")

    # Statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics:")
    print(f"  Total sequences: {len(df)}")
    print(f"  Ordered sequences: {df['is_ordered'].sum()} ({df['is_ordered'].mean()*100:.1f}%)")
    print(f"  Unordered sequences: {len(df) - df['is_ordered'].sum()} ({(1-df['is_ordered'].mean())*100:.1f}%)")

    # Save dataset
    save_dataset(df)

    # Print a full prompt example
    print("\n" + "=" * 80)
    print("Example prompt for LLM:")
    print("-" * 80)
    print(df.iloc[0]["prompt"])
