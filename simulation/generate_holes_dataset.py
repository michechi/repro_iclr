"""
Generate sequences with holes for ordering task.

Dataset specification:
- Sequences of 20 letters
- Order defined by random 6-letter permutation (key) from alphabet
- Lag system (lag=7): consecutive key letters appear with EXACTLY 'lag' positions between them
- 400K examples: 40% ordered, 60% unordered
- Holes: mask positions with metadata about key membership and valid alternatives

Terminology (IMPORTANT):
- A sequence is "ordered" if it contains at least one CHAIN of consecutive key letters
  with exactly 'lag' positions between each adjacent pair in the chain.
- Chain lengths:
  - pair: 2 consecutive key letters (e.g., A at pos 0, B at pos 7 with lag=7)
  - triplet: 3 consecutive key letters (e.g., A at pos 0, B at pos 7, C at pos 14)
  - quadruplet: 4 consecutive key letters, etc.
- A triplet is ONE chain, NOT "two separate pairs". For example, if letters A, B, C
  (consecutive in the key) appear at positions 0, 7, 14 with lag=7, this is stored as
  a single chain: (0,7,14:A,B,C), NOT as (0,7,A,B);(7,14,B,C).

Output columns:
- chain_length: length of the chain (2=pair, 3=triplet, 0=unordered)
- ordered_chain: chain in format (pos1,pos2,...:letter1,letter2,...), e.g., (0,7,14:I,H,V)
"""

import random
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configuration defaults
ALPHABET = list(string.ascii_uppercase)
SEQUENCE_LENGTH = 20
KEY_SIZE = 6
LAG = 7  # Minimum positions between two ordered key letters
TOTAL_SAMPLES = 400_000
ORDERED_RATIO = 0.40
SEED = 42


def generate_random_key(key_size=KEY_SIZE):
    """Generate a random permutation of key_size letters as the ordering key."""
    return random.sample(ALPHABET, key_size)


def is_sequence_ordered(sequence, key, lag=LAG):
    """
    Check if sequence is ordered: contains at least one CHAIN of consecutive key letters
    (pair, triplet, quadruplet, etc.) with EXACTLY 'lag' positions between each.

    Terminology:
    - pair: 2 consecutive key letters with exact lag spacing (chain_length=2)
    - triplet: 3 consecutive key letters with exact lag spacing (chain_length=3)
    - quadruplet: 4 consecutive key letters with exact lag spacing (chain_length=4)

    A triplet is ONE chain, NOT "two pairs". E.g., if key letters A, B, C appear at
    positions 0, 7, 14 (with lag=7), this is a single triplet (0,7,14:A,B,C).

    Args:
        sequence: list of letters
        key: ordered list of key letters
        lag: EXACT positions between consecutive key letters in the chain

    Returns:
        tuple: (bool is_ordered, list of chains)
               Each chain is a dict: {'positions': [pos1, pos2, ...], 'letters': [l1, l2, ...], 'key_start_idx': int}
    """
    # Find all positions of key letters in sequence
    key_positions = {}  # letter -> list of positions
    for i, letter in enumerate(sequence):
        if letter in key:
            if letter not in key_positions:
                key_positions[letter] = []
            key_positions[letter].append(i)

    # Find all maximal chains of consecutive key letters with exact lag spacing
    chains = []

    # For each possible starting position and key index, try to extend the chain
    for key_start_idx in range(len(key)):
        start_letter = key[key_start_idx]
        if start_letter not in key_positions:
            continue

        for start_pos in key_positions[start_letter]:
            # Try to build a chain starting here
            chain_positions = [start_pos]
            chain_letters = [start_letter]
            current_pos = start_pos
            current_key_idx = key_start_idx

            # Extend chain as far as possible
            while current_key_idx + 1 < len(key):
                next_key_idx = current_key_idx + 1
                next_letter = key[next_key_idx]
                next_pos = current_pos + lag

                if next_letter in key_positions and next_pos in key_positions[next_letter]:
                    chain_positions.append(next_pos)
                    chain_letters.append(next_letter)
                    current_pos = next_pos
                    current_key_idx = next_key_idx
                else:
                    break

            # Only add if chain has at least 2 elements (a pair or longer)
            if len(chain_positions) >= 2:
                # Check if this chain is already covered by a longer chain
                is_subchain = False
                for existing in chains:
                    if (chain_positions[0] in existing['positions'] and
                        chain_positions[-1] in existing['positions']):
                        is_subchain = True
                        break

                if not is_subchain:
                    # Remove any existing chains that are subchains of this one
                    chains = [c for c in chains if not (
                        c['positions'][0] >= chain_positions[0] and
                        c['positions'][-1] <= chain_positions[-1] and
                        c['positions'][0] in chain_positions
                    )]
                    chains.append({
                        'positions': chain_positions,
                        'letters': chain_letters,
                        'key_start_idx': key_start_idx
                    })

    return len(chains) > 0, chains


def generate_ordered_sequence(key, seq_length=SEQUENCE_LENGTH, lag=LAG):
    """
    Generate a sequence that is ordered: contains at least one CHAIN of consecutive
    key letters (pair, triplet, etc.) with EXACTLY 'lag' positions between them.

    With seq_length=20 and lag=7, max positions are 0, 7, 14 (chain_length up to 3).

    Terminology:
    - chain_length=2: pair (e.g., positions 0,7 with letters A,B)
    - chain_length=3: triplet (e.g., positions 0,7,14 with letters A,B,C)

    Strategy:
    - Choose chain length (2 or 3 for 20-len, lag=7)
    - Place consecutive key letters at positions with exactly 'lag' spacing
    - Fill remaining positions with non-key letters

    Returns:
        tuple: (sequence, chain_info)
               chain_info is a dict: {'positions': [...], 'letters': [...], 'key_start_idx': int}
    """
    non_key_letters = [l for l in ALPHABET if l not in key]

    # Calculate max chain length that can fit
    # Positions: 0, lag, 2*lag, ... up to seq_length-1
    max_chain_length = (seq_length - 1) // lag + 1  # For 20, lag=7: positions 0,7,14 = 3
    max_chain_length = min(max_chain_length, len(key))

    # Choose chain length (at least 2 for a pair)
    chain_length = random.randint(2, max_chain_length)

    # Choose starting index in the key (must have room for chain_length consecutive letters)
    max_start_idx = len(key) - chain_length
    key_start_idx = random.randint(0, max_start_idx)

    # Choose starting position in sequence
    # Last key letter will be at start_pos + (chain_length-1) * lag
    # This must be < seq_length
    max_start_pos = seq_length - 1 - (chain_length - 1) * lag
    start_pos = random.randint(0, max_start_pos)

    # Create sequence with non-key letters
    sequence = [random.choice(non_key_letters) for _ in range(seq_length)]

    # Place the consecutive key letters with exactly lag spacing (the chain)
    chain_positions = []
    chain_letters = []
    for i in range(chain_length):
        pos = start_pos + i * lag
        key_idx = key_start_idx + i
        letter = key[key_idx]
        sequence[pos] = letter
        chain_positions.append(pos)
        chain_letters.append(letter)

    chain_info = {
        'positions': chain_positions,
        'letters': chain_letters,
        'key_start_idx': key_start_idx
    }

    return sequence, chain_info


def generate_unordered_sequence(key, seq_length=SEQUENCE_LENGTH, lag=LAG):
    """
    Generate a sequence that is NOT ordered: no CHAIN of consecutive key letters
    appears with EXACTLY 'lag' positions between them.

    A sequence is unordered if it contains no pair, triplet, or any chain of
    consecutive key letters with exact lag spacing.

    Strategies:
    1. Use only non-key letters
    2. Place key letters in reverse order
    3. Place key letters with wrong spacing (not exactly lag)
    4. Place only one key letter
    """
    non_key_letters = [l for l in ALPHABET if l not in key]
    strategy = random.choice(['no_key', 'reverse', 'wrong_spacing', 'single'])

    if strategy == 'no_key':
        # No key letters at all
        sequence = [random.choice(non_key_letters) for _ in range(seq_length)]

    elif strategy == 'reverse':
        # Place two consecutive key letters in REVERSE order with exact lag
        key_idx1 = random.randint(0, len(key) - 2)
        key_idx2 = key_idx1 + 1

        letter1 = key[key_idx1]  # Should come first in key
        letter2 = key[key_idx2]  # Should come second in key

        # Place them in reverse order: letter2 first, then letter1 at +lag
        sequence = [random.choice(non_key_letters) for _ in range(seq_length)]
        max_pos2 = seq_length - lag - 1
        if max_pos2 >= 0:
            pos2 = random.randint(0, max_pos2)  # letter2 position
            pos1 = pos2 + lag  # letter1 comes after (wrong order!)
            sequence[pos2] = letter2
            sequence[pos1] = letter1

    elif strategy == 'wrong_spacing':
        # Place two consecutive key letters in correct order but NOT exactly lag apart
        key_idx1 = random.randint(0, len(key) - 2)
        key_idx2 = key_idx1 + 1

        letter1 = key[key_idx1]
        letter2 = key[key_idx2]

        sequence = [random.choice(non_key_letters) for _ in range(seq_length)]
        pos1 = random.randint(0, seq_length - 2)

        # Choose spacing that is NOT exactly lag
        possible_spacings = [s for s in range(1, seq_length - pos1) if s != lag]
        if possible_spacings:
            spacing = random.choice(possible_spacings)
            pos2 = pos1 + spacing
            if pos2 < seq_length:
                sequence[pos1] = letter1
                sequence[pos2] = letter2

    else:  # single
        # Only one key letter (can't form a pair)
        sequence = [random.choice(non_key_letters) for _ in range(seq_length)]
        pos = random.randint(0, seq_length - 1)
        sequence[pos] = random.choice(key)

    # Verify it's actually unordered
    is_ord, _ = is_sequence_ordered(sequence, key, lag)
    if is_ord:
        # If still ordered, remove all key letters
        sequence = [l if l not in key else random.choice(non_key_letters) for l in sequence]

    return sequence


def create_hole(sequence, key, chain_info=None, chains=None):
    """
    Create a hole (mask) in the sequence and analyze valid alternatives.

    Args:
        sequence: list of letters
        key: the ordering key
        chain_info: dict with 'positions', 'letters', 'key_start_idx' for the generated chain
        chains: list of chain dicts from is_sequence_ordered

    Returns:
        dict with hole information
    """
    # Find key letters in the sequence
    key_letter_positions = []
    for i, letter in enumerate(sequence):
        if letter in key:
            key_letter_positions.append((i, letter, key.index(letter)))

    non_key_positions = [i for i in range(len(sequence)) if sequence[i] not in key]

    # Decide whether to mask a key letter or non-key letter (50/50)
    if key_letter_positions and non_key_positions:
        mask_key = random.choice([True, False])
    elif key_letter_positions:
        mask_key = True
    else:
        mask_key = False

    if mask_key and key_letter_positions:
        # Mask a key letter
        pos, letter, key_idx = random.choice(key_letter_positions)
        is_key_letter = True

        # Find valid alternatives
        valid_alternatives = find_valid_alternatives_for_chain(
            sequence, pos, key, key_idx, chains
        )
    else:
        # Mask a non-key letter
        pos = random.choice(non_key_positions) if non_key_positions else random.randint(0, len(sequence) - 1)
        letter = sequence[pos]
        is_key_letter = False
        key_idx = -1
        valid_alternatives = list(ALPHABET)  # Any letter works for non-key position

    # Create sequence with hole
    sequence_with_hole = sequence.copy()
    sequence_with_hole[pos] = '[MASK]'

    return {
        'hole_position': pos,
        'original_letter': letter,
        'is_key_letter': is_key_letter,
        'key_index': key_idx,
        'valid_alternatives': valid_alternatives,
        'sequence_with_hole': sequence_with_hole
    }


def find_valid_alternatives_for_chain(sequence, hole_pos, key, key_idx, chains=None):
    """
    Find all letters that could fill the hole and maintain (or break) ordering.

    For an ordered sequence with a key letter masked:
    - The original key letter is valid (maintains ordering)
    - Other key letters that would still satisfy ordering are valid
    - Non-key letters may or may not maintain ordering

    Args:
        sequence: the sequence
        hole_pos: position of the hole
        key: the ordering key
        key_idx: index in key of the original letter at hole_pos
        chains: list of chain dicts from is_sequence_ordered

    Returns list of valid alternative letters.
    """
    valid = set()

    # Original key letter is always valid for maintaining the pattern
    valid.add(key[key_idx])

    # Check if this position is part of an ordering chain
    is_critical = False
    if chains:
        for chain in chains:
            if hole_pos in chain['positions']:
                is_critical = True
                break

    if is_critical:
        # This is a critical position in a chain - only key letters that preserve ordering are valid
        # Find all key letter positions in the sequence
        key_positions_in_seq = []
        for i, letter in enumerate(sequence):
            if letter in key:
                key_positions_in_seq.append((i, letter, key.index(letter)))

        # Find the key letters immediately before and after hole_pos in the sequence
        prev_key_idx = -1  # No constraint from the left
        next_key_idx = len(key)  # No constraint from the right

        for seq_pos, letter, k_idx in key_positions_in_seq:
            if seq_pos < hole_pos:
                # This is before the hole - track the maximum key index seen
                prev_key_idx = max(prev_key_idx, k_idx)
            elif seq_pos > hole_pos:
                # This is after the hole - track the minimum key index seen
                next_key_idx = min(next_key_idx, k_idx)

        # Valid key letters are those with index strictly between prev_key_idx and next_key_idx
        # This preserves the ordering: prev < fill < next
        for k_idx, k_letter in enumerate(key):
            if prev_key_idx < k_idx < next_key_idx:
                valid.add(k_letter)
    else:
        # Not a critical position - any letter works
        valid.update(ALPHABET)

    return list(valid)


def generate_dataset(
    total_samples=TOTAL_SAMPLES,
    ordered_ratio=ORDERED_RATIO,
    seq_length=SEQUENCE_LENGTH,
    key_size=KEY_SIZE,
    lag=LAG,
    seed=SEED
):
    """Generate the full dataset."""

    random.seed(seed)
    np.random.seed(seed)

    n_ordered = int(total_samples * ordered_ratio)
    n_unordered = total_samples - n_ordered

    # Generate ONE fixed key for the entire dataset
    key = generate_random_key(key_size)

    logger.info(f"Generating {total_samples:,} samples:")
    logger.info(f"  - Ordered: {n_ordered:,} ({ordered_ratio*100:.0f}%)")
    logger.info(f"  - Unordered: {n_unordered:,} ({(1-ordered_ratio)*100:.0f}%)")
    logger.info(f"  - Sequence length: {seq_length}")
    logger.info(f"  - Key size: {key_size}")
    logger.info(f"  - Lag: {lag}")
    logger.info(f"  - FIXED KEY: {''.join(key)} (ordering: {' -> '.join(key)})")

    data = []

    # Helper function to format a chain as string: (pos1,pos2,...:letter1,letter2,...)
    def format_chain(chain):
        positions_str = ','.join(map(str, chain['positions']))
        letters_str = ','.join(chain['letters'])
        return f"({positions_str}:{letters_str})"

    # Generate ordered sequences
    logger.info("Generating ordered sequences...")
    for _ in tqdm(range(n_ordered), desc="Ordered"):
        sequence, chain_info = generate_ordered_sequence(key, seq_length, lag)

        # Verify it's actually ordered
        is_ord, chains = is_sequence_ordered(sequence, key, lag)
        assert is_ord, "Generated sequence is not ordered!"

        # Create hole
        hole_info = create_hole(sequence, key, chain_info, chains)

        # Store the ordered chain (there should typically be one chain)
        # Format: (pos1,pos2,pos3:letter1,letter2,letter3)
        # For multiple chains, separate with semicolons
        chain_str = ';'.join([format_chain(c) for c in chains]) if chains else ''

        data.append({
            'sequence': ''.join(sequence),
            'sequence_with_hole': ''.join(hole_info['sequence_with_hole']),
            'key': ''.join(key),
            'is_ordered': 1,
            'hole_position': hole_info['hole_position'],
            'original_letter': hole_info['original_letter'],
            'is_key_letter': hole_info['is_key_letter'],
            'key_index': hole_info['key_index'],
            'valid_alternatives': ','.join(hole_info['valid_alternatives']),
            'chain_length': len(chain_info['positions']),
            'ordered_chain': chain_str
        })

    # Generate unordered sequences
    logger.info("Generating unordered sequences...")
    for _ in tqdm(range(n_unordered), desc="Unordered"):
        sequence = generate_unordered_sequence(key, seq_length, lag)

        # Verify it's actually unordered
        is_ord, _ = is_sequence_ordered(sequence, key, lag)
        if is_ord:
            # Force unordered by removing key letters
            non_key = [l for l in ALPHABET if l not in key]
            sequence = [l if l not in key else random.choice(non_key) for l in sequence]

        # Create hole
        hole_info = create_hole(sequence, key, None, None)

        data.append({
            'sequence': ''.join(sequence),
            'sequence_with_hole': ''.join(hole_info['sequence_with_hole']),
            'key': ''.join(key),
            'is_ordered': 0,
            'hole_position': hole_info['hole_position'],
            'original_letter': hole_info['original_letter'],
            'is_key_letter': hole_info['is_key_letter'],
            'key_index': hole_info['key_index'],
            'valid_alternatives': ','.join(hole_info['valid_alternatives']),
            'chain_length': 0,
            'ordered_chain': ''
        })

    # Shuffle the dataset
    random.shuffle(data)

    df = pd.DataFrame(data)

    logger.info(f"\nDataset statistics:")
    logger.info(f"  Total samples: {len(df):,}")
    logger.info(f"  Ordered ratio: {df['is_ordered'].mean():.2%}")
    logger.info(f"  Key letter masked: {df['is_key_letter'].mean():.2%}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate sequences with holes dataset")
    parser.add_argument("--total_samples", type=int, default=TOTAL_SAMPLES)
    parser.add_argument("--ordered_ratio", type=float, default=ORDERED_RATIO)
    parser.add_argument("--seq_length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--key_size", type=int, default=KEY_SIZE)
    parser.add_argument("--lag", type=int, default=LAG)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default="./data/simulation/holes/generated_holes_dataset.csv")

    args = parser.parse_args()

    df = generate_dataset(
        total_samples=args.total_samples,
        ordered_ratio=args.ordered_ratio,
        seq_length=args.seq_length,
        key_size=args.key_size,
        lag=args.lag,
        seed=args.seed
    )

    # Save dataset
    logger.info(f"\nSaving to {args.output}...")
    df.to_csv(args.output, index=False)

    # Print sample
    logger.info("\nSample entries:")
    print(df.head(10).to_string())

    return df


if __name__ == "__main__":
    main()
