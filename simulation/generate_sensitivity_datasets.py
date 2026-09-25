"""
Generate synthetic datasets for parameter sensitivity analysis (Q1 response).

Produces tricky-deterministic datasets with varied (n, m, λ) configurations,
resampled to a fixed class balance ρ to isolate structural effects from
class-imbalance effects.

Pipeline:
    1. Generate random sequences and label deterministically (ordered → 1)
    2. Resample to target ρ_int using sample_with_proportion
    3. If --rnd: apply symmetric label noise post-hoc (π = 1 − pr_1)

This ordering ensures ρ_int is controlled by resampling and π by noise,
with ρ_obs = ρ_int(1−π) + (1−ρ_int)π following deterministically.

Parallelisation:
    - Labeling uses batch-parallel multiprocessing (no per-sequence DataFrame)
    - Configs with the same n share the same generated sequences

Usage:
    python -m simulation.generate_sensitivity_datasets --all \
        --output_dir data/simulation/sensitivity/
    python -m simulation.generate_sensitivity_datasets --list
"""

import argparse
import logging
import os
import random
import string
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from simulation.do_check_lag import check_lag
from simulation.utils import sample_with_proportion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALPHABET = list(string.ascii_uppercase)
SEP = "\x1f"
SEED = 42
RAW_SAMPLES = 10_000_000     # large pool for resampling
FINAL_SAMPLES = 400_000       # target size after resampling to fixed ρ
DEFAULT_RHO = 0.293            # baseline class balance (tricky deterministic)

# ---------------------------------------------------------------------------
# Configuration grid
# ---------------------------------------------------------------------------
# Structural grid: n × m × λ, all resampled to ρ = 0.293
#   n ∈ {15, 20, 25},  m ∈ {3, 6, 10},  λ ∈ {3, 7, 10}
#
# ID scheme:  n_offset + m_offset + λ_offset
#   n:  15 → 2xx,  20 → 3xx,  25 → 4xx
#   m:   3 → x0x,   6 → x1x,  10 → x2x
#   λ:   3 → xx0,   7 → xx1,  10 → xx2
# (dataset_id, n, m, λ, ρ, description)

_N_VALUES = [15, 20, 25]
_M_VALUES = [3, 6, 10]
_L_VALUES = [3, 7, 10]
_N_OFFSETS = {15: 200, 20: 300, 25: 400}
_M_OFFSETS = {3: 0, 6: 10, 10: 20}
_L_OFFSETS = {3: 0, 7: 1, 10: 2}

STRUCTURAL_GRID = []
for _n in _N_VALUES:
    for _m in _M_VALUES:
        for _lam in _L_VALUES:
            _did = _N_OFFSETS[_n] + _M_OFFSETS[_m] + _L_OFFSETS[_lam]
            _tag = " (baseline)" if (_n == 20 and _m == 6 and _lam == 7) else ""
            STRUCTURAL_GRID.append(
                (_did, _n, _m, _lam, DEFAULT_RHO, f"n={_n}, m={_m}, λ={_lam}{_tag}")
            )

# ρ ablation at baseline structure (n=20, m=6, λ=7)
RHO_ABLATION = [
    (500, 20, 6, 7, 0.1, "ρ=0.1 (baseline structure)"),
    # ρ=0.293 is already ID 311 in STRUCTURAL_GRID
    (501, 20, 6, 7, 0.4, "ρ=0.4 (baseline structure)"),
]

ABLATION_CONFIGS = STRUCTURAL_GRID + RHO_ABLATION


# ---------------------------------------------------------------------------
# Batch-parallel labeling (standard tricky task — no parity flip)
# ---------------------------------------------------------------------------
def _check_chain(chain, c_ord, min_chain_length, tolerance):
    """Check if a lagged key chain is in the correct order."""
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


def _label_batch(args):
    """Label a batch of sequences. Worker function for Pool."""
    batch_seqs, c_vocab, lag, tolerance, min_chain_length, sep = args
    outcomes = []
    for seq in batch_seqs:
        tokens = seq.split(sep)
        lagged = check_lag(tokens, c_vocab.keys(), lag)

        is_ordered = False
        if lagged:
            if all(isinstance(item, list) for item in lagged):
                is_ordered = any(
                    _check_chain(chain, c_vocab, min_chain_length, tolerance)
                    for chain in lagged
                )
            else:
                is_ordered = _check_chain(lagged, c_vocab, min_chain_length, tolerance)

        outcomes.append(int(is_ordered))
    return outcomes


def label_sequences_parallel(sequences, c_vocab, lag, tolerance=False,
                             min_chain_length=2, n_workers=None):
    """Parallel labeling optimised for bulk generation. Returns outcome array."""
    if n_workers is None:
        n_workers = cpu_count()

    n_chunks = n_workers * 4
    chunk_size = max(1, len(sequences) // n_chunks)
    chunks = [
        sequences[i:i + chunk_size]
        for i in range(0, len(sequences), chunk_size)
    ]
    args_list = [
        (chunk, c_vocab, lag, tolerance, min_chain_length, SEP)
        for chunk in chunks
    ]

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(_label_batch, args_list),
            total=len(chunks),
            desc="Labeling",
        ))

    return np.array([o for batch in results for o in batch], dtype=np.int32)


def apply_label_noise(df, pr_1=0.7, seed=SEED):
    """
    Apply symmetric label noise post-hoc.

    Each label is flipped independently with probability π = 1 − pr_1.
    Applied AFTER resampling so that ρ_int is controlled by resampling
    and π by this step.
    """
    pi = 1 - pr_1
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(len(df)) < pi
    df = df.copy()
    df.loc[flip_mask, 'Outcome'] = 1 - df.loc[flip_mask, 'Outcome']
    df['Outcome'] = df['Outcome'].astype(int)
    rho_obs = df['Outcome'].mean()
    logger.info(f"Applied label noise π={pi:.2f}: ρ_obs={rho_obs:.4f}")
    return df


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------
def generate_random_sequences(alphabet, seq_length, n_sequences, seed=SEED):
    """Generate n_sequences random letter sequences of given length (with replacement)."""
    rng = np.random.default_rng(seed)
    L = np.array(alphabet)
    indices = rng.integers(0, len(L), size=(n_sequences, seq_length))
    seqs = L[indices]
    return [SEP.join(row) for row in seqs]


# Fixed keys matching dataset 9 (nested: m=3 ⊂ m=6 ⊂ m=10)
FIXED_KEYS = {
    3:  ["W", "Q", "X"],
    6:  ["W", "D", "Q", "J", "X", "N"],
    10: ["W", "D", "Q", "J", "X", "N", "A", "P", "G", "Z"],
}


def generate_random_key(alphabet, key_size, seed=SEED):
    """Return the fixed key for this key_size, matching dataset 9."""
    if key_size in FIXED_KEYS:
        key_letters = FIXED_KEYS[key_size]
    else:
        rng = random.Random(seed)
        key_letters = rng.sample(alphabet, key_size)
    c_vocab = {letter: rank for rank, letter in enumerate(key_letters)}
    return key_letters, c_vocab


def generate_dataset(seq_length, key_size, lag, total_samples=RAW_SAMPLES,
                     seed=SEED, tolerance=False, sequences=None):
    """
    Generate a raw (unresampled) dataset with deterministic labels.

    If *sequences* is provided, reuses them (avoids regeneration when
    multiple configs share the same n).
    """
    key_letters, c_vocab = generate_random_key(ALPHABET, key_size, seed)

    logger.info(f"Parameters: n={seq_length}, m={key_size}, λ={lag}, ℓ={len(ALPHABET)}")
    logger.info(f"Key: {' -> '.join(key_letters)}, tolerance={tolerance}")

    max_chain = (seq_length - 1) // lag + 1 if lag > 0 else seq_length
    max_chain = min(max_chain, key_size)
    logger.info(f"Max possible chain length: {max_chain}")

    if max_chain < 2 and lag > 0:
        logger.warning(f"WARNING: max chain length < 2, no ordered sequences possible! "
                       f"(n={seq_length}, λ={lag})")

    if sequences is None:
        logger.info(f"Generating {total_samples:,} random sequences...")
        sequences = generate_random_sequences(ALPHABET, seq_length, total_samples, seed)

    outcomes = label_sequences_parallel(
        sequences, c_vocab, lag, tolerance=tolerance, min_chain_length=2,
    )

    df = pd.DataFrame({
        'Sequences': sequences,
        'Outcome': outcomes,
    })

    rho = df['Outcome'].mean()
    logger.info(f"Natural ρ_int = {rho:.4f} (ordered={df['Outcome'].sum():,}, "
                f"unordered={len(df) - df['Outcome'].sum():,})")

    return df, key_letters, c_vocab


def split_and_save(df, output_dir, dataset_id, seed=999):
    """Split dataset into train/val/test (80/10/10) and save as CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    X = df[['Sequences']]
    y = df[['Outcome']]

    X_train, X_val_test, y_train, y_val_test = train_test_split(
        X, y, train_size=0.80, random_state=seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test, y_val_test, train_size=0.50, random_state=seed, stratify=y_val_test
    )

    for data, name in [
        (X_train, f"X_train_{dataset_id}"),
        (X_val, f"X_val_{dataset_id}"),
        (X_test, f"X_test_{dataset_id}"),
        (y_train, f"y_train_{dataset_id}"),
        (y_val, f"y_val_{dataset_id}"),
        (y_test, f"y_test_{dataset_id}"),
    ]:
        path = os.path.join(output_dir, f"{name}.csv")
        data.to_csv(path, index=False)

    logger.info(f"Saved: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
                f"→ {output_dir}/*_{dataset_id}.csv")


def print_config_list():
    """Pretty-print all planned configurations."""
    print("\n=== Sensitivity Analysis: Design A ===")
    print(f"Raw pool: {RAW_SAMPLES:,} → resampled to {FINAL_SAMPLES:,} at target ρ\n")

    print("All configurations:")
    print(f"{'ID':<6} {'n':<5} {'m':<5} {'λ':<5} {'ρ':<8} {'Description'}")
    print("-" * 65)
    for did, n, m, lag, rho, desc in ABLATION_CONFIGS:
        print(f"{did:<6} {n:<5} {m:<5} {lag:<5} {rho:<8.3f} {desc}")

    print(f"\n--- Structural grid: {len(STRUCTURAL_GRID)} configs (all at ρ={DEFAULT_RHO}) ---")
    for n_val in _N_VALUES:
        print(f"\n  n={n_val}:")
        header = f"{'':>8}" + "".join(f"{'λ='+str(lam):<8}" for lam in _L_VALUES)
        print(f"  {header}")
        for m_val in _M_VALUES:
            did_row = []
            for lam_val in _L_VALUES:
                did = _N_OFFSETS[n_val] + _M_OFFSETS[m_val] + _L_OFFSETS[lam_val]
                marker = "*" if (n_val == 20 and m_val == 6 and lam_val == 7) else ""
                did_row.append(f"{did}{marker}")
            row = f"{'m='+str(m_val):>8}" + "".join(f"{d:<8}" for d in did_row)
            print(f"  {row}")

    print("\n--- ρ ablation (n=20, m=6, λ=7): 3 configs ---")
    print("  ρ=0.1   → ID 500")
    print("  ρ=0.293 → ID 311 (in structural grid)")
    print("  ρ=0.4   → ID 501")
    print("\n  * = baseline")
    print(f"\nTotal: {len(ABLATION_CONFIGS)} configurations\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis datasets (Design A: "
                    "structural grid + ρ ablation, all resampled to fixed ρ)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str, default="data/simulation/sensitivity/")
    parser.add_argument("--raw_samples", type=int, default=RAW_SAMPLES,
                        help=f"Raw pool size before resampling (default: {RAW_SAMPLES:,})")
    parser.add_argument("--final_samples", type=int, default=FINAL_SAMPLES,
                        help=f"Final dataset size after resampling (default: {FINAL_SAMPLES:,})")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tolerance", action="store_true",
                        help="Allow one ordering violation (default: False for deterministic)")
    parser.add_argument("--rnd", action="store_true",
                        help="Apply symmetric label noise post-hoc (π = 1 − pr_1)")
    parser.add_argument("--pr_1", type=float, default=0.7,
                        help="P(keep label) under noise; π = 1 − pr_1 (default: 0.7 → π=0.3)")

    # Single config mode
    parser.add_argument("--dataset_id", type=int, default=None,
                        help="Generate a single dataset with this ID")
    parser.add_argument("--seq_length", type=int, default=20)
    parser.add_argument("--key_size", type=int, default=6)
    parser.add_argument("--lag", type=int, default=7)
    parser.add_argument("--target_rho", type=float, default=DEFAULT_RHO,
                        help=f"Target class balance for resampling (default: {DEFAULT_RHO})")

    # Utility
    parser.add_argument("--list", action="store_true",
                        help="List all planned configurations and exit")
    parser.add_argument("--all", action="store_true",
                        help="Generate all configurations")
    parser.add_argument("--only", type=str, default=None,
                        help="Generate only these IDs (comma-separated, e.g. '311,500,501')")

    args = parser.parse_args()

    if args.list:
        print_config_list()
        return

    if args.all or args.only:
        if args.only:
            only_ids = set(int(x) for x in args.only.split(','))
            configs = [c for c in ABLATION_CONFIGS if c[0] in only_ids]
            if not configs:
                parser.error(f"No configs found for IDs: {only_ids}")
        else:
            configs = ABLATION_CONFIGS

        # Group configs by n to share generated sequences
        configs_by_n = defaultdict(list)
        for config in configs:
            configs_by_n[config[1]].append(config)

        skipped = []
        for n_val in sorted(configs_by_n):
            n_configs = configs_by_n[n_val]

            logger.info(f"\n{'#'*60}")
            logger.info(f"Generating {args.raw_samples:,} sequences for n={n_val} "
                        f"(shared by {len(n_configs)} configs)")
            logger.info(f"{'#'*60}")
            sequences = generate_random_sequences(ALPHABET, n_val, args.raw_samples, args.seed)

            for did, _, m, lag, rho, desc in n_configs:
                logger.info(f"\n{'='*60}")
                logger.info(f"Config {did}: {desc} (target ρ={rho})")
                logger.info(f"{'='*60}")

                # Step 1: label with this (m, λ) — reusing shared sequences
                df_raw, key, c_vocab = generate_dataset(
                    n_val, m, lag, args.raw_samples, args.seed,
                    tolerance=args.tolerance, sequences=sequences,
                )

                natural_rho = df_raw['Outcome'].mean()
                n_pos = int(df_raw['Outcome'].sum())
                n_neg = len(df_raw) - n_pos
                need_pos = int(args.final_samples * rho)
                need_neg = args.final_samples - need_pos

                logger.info(f"Natural ρ_int={natural_rho:.4f} | Pool: {n_pos:,} pos, {n_neg:,} neg | "
                            f"Need: {need_pos:,} pos, {need_neg:,} neg")

                # Step 2: resample to target ρ_int
                try:
                    df = sample_with_proportion(df_raw, n=args.final_samples, pi=rho)
                except ValueError as e:
                    logger.error(f"SKIPPING config {did}: {e}")
                    skipped.append((did, desc, str(e)))
                    continue

                logger.info(f"Resampled: {len(df):,} samples, ρ_int={df['Outcome'].mean():.4f}")

                # Step 3: apply noise post-hoc if requested
                if args.rnd:
                    df = apply_label_noise(df, pr_1=args.pr_1, seed=args.seed)

                split_and_save(df, args.output_dir, did)

        # Save config metadata (only successfully generated configs)
        generated_ids = {did for did, *_ in configs} - {s[0] for s in skipped}
        meta = pd.DataFrame(
            [(did, n, m, lag, rho, desc) for did, n, m, lag, rho, desc in configs
             if did in generated_ids],
            columns=['dataset_id', 'n', 'm', 'lag', 'rho', 'description']
        )
        meta_path = os.path.join(args.output_dir, "sensitivity_configs.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        meta.to_csv(meta_path, index=False)
        logger.info(f"\nConfig metadata saved to {meta_path}")

        if skipped:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"SKIPPED {len(skipped)} configs (insufficient samples):")
            for did, desc, reason in skipped:
                logger.warning(f"  {did}: {desc} — {reason}")
            logger.warning(f"Consider increasing --raw_samples beyond {args.raw_samples:,}")
        return

    # Single config mode
    if args.dataset_id is None:
        parser.error("Specify --dataset_id for single config, or use --all or --list")

    df_raw, key, c_vocab = generate_dataset(
        args.seq_length, args.key_size, args.lag,
        args.raw_samples, args.seed, tolerance=args.tolerance,
    )

    natural_rho = df_raw['Outcome'].mean()
    logger.info(f"Natural ρ_int={natural_rho:.4f}, target ρ_int={args.target_rho}")

    df = sample_with_proportion(df_raw, n=args.final_samples, pi=args.target_rho)
    logger.info(f"Resampled: {len(df):,} samples, ρ_int={df['Outcome'].mean():.4f}")

    if args.rnd:
        df = apply_label_noise(df, pr_1=args.pr_1, seed=args.seed)

    split_and_save(df, args.output_dir, args.dataset_id)


if __name__ == "__main__":
    main()
