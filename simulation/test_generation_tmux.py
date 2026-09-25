import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections
import sys

from joblib import dump
from pathlib import Path
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from collections import Counter
from scipy.stats import bernoulli
from itertools import product
from typing import List, Union
from tqdm import tqdm

# # Add the MIMICIV directory to the path FOR DEBUG ONLY
# root_dir = Path(__file__).parent.parent  # Goes up two levels
# sys.path.insert(0, str(root_dir))
from simulation.do_check_lag import check_lag
from simulation.do_strategy import do_strategy, do_chek_order, do_order

# For multi-ordering key test
from simulation.do_multiple_key_ordering import do_keys, do_lags, do_multiple_key_ordering

def rm_all():
    [globals().pop(var) for var in list(globals()) if not var.startswith('_')]

def generate_sequences(
    letters:list, n:int=10, m:int=10_000, replacement:bool=False, seed:int=None, batch_size:int=None, duplicates:bool=False
):
    """
    Generate exactly m unique sequences (rows) of length n where each event is 'Letter' + 'Digit' (A1, Z0, H4, ...).
    - If replacement=False, each *row* has no repeated letters and no repeated digits.
    - Across rows, duplicates are removed; generation continues until m unique rows are collected.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(letters, dtype='<U8')
    Llen = len(L)

    if not replacement and n > Llen:
        raise ValueError("n must be <= len(letters) when replacement=False")

    # Choose a sensible batch size (oversample a bit to reduce iterations)
    if batch_size is None:
        batch_size = min(10_000, m // 10)

    # Store unique rows via a hash set of compact string keys
    # Use a separator that won't appear in tokens (ASCII Unit Separator)
    SEP = "\x1f"
    seen = set()
    out_rows = []

    def _gen_batch(k):
        """k batch dimension"""
        if replacement:
            print("replacement!\n")
            li = rng.integers(0, Llen, size=(k, n))
        else:
            print("no replacement!\n")
            # Per-row permutations (no repeats within a row)
            l_scores = rng.random((k, Llen))
            li = np.argsort(l_scores, axis=1)[:, :n]
        seq = L[li]  # shape (k, n), dtype '<U...'
        return seq

    while len(out_rows) < m:
        need = m - len(out_rows)
        k = max(batch_size, need)  # at least batch_size to amortize costs
        seq = _gen_batch(k)        # (k, n)

        # Build vectorized keys for dedup: "A|B|..."
        keys = [SEP.join(key) for key in seq.tolist()]
        # Inefficient! (?)
        # for j in range(1, n):
        #     keys = np.char.add(np.char.add(keys, SEP), seq[:, j])

        # Filter rows not seen before
        mask_new = np.fromiter((key not in seen for key in keys), count=k, dtype=bool)
        if not mask_new.any():
            continue

        # new_keys = keys[mask_new] # Now keys it's a list not an array so this won't work
        new_keys = [key for key,to_keep in zip(keys, mask_new) if to_keep]

        out_rows+=new_keys
        for key in new_keys:
            seen.add(key)

    return out_rows

# For parallelization
def worker_generate(args):
    letters, n, m_chunk, replacement, seed = args 
    return generate_sequences(letters, n, m_chunk, replacement, seed)

def assign_outcome_positional(
    seq:str, 
    c_ord:dict, 
    lags:Union[int, List], 
    rnd:bool=False, 
    sep:str="\x1f", 
    already_splitted=False, 
    pr_1 = 0.9, 
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

def assign_outcome_positional_2_steps(
    seq:str, 
    c_ord:dict, # {1:{A:0, B:1, ...}, 2:{Z:0, T:2, ...}, 3:{F:0, G:1, ...}}
    lags:Union[int, List], 
    rnd:bool=False, 
    sep:str="\x1f", 
    already_splitted=False, 
    pr_1 = 0.9, 
    tolerance=True,
    debugging=False
    ) -> dict:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    This is a two steps implementation (in two subsequences).
    If `debugging==True` gives extra values to check validity of code  
    TODO/LIST:
        * Better management of lags values (different strategies)
        * Better management of hardcoded values
        * Debugging values
        * Checkpoint loadings (to have the possibility)
        * .. 
    """
    
    if len(lags) != 4:
        raise ValueError("Need exactly 4 lags")
    
    lag_1, lag_2, lag_3, lag_4 = lags

    required_keys = ['strategy', 'first_order', 'second_order']
    if not all(k in c_ord for k in required_keys):
        raise ValueError(f"c_ord must contain keys: {required_keys}")

    keys = dict(
        strategy=c_ord['strategy'],
        both_orders=[c_ord['first_order'], c_ord['second_order']],
        first_order= c_ord['first_order'], 
        second_order=c_ord['second_order'],
        no_order=None
        )
    
    test_seq_splt = seq if already_splitted else seq.split(sep) 
    
    # Making subsequences:
    midpoint = len(test_seq_splt)//2
    test_seq_splt_1 = test_seq_splt[:midpoint]
    test_seq_splt_2 = test_seq_splt[midpoint:]

    # Determine the strategy looking at the first subsequence
    info_2_debug_1, strategy = do_strategy(
        test_seq_splt_1, 
        [lag_1, lag_2], 
        keys["strategy"], 
        debugging)
    

    # Now, check the order looking at the second subsequence given the strategy
    info_2_debug_2, order = do_order(
        test_seq_splt_2,
        lag_3, 
        keys[strategy], 
        strategy, 
        debugging
        )
    
    # Compute the probability to simulate
    pr_to_simulate = pr_1 if order else 1-pr_1
    

    if rnd:
        # Stochastics outcome
        outcome = bernoulli.rvs(pr_to_simulate)
    else:
        # Deterministic outcome
        outcome = 1 if pr_to_simulate==pr_1 else 0
    
    # Build conistent return dict
    results_2_debug = {
        'outcome': outcome,
        'seq': seq,
        'pr_2_sim': pr_to_simulate
    }

    if debugging:
        results_2_debug.update({
            'lagged_keys': [
                info_2_debug_1['lagged_1'], 
                info_2_debug_1['lagged_2'], 
                info_2_debug_2['lagged_3']
            ],
            'all_ordered': [
                info_2_debug_1['order'], 
                info_2_debug_2['final_order']
            ],
            'lags': lags
        })
        
    return(results_2_debug)
    
# seq4 = ['A', 'A', 'C', 'D', 'B', 'A', 'Z', 'H', 'C']
# # seqT = ['M','V','D','V','M','T','L','C','C','G','X','J','C','Y','J','B','C','Q','F','M']
# keys = ['A', 'B', 'C']
# c_vocab4 = {w:p for p,w in enumerate(keys,start=0)}
# # check_lag(seqT, ["W", "D", "Q", "J", "U"], 7)
# assign_outcome_positional(seq4, c_vocab4, lags=3, already_splitted=True)

random.seed(959693)

n_events = 40 # More
n_seq = 50_000_000
n_0s = 100_000
n_1s = 100_000
n_tot = n_0s + n_1s
n_train = n_tot * 0.80 # 80% of n_tot
n_val = n_tot * 0.10
n_test = n_tot * 0.10
generate = True
parallel = True
debugging=False
rnd=True # randomic outcome?
# pr_1=0.9 # If so, which is P(Y=1|X=Ordered)?
number_csv=5 # If not generate, which csv has to be uploaded?
#sum([n_train, n_test, n_val]) == n_tot

letters = list(string.ascii_uppercase)
# letters_4_key = letters.copy()
# random.shuffle(letters_4_key)
letters_4_key = ["W", "D", "Q", "J", "X", "N"] # Added X Keep it fixed!!
# letters_4_key = ["W", "D", "Q", "J", "U"] # TEST REPLICA CURRENT CSV 9
c_vocab = {w:p for p,w in enumerate(letters_4_key,start=0)}
# Letter used for the test on two steps (10 rnd, 11nonrnd)
# set_1_key = ["W", "D", "Q", "J", "X", "U"]
# set_2_key = ["M", "A", "L", "J", "V", "I"]
# set_3_key = ["Q", "O", "Y", "D", "T", "S"]

# digits = list(map(str, range(10)))

# random.shuffle(letters_4_key)
# random.shuffle(digits)

# c_vocab = dict(
#     strategy={w:p for p,w in enumerate(set_1_key,start=0)},
#     first_order={w:p for p,w in enumerate(set_2_key,start=0)},
#     second_order={w:p for p,w in enumerate(set_3_key,start=0)}
# )

# c_vocab = do_keys(letters, 1234)

# This is sequential
if generate:
    if not parallel:
        sequences = generate_sequences(letters=letters, n=n_events, m=n_seq, replacement=True)
    else:
        # Parallel version
        n_cores = cpu_count()-1
        rng = np.random.default_rng(999)
        seeds = rng.integers(0, 2**31, size=n_cores)
        m_per_core = int(n_seq * 1.2 / n_cores)  # 20% oversample
        tasks = [(letters, n_events, m_per_core, True, seed) for seed in seeds]
        with Pool(processes=n_cores) as pool:
            results = pool.map(worker_generate, tasks)
        # Deduplicate and trim
        all_sequences = [seq for result in results for seq in result]
        sequences = list(dict.fromkeys(all_sequences))[:n_seq]
else:
    # Load pre-generated sequences (from previous runs)
    sequences = pd.read_csv(f"data/simulation/X_test_{number_csv}.csv")["Sequences"].tolist()
    labels = pd.read_csv(f"data/simulation/y_test_{number_csv}.csv")["Outcome"].tolist()

    sequences += pd.read_csv(f"data/simulation/X_train_{number_csv}.csv")["Sequences"].tolist()
    labels += pd.read_csv(f"data/simulation/y_train_{number_csv}.csv")["Outcome"].tolist()

    sequences += pd.read_csv(f"data/simulation/X_val_{number_csv}.csv")["Sequences"].tolist()
    labels += pd.read_csv(f"data/simulation/y_val_{number_csv}.csv")["Outcome"].tolist()

n_seq = len(sequences)
set_seq = set(sequences)
n_set_seq = len(set_seq)

if n_seq != n_set_seq:
    print(f"Attention! There are {n_seq-n_set_seq} duplicates!\nRemoving them..")
    sequences = set_seq.copy()
    n_seq = n_set_seq
    del set_seq, n_set_seq
    print("Done!")

def efficient_check_v1(sequences, c_vocab, assign_outcome_positional, tolerance=True, lags=7, rnd=False, debugging=False):
    """Compute once and then reuse the results"""
    # Compute ONCE for each sequence
    print(f"Tolerance: {tolerance}, lags: {lags}!\n")
    outcomes = []

    if not debugging:
        df_2_monitor = pd.DataFrame({
            'outcome':[],
            'seq':[],
            'pr_2_sim':[],
            # 'lagged_keys':[],
            # 'all_ordered':[]
        })
    else:
        df_2_monitor = pd.DataFrame({
            'outcome':[],
            'seq':[],
            'pr_2_sim':[],
            'lagged_keys':[],
            'all_ordered':[],
            'lags':[]
        })

    for seq in tqdm(sequences):
        df_results = pd.DataFrame(assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=lags, tolerance=tolerance, rnd=rnd))
        df_2_monitor = pd.concat([df_2_monitor, df_results])
        # outcomes += assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=lags, tolerance=tolerance)
    # outcomes = [assign_outcome_positional(seq, c_vocab, already_splitted=False, lags=8) for seq in sequences]
    
    sequences, outcomes = df_2_monitor["seq"], df_2_monitor["outcome"]
    
    # Now use zip to pair sequences with their outcomes
    sequences_with_outcomes = list(zip(sequences, outcomes))
    
    # Filter based on the results already computed
    which_1s = [seq for seq, outcome in sequences_with_outcomes if outcome == 1]
    which_0s = [seq for seq, outcome in sequences_with_outcomes if outcome == 0]
    
    n_1s = len(which_1s)
    n_0s = len(which_0s)
    
    if n_1s > 0:
        print(f"There are {n_1s} ordered sequences! ({n_1s/len(sequences)*100:.2f}%)")
        
        if (n_0s + n_1s) == len(sequences):
            print("All good!")
        else:
            print("Figures do not add up!")
    else:
        print("No valid sequences found")
    
    return ({
        'valid_sequences': which_1s,
        'invalid_sequences': which_0s,
        'outcomes': outcomes
    }, df_2_monitor)

def process_single_sequence(args):
    """Helper function for multiprocessing"""
    seq, c_vocab, assign_outcome_positional, lags, tolerance, rnd, debugging, min_chain_length = args
    result = assign_outcome_positional(
        seq, c_vocab, 
        already_splitted=False, 
        lags=lags, 
        tolerance=tolerance, 
        rnd=rnd,
        debugging=debugging,
        min_chain_length=min_chain_length
    )
    # Convert the dictionary to a DataFrame
    return pd.DataFrame(result)

def efficient_check_parallel(sequences, c_vocab, assign_outcome_positional, 
                             tolerance=True, lags=7, rnd=False, debugging=debugging, min_chain_length=3, n_workers=None):
    """Parallelized version using multiprocessing.Pool"""
    print(f"Tolerance: {tolerance}, lags: {lags}!\n")
    
    # Use all cores-1 if not specified
    if n_workers is None:
        n_workers = cpu_count()
    
    # Prepare the arguments for each sequence
    args_list = [
        (seq, c_vocab, assign_outcome_positional, lags, tolerance, rnd, debugging, min_chain_length) 
        for seq in sequences
    ]
    
    # Parallelize with Pool
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_sequence, args_list), 
            total=len(sequences)
        ))
    
    # Combine all DF
    df_2_monitor = pd.concat(results, ignore_index=True)
    
    sequences, outcomes = df_2_monitor["seq"], df_2_monitor["outcome"]
    sequences_with_outcomes = list(zip(sequences, outcomes))
    
    which_1s = [seq for seq, outcome in sequences_with_outcomes if outcome == 1]
    which_0s = [seq for seq, outcome in sequences_with_outcomes if outcome == 0]
    
    n_1s = len(which_1s)
    n_0s = len(which_0s)
    
    if n_1s > 0:
        print(f"There are {n_1s} ordered sequences! ({n_1s/len(sequences)*100:.2f}%)")
        if (n_0s + n_1s) == len(sequences):
            print("All good!")
        else:
            print("Figures do not add up!")
    else:
        print("No valid sequences found")
    
    return ({
        'valid_sequences': which_1s,
        'invalid_sequences': which_0s,
        'outcomes': outcomes
    }, df_2_monitor)

# Which lags do we prefer?
# lags = [9,8,7,6] # csv 2
# lags = [7] # csv 1
# lags = [4, 3, 2] # csv 0
lags = 7

# Testing for multiple key ordering
# lags = do_lags(letters, 1234)
function_2_use_4_outcome = assign_outcome_positional

if parallel:
    # Parallel:
    n_cores = cpu_count()-1
    
    check, df_2_monitor = efficient_check_parallel(
            sequences=sequences,
            c_vocab=c_vocab,
            assign_outcome_positional=function_2_use_4_outcome,
            tolerance=False,
            lags=lags,
            rnd=rnd,              #### >>>>>>>>> CHANGE HERE PARAMETER !!!
            debugging=debugging,
            min_chain_length=5,   #### >>>>>>>>> CHANGE HERE PARAMETER !!!
            n_workers=n_cores  # or None to use all cores
        )
else:
    # Sequential:
    check, df_2_monitor = efficient_check_v1(
        sequences=sequences, 
        c_vocab=c_vocab, 
        assign_outcome_positional=function_2_use_4_outcome, 
        tolerance=False, 
        lags=lags, 
        rnd=rnd,
        debugging=debugging)


dump(
    {
        "check": check,
        "df_2_monitor": df_2_monitor,
        "sequences": sequences,
    },
    "simulation_results_rnd_09.joblib",
    compress=3
)