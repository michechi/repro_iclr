import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections

from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count
from collections import Counter

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
    # Import inside worker if needed (like clusterEvalQ)
    # import pandas as pd  
    return generate_sequences(letters, n, m_chunk, replacement, seed)

def assign_outcome(seq:str, c_ord:dict, rnd:bool=False, sep:str="\x1f", already_splitted=False) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = [x for x in seq.split(sep) if x in l_keys] # avoiding noising letters
    else: 
        test_seq_splt = [x for x in seq if x in l_keys]
    
    tolerance=True # For the cyclic ordering
    for x,y in zip(test_seq_splt[:-1], test_seq_splt[1:]):
        if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
                if tolerance:
                    tolerance=False
                else:
                    return 0
    # if we exit for cycle then it's all ordered
    return 1


def assign_outcome_numeric(seq:str, c_ord:dict, rnd:bool=False, sep:str="\x1f", already_splitted=False) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    l_keys = c_ord.keys()

    if not already_splitted:
        test_seq_splt = [x for x in seq.split(sep) if x in l_keys] # avoiding noising letters
    else: 
        test_seq_splt = [x for x in seq if x in l_keys]
    
    # tolerance=False # For the cyclic ordering
    if sum(list(map(c_ord.get, test_seq_splt))) <= 9:
        return 1
    else:
        return 0

random.seed(959693)

n_events = 5
n_seq = 1_000_000
k =4
n_0s = 1_000_000
n_1s = 1_000_000
n_tot = n_0s + n_1s
n_train = n_tot * 0.80 # 80% of n_tot
n_val = n_tot * 0.10
n_test = n_tot * 0.10

#sum([n_train, n_test, n_val]) == n_tot

letters = list(string.ascii_uppercase)

# letters_4_key = letters.copy()
# random.shuffle(letters_4_key)
# letters_4_key = ["W", "D", "Q", "J", "U", "H"]
# digits = list(map(str, range(10)))

# random.shuffle(letters_4_key)
# random.shuffle(digits)

# c_vocab = {w:p for p,w in enumerate(letters_4_key,start=0)}

# # My numeric test
# c_vocab = {
# 'C': 2, 
# 'Q': 1,
# 'O': 0,
# 'H': 2,
# 'Y': 1,
# 'X': 0,
# 'S': 2,
# 'U': 1,
# 'N': 0,
# 'J': 2,
# 'F': 1,
# 'E': 0,
# 'W': 2,
# 'Z': 1,
# 'B': 0,
# 'G': 2,
# 'L': 1,
# 'A': 0,
# 'K': 2,
# 'I': 1,
# 'V': 0,
# 'P': 2,
# 'T': 1,
# 'D': 0,
# 'M': 2,
# 'R': 1}

# Test double alphabet
start_n_stop = len(letters)//2
train_letters = letters[:start_n_stop]
test_letters = letters[start_n_stop:]

key_order_train=random.sample(range(0, start_n_stop), k=start_n_stop)
key_order_test=random.sample(range(0, start_n_stop), k=start_n_stop)

# Two different ordering
train_c_vocab, test_c_vocab = dict(zip(train_letters, key_order_train)), dict(zip(test_letters, key_order_test))


# # This is sequential
train_sequences = generate_sequences(letters=train_letters, n=n_events, m=n_seq, replacement=True)
test_sequences = generate_sequences(letters=test_letters, n=n_events, m=n_seq, replacement=True)

# Parallel version
# n_cores = cpu_count()-1
# rng = np.random.default_rng(999)
# seeds = rng.integers(0, 2**31, size=n_cores)
# m_per_core = int(n_seq * 1.2 / n_cores)  # 20% oversample
# tasks = [(letters, n_events, m_per_core, True, seed) 
#              for seed in seeds]

# with Pool(processes=n_cores) as pool:
#     results = pool.map(worker_generate, tasks)

# # Deduplicate and trim
# all_sequences = [seq for result in results for seq in result]
# sequences = list(dict.fromkeys(all_sequences))[:n_seq]

train_set_seq = set(train_sequences)
test_set_seq = set(test_sequences)

n_train_set_seq = len(train_set_seq)
n_test_set_seq = len(test_set_seq)

if n_seq != n_train_set_seq:
    print(f"Attention! There are {n_seq-n_train_set_seq} trainig duplicates!\nRemoving them..")
    train_sequences = train_set_seq.copy()
    n_train_seq = n_train_set_seq
    del train_set_seq, n_train_set_seq
    print("Done!")

if n_seq != n_test_set_seq:
    print(f"Attention! There are {n_seq-n_test_set_seq} trainig duplicates!\nRemoving them..")
    test_sequences = test_set_seq.copy()
    n_test_seq = n_test_set_seq
    del test_set_seq, n_test_set_seq
    print("Done!")

# check for my training set:
check_training=list(map(lambda x: assign_outcome(x, train_c_vocab), train_sequences))

if sum(check_training) > 0:
    # There are some valid sequences
    train_which_1s = list(filter(lambda x: assign_outcome(x, train_c_vocab, already_splitted=False)==1, train_sequences))
    train_n_1s = len(train_which_1s)
    print(f"There are {train_n_1s} ordered sequences! ({train_n_1s/len(train_sequences)*100:.2f}%)")

train_which_0s = list(filter(lambda x: assign_outcome(x, train_c_vocab, already_splitted=False)==0, train_sequences))
train_n_0s = len(train_which_0s)
if (train_n_0s + train_n_1s) == len(train_sequences):
    print("All good!")
else:
    print("Figures do not add up!")

# check for my test set:
check_test=list(map(lambda x: assign_outcome(x, test_c_vocab), test_sequences))

if sum(check_test) > 0:
    # There are some valid sequences
    test_which_1s = list(filter(lambda x: assign_outcome(x, test_c_vocab, already_splitted=False)==1, test_sequences))
    test_n_1s = len(test_which_1s)
    print(f"There are {test_n_1s} ordered sequences! ({test_n_1s/len(test_sequences)*100:.2f}%)")

test_which_0s = list(filter(lambda x: assign_outcome(x, test_c_vocab, already_splitted=False)==0, test_sequences))
test_n_0s = len(test_which_0s)
if (test_n_0s + test_n_1s) == len(test_sequences):
    print("All good!")
else:
    print("Figures do not add up!")

# Before moving to the cyclic-ordering
# l_all_valid_seq = list(itertools.combinations(letters, n_events)) # All possible valid letters combinations
# d_all_valid_seq = list(itertools.combinations(digits, n_events)) # Not really important in this case since for now we don't use
#                                                                  # repeated letters (-> numbers are not important then)

# all_valid_seq = [np.char.add(x,y).tolist() for x in l_all_valid_seq for y in  d_all_valid_seq]
# random.shuffle(all_valid_seq)

# # Double Check if they are all valid
# check_valid=list(map(lambda x: assign_outcome(x, already_splitted=True), all_valid_seq))
# sum(check_valid) == len(check_valid) # True, all 1!

# all_valid_seq_str = ["\x1f".join(x) for x in all_valid_seq]

def extract_characters(seq:str, sep:str="\x1f") -> str:
    seq_char = "-".join([x[0] for x in seq.split(sep)])
    return(seq_char)

train_chr_seq_1s=list(map(extract_characters, train_which_1s))
train_chr_seq_0s=list(map(extract_characters, train_which_0s))

i = 0
train_stats_1s, train_stats_0s = collections.Counter(x[i*2] for x in train_chr_seq_1s), collections.Counter(x[i*2] for x in train_chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [train_stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [train_stats_0s.get(letter, 0) for letter in letters_ord]

# Test
test_chr_seq_1s=list(map(extract_characters, test_which_1s))
test_chr_seq_0s=list(map(extract_characters, test_which_0s))

i = 0
test_stats_1s, test_stats_0s = collections.Counter(x[i*2] for x in test_chr_seq_1s), collections.Counter(x[i*2] for x in test_chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [test_stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [test_stats_0s.get(letter, 0) for letter in letters_ord]


# Side-by-side plot
x = np.arange(len(letters_ord))
width = 0.35  # Bar width

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, counts_1s, width, label='Label=1', alpha=0.8)
ax.bar(x + width/2, counts_0s, width, label='Label=0', alpha=0.8)
ax.set_xlabel('Letters')
ax.set_ylabel('Count')
ax.set_title(f'{i+1}st/nd Letter Frequency by Label')
ax.set_xticks(x)
ax.set_xticklabels(letters_ord)
ax.legend()
plt.show()


# Create Training Dataset
df_train = pd.DataFrame({"Sequences":list(train_sequences)})
df_train["Outcome"] = list(map(lambda x: assign_outcome(x, train_c_vocab), df_train.Sequences))

df_test = pd.DataFrame({"Sequences":list(test_sequences)})
df_test["Outcome"] = list(map(lambda x: assign_outcome(x, test_c_vocab), df_test.Sequences))



# Splitting Train Val Test
# Now train_val_test split:
X_train_val, y_train_val = df_train["Sequences"], df_train["Outcome"]
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=0.90, random_state=999)
X_test, y_test = df_test["Sequences"], df_test["Outcome"]

# Let's visualize bigrams
def plot_bigram_distribution(X_train, y_train, top_n=30):
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
    top_bigrams = sorted(bigram_diff.items(), key=lambda x: x[1][2], reverse=True)[:top_n]
    
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
plot_bigram_distribution(X_train, y_train, top_n=1000)

plot_bigram_distribution(X_test, y_test, top_n=1000)


# Uncomment just if you want to save data!
for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], ["X_train_4", "X_val_4", "X_test_4", "y_train_4", "y_val_4", "y_test_4"]):
    df.to_csv(f"data/simulation/{name}.csv", index=False)
