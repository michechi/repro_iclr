import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import random
import collections

from sklearn.model_selection import train_test_split
# import re

def rm_all():
    [globals().pop(var) for var in list(globals()) if not var.startswith('_')]

random.seed(999)

n_events = 5
n_seq = 100_000_000

n_0s = 200_000
n_1s = 100_000
n_tot = n_0s + n_1s
n_train = n_tot * 0.80 # 80% of n_tot
n_val = n_tot * 0.10
n_test = n_tot * 0.10

#sum([n_train, n_test, n_val]) == n_tot

letters = list(string.ascii_uppercase)
digits = list(map(str, range(10)))

random.shuffle(letters)
random.shuffle(digits)

def generate_sequences(
    letters:list, digits:list, n:int=10, m:int=10_000, replacement:bool=False, seed:int=None, batch_size:int=None, duplicates:bool=False
):
    """
    Generate exactly m unique sequences (rows) of length n where each event is 'Letter' + 'Digit' (A1, Z0, H4, ...).
    - If replacement=False, each *row* has no repeated letters and no repeated digits.
    - Across rows, duplicates are removed; generation continues until m unique rows are collected.
    """
    rng = np.random.default_rng(seed)
    L = np.asarray(letters, dtype='<U8')
    D = np.asarray(digits,  dtype='<U8')
    Llen, Dlen = len(L), len(D)

    if not replacement and (n > Llen or n > Dlen):
        raise ValueError("n must be <= len(letters) and len(digits) when replacement=False")

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
            di = rng.integers(0, Dlen, size=(k, n))
        else:
            print("no replacement!\n")
            # Per-row permutations (no repeats within a row)
            l_scores = rng.random((k, Llen))
            d_scores = rng.random((k, Dlen))
            li = np.argsort(l_scores, axis=1)[:, :n]
            di = np.argsort(d_scores, axis=1)[:, :n]
        seq = L[li] + D[di]  # shape (k, n), dtype '<U...'
        return seq

    while len(out_rows) < m:
        need = m - len(out_rows)
        k = max(batch_size, need)  # at least batch_size to amortize costs
        seq = _gen_batch(k)        # (k, n)

        # Build vectorized keys for dedup: "A1|B2|..."
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

c_vocab = {w:p for p,w in enumerate(letters,start=1)}
d_vocab = {d:p for p,d in enumerate(digits,start=1)}

def assign_outcome(seq:str, rnd:bool=False, c_ord:dict=c_vocab, d_ord:dict=d_vocab, sep:str="\x1f", already_splitted=False) -> int:
    """
    function that, given a sequence, says 1 or 0, depending on ordering.
    It allows for just one cycle.
    """
    # test_seq = 'D7\x1fH5\x1fA7\x1fR5\x1fL1\x1fE4\x1fF8\x1fC0\x1fA8\x1fN0' # sequences[0]
    # test_seq_splt = test_seq.split("\x1f")
    if not already_splitted:
        test_seq_splt = seq.split(sep)
    else: 
        test_seq_splt = seq
    
    still_cycle_1 = True
    for x,y in zip(test_seq_splt[:-1], test_seq_splt[1:]):
        if ((2*c_ord[x[0]])>(2*c_ord[y[0]])):
            if still_cycle_1:
                still_cycle_1 = False
            else:
                return 0
        elif x[0]==y[0]:
            if(2*d_ord[str(x[1])])>(2*d_ord[str(y[1])]):
                if still_cycle_1:
                    still_cycle_1 = False
                else:
                    return 0
    # if we exit for cycle then it's all ordered
    return 1

sequences = generate_sequences(letters=letters, digits=digits, n=n_events, m=n_seq, replacement=False)

set_seq = set(sequences)
n_set_seq = len(set_seq)

if n_seq != n_set_seq:
    print(f"Attention! There are {n_seq-n_set_seq} duplicates!\nRemoving them..")
    sequences = set_seq.copy()
    n_seq = n_set_seq
    del set_seq, n_set_seq
    print("Done!")

check=list(map(lambda x: assign_outcome(x), sequences))

if sum(check) > 0:
    # There are some valid sequences
    which_1s = list(filter(lambda x: assign_outcome(x, already_splitted=False)==1, sequences))
    n_1s = len(which_1s)
    print(f"There are {n_1s} ordered sequences! ({n_1s/len(sequences)*100:.2f}%)")

which_0s = list(filter(lambda x: assign_outcome(x, already_splitted=False)==0, sequences))
n_0s = len(which_0s)
if (n_0s + n_1s) == len(sequences):
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

chr_seq_1s=list(map(extract_characters, which_1s))
chr_seq_0s=list(map(extract_characters, which_0s))

i = 0
stats_1s, stats_0s = collections.Counter(x[i*2] for x in chr_seq_1s), collections.Counter(x[i*2] for x in chr_seq_0s)

letters_ord = string.ascii_uppercase
counts_1s = [stats_1s.get(letter, 0) for letter in letters_ord]
counts_0s = [stats_0s.get(letter, 0) for letter in letters_ord]

# Plot affiancato
x = np.arange(len(letters_ord))
width = 0.35  # Larghezza barre

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


# # Test debug why we cannot find valid sequences in sequences in all_valid_seq_str
# # I think in all_valid_seq_str we have always a correct number combo which in our 
# # setup shouldn't be always the case (if we have a correct alphabet placement)
# p=re.compile(r'^B\d\x1fM\d\x1fU\d\x1fY\d\x1fQ\d\x1fD\d\x1fT\d\x1fR\d\x1fS\d$')
# check_presence = [s for s in all_valid_seq_str if p.match(s)]
# # There are sequences with that enumeration. Hence the problem is given by the numbers. 

seq_0s = random.sample(sequences, k=n_0s)
check_seq_0s = list(map(lambda x: assign_outcome(x, already_splitted=False), seq_0s))
are_there_1s = sum(check_seq_0s) > 0 # 0

if are_there_1s:
    which_1 = list(filter(lambda x: assign_outcome(x, already_splitted=False)==1, seq_0s))
    if not set(which_1).issubset(set(all_valid_seq_str)): # empty set is subset by definition
        print("Attention! Something is wrong!")
    idx_2_modify = seq_0s.index(which_1[0])

seq_1s = random.sample(all_valid_seq_str, k=n_1s)

all_seq = seq_1s + seq_0s

# Now we need to create a df
df_seq = pd.DataFrame({'Sequences':all_seq})

df_seq["Outcome"] = list(map(lambda x: assign_outcome(x) , df_seq['Sequences']))

# Check duplicates
duplicates = df_seq.duplicated()
if any(duplicates):
    # If there are duplicates, show them
    print(df_seq[duplicates])
else:
    print("No duplicates!")

# Now train_val_test split:
X, y = df_seq["Sequences"], df_seq["Outcome"]
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.80, random_state=999)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.50, random_state=999)

for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]):
    df.to_csv(f"data/simulation/{name}.csv", index=False)