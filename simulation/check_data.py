import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import string

# Read df
X_train = pd.read_csv("data/simulation/X_s_train_2.csv")
y_train = pd.read_csv("data/simulation/y_s_train_2.csv")

train_df = X_train.copy()
train_df["y_train"] = y_train

# Looking to the data it could be that the sequences of numbers for each 1sequence
# are the same. 
# Let's check this:
df_1s = train_df[train_df["y_train"]==1] # all ordered sequences in my df
df_0s = train_df[train_df["y_train"]==0] # all ordered sequences in my df
df_1s["Sequences"] # df_1s.Sequences df_1s.loc[:, "Sequences"] df_1s.loc[:, []]

# Sequences numbers:
def extract_numbers(seq:str, sep:str="\x1f") -> str:
    seq_numbers = "-".join([x[1] for x in seq.split(sep)])
    return(seq_numbers)

def extract_characters(seq:str, sep:str="\x1f") -> str:
    seq_char = "-".join([x[0] for x in seq.split(sep)])
    return(seq_char)

num_seq_1s=list(map(extract_numbers, df_1s.Sequences))
chr_seq_1s=list(map(extract_characters, df_1s.Sequences))

num_seq_0s=list(map(extract_numbers, df_0s.Sequences))
chr_seq_0s=list(map(extract_characters, df_0s.Sequences))

len(num_seq_1s)
len(set(num_seq_1s)) # 10 numbers

len(chr_seq_1s)
len(set(chr_seq_1s))
# since there are some duplicates..
chr_duplicates_1s = [(item, count) for item, count in collections.Counter(chr_seq_1s).items() if count > 1]
chr_duplicates_1s

# Let's check in the 0s sequences
len(num_seq_0s)
len(set(num_seq_0s))

len(chr_seq_0s)
len(set(chr_seq_0s))

# Check frequencies of starting letters:
# both for 0s and 1s
i = 8
stats_1s = collections.Counter(x[i*2] for x in chr_seq_1s)
stats_0s = collections.Counter(x[i*2] for x in chr_seq_0s)

# Ordina per alfabeto
letters = string.ascii_uppercase
counts_1s = [stats_1s.get(letter, 0) for letter in letters]
counts_0s = [stats_0s.get(letter, 0) for letter in letters]

# Plot affiancato
x = np.arange(len(letters))
width = 0.35  # Larghezza barre

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, counts_1s, width, label='Label=1', alpha=0.8)
ax.bar(x + width/2, counts_0s, width, label='Label=0', alpha=0.8)

ax.set_xlabel('Letters')
ax.set_ylabel('Count')
ax.set_title(f'{i+1}st/nd Letter Frequency by Label')
ax.set_xticks(x)
ax.set_xticklabels(letters)
ax.legend()
plt.show()
