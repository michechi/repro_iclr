import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

csv_number = 'odd_3'
X_train = pd.read_csv(f"data/simulation/X_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv(f"data/simulation/y_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv(f"data/simulation/X_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv(f"data/simulation/y_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_test = pd.read_csv(f"data/simulation/X_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_test = pd.read_csv(f"data/simulation/y_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

df = pd.DataFrame({
    "Sequences":pd.concat([X_train.Sequences, X_val.Sequences, X_test.Sequences]),
    "Outcome":pd.concat([y_train.Outcome, y_val.Outcome, y_test.Outcome])
})
df['Outcome'].value_counts()/len(df)

def do_position(seq:str, sep:str="\x1f", shuffle=True)->str:
    """Function for placing positional numbers"""
    splitted = seq.split(sep)
    positions = range(1,len(splitted)+1)
    pos_seq = [f"{n}:{l}" for n, l in zip(positions, splitted)]
    if shuffle:
        random.shuffle(pos_seq)
    
    return(sep.join(pos_seq))

df["Sequences"] = df["Sequences"].apply(lambda x: do_position(x))

X, y = df["Sequences"], df["Outcome"]
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.80, random_state=999)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.50, random_state=999)

# Uncomment just if you want to save data!
number_csv = "odd_3_shf"
for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], [f"X_train_{number_csv}", f"X_val_{number_csv}", f"X_test_{number_csv}", f"y_train_{number_csv}", f"y_val_{number_csv}", f"y_test_{number_csv}"]):
    df.to_csv(f"data/simulation/{name}.csv", index=False)
