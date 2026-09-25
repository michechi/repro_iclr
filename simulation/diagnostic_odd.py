import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split



# Reading data - already splitted!
# 1: lags [7,5,4,2]
# 2: lags [9,8,7,6]
letters_4_key = ["W", "D", "Q", "J", "X", "N"]

csv_number = 'odd_filt'
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

df['even'] = df['Sequences'].apply(lambda x: len(x.split("\x1f"))%2==0)
df['last_letter'] = df['Sequences'].apply(lambda x: x.split("\x1f")[-1])
df['is_key']=df['last_letter'].apply(lambda x: x in letters_4_key)

df[['even','is_key']].groupby(["even"]).value_counts()

df_filtered = df.loc[(~(~df['even'] & df['is_key']))]

X, y = df_filtered["Sequences"], df_filtered["Outcome"]
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.80, random_state=999)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, train_size=0.50, random_state=999)

# Uncomment just if you want to save data!
number_csv = "odd_filt"  # partendo da odd_2 e filtrando per df_filtered = df.loc[(~(~df['even'] & df['is_key']))]
for df, name in zip([X_train, X_val, X_test, y_train, y_val, y_test], [f"X_train_{number_csv}", f"X_val_{number_csv}", f"X_test_{number_csv}", f"y_train_{number_csv}", f"y_val_{number_csv}", f"y_test_{number_csv}"]):
    df.to_csv(f"data/simulation/{name}.csv", index=False)
