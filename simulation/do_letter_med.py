import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score, f1_score

diagnosis_map = {
    "A": {"diagnosis": "Allergic rhinitis", "ICD10": "J30.9", "CCS": 126},
    "B": {"diagnosis": "Bronchitis (acute)", "ICD10": "J20.9", "CCS": 125},
    "C": {"diagnosis": "Conjunctivitis", "ICD10": "H10.9", "CCS": 91},
    "D": {"diagnosis": "Depression", "ICD10": "F32.9", "CCS": 657},
    "E": {"diagnosis": "Eczema (atopic dermatitis)", "ICD10": "L20.9", "CCS": 203},
    "F": {"diagnosis": "Flu (influenza)", "ICD10": "J11.1", "CCS": 123},
    "G": {"diagnosis": "Gastroenteritis", "ICD10": "A09", "CCS": 135},
    "H": {"diagnosis": "Hypertension (mild)", "ICD10": "I10", "CCS": 98},
    "I": {"diagnosis": "Insomnia", "ICD10": "G47.00", "CCS": 653},
    "J": {"diagnosis": "Jaundice", "ICD10": "R17", "CCS": 108},
    "K": {"diagnosis": "Knee pain (patellofemoral syndrome)", "ICD10": "M22.2X9", "CCS": 205},
    "L": {"diagnosis": "Laryngitis", "ICD10": "J04.0", "CCS": 124},
    "M": {"diagnosis": "Migraine", "ICD10": "G43.909", "CCS": 84},
    "N": {"diagnosis": "Nasal congestion", "ICD10": "R09.81", "CCS": 126},
    "O": {"diagnosis": "Otitis media", "ICD10": "H66.90", "CCS": 92},
    "P": {"diagnosis": "Pharyngitis", "ICD10": "J02.9", "CCS": 124},
    "Q": {"diagnosis": "Q fever", "ICD10": "A78", "CCS": 1},
    "R": {"diagnosis": "Rhinosinusitis (acute)", "ICD10": "J01.90", "CCS": 126},
    "S": {"diagnosis": "Sprain (ankle)", "ICD10": "S93.409A", "CCS": 232},
    "T": {"diagnosis": "Tonsillitis", "ICD10": "J03.90", "CCS": 124},
    "U": {"diagnosis": "Urinary tract infection (UTI)", "ICD10": "N39.0", "CCS": 159},
    "V": {"diagnosis": "Vertigo (benign positional)", "ICD10": "H81.10", "CCS": 95},
    "W": {"diagnosis": "Wernicke encephalopathy", "ICD10": "E51.2", "CCS": 94},
    "X": {"diagnosis": "Xerostomia", "ICD10": "K11.7", "CCS": 57},
    "Y": {"diagnosis": "Yeast infection (Candida vaginitis)", "ICD10": "B37.3", "CCS": 13},
    "Z": {"diagnosis": "Zoster (mild shingles)", "ICD10": "B02.9", "CCS": 12}
}

# Reading data - already splitted!
csv_number = 9
letters_4_key = ["W", "D", "Q", "J", "X", "U"] # would it be nice to have a dictionary with key-csv-number
X_train = pd.read_csv(f"data/simulation/tested/X_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_train = pd.read_csv(f"data/simulation/tested/y_train_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_val = pd.read_csv(f"data/simulation/tested/X_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_val = pd.read_csv(f"data/simulation/tested/y_val_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
X_test = pd.read_csv(f"data/simulation/tested/X_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
y_test = pd.read_csv(f"data/simulation/tested/y_test_{csv_number}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

def do_med_seq(seq):
    med_seq = "\x1f".join([diagnosis_map[x]['diagnosis'] for x in seq.split('\x1f')])
    return med_seq

X_train['Med'] = X_train['Sequences'].apply(do_med_seq)
X_val['Med'] = X_val['Sequences'].apply(do_med_seq)
X_test['Med'] = X_test['Sequences'].apply(do_med_seq)


X_test.to_csv("./data/simulation/X_test_9_med.csv", index=False)
X_train.to_csv("./data/simulation/X_train_9_med.csv", index=False)
X_val.to_csv("./data/simulation/X_val_9_med.csv", index=False)
