import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("./src/landmark_df_evo_correct.csv", na_values=['', 'None', 'NaN', 'na', 'nan'])
data.fillna('', inplace=True)

dataframe_no_overlap_2_merge = pd.read_csv("./src/dataframe_no_overlap_2_merge.csv")
landmark_df_evo_correct_no_overlap = data.merge(dataframe_no_overlap_2_merge, on=['subject_id', 'hadm_id'], how='inner')
landmark_df_evo_correct_no_overlap['days_since_last_visit'] = landmark_df_evo_correct_no_overlap['days_since_last_visit'].replace(0, 1)
landmark_df_evo_correct_no_overlap['days_since_last_visit_cumulate'] = landmark_df_evo_correct_no_overlap.groupby('subject_id')['days_since_last_visit'].transform(lambda x: [list(x[:i+1]) for i in range(len(x))])
landmark_df_evo_correct_no_overlap['days_since_last_visit_cumulate_sum'] = landmark_df_evo_correct_no_overlap['days_since_last_visit_cumulate'].apply(lambda x: list(np.cumsum([y for y in x if y > 0][::-1])[::-1]) if isinstance(x, list) else [-1])


# Split each subpopulation separately
train_list, val_list, test_list = [], [], []
visit_counts = landmark_df_evo_correct_no_overlap['subject_id'].value_counts()


for visit_count in range(1, 6):  # Visits 1 to 5
    selected_patients = visit_counts[visit_counts == visit_count].index
    subset = landmark_df_evo_correct_no_overlap[landmark_df_evo_correct_no_overlap['subject_id'].isin(selected_patients)].copy()

    patients = subset['subject_id'].unique()
    train_val_patients, test_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=9550,
        stratify=subset.groupby('subject_id')['death_in_90days'].max()
    )

    train_val_subset = subset[subset['subject_id'].isin(train_val_patients)]
    test_subset = subset[subset['subject_id'].isin(test_patients)]

    # Further split for validation
    train_patients, val_patients = train_test_split(
        train_val_subset['subject_id'].unique(), test_size=0.1, random_state=9550,
        stratify=train_val_subset.groupby('subject_id')['death_in_90days'].max()
    )

    train_subset = train_val_subset[train_val_subset['subject_id'].isin(train_patients)]
    val_subset = train_val_subset[train_val_subset['subject_id'].isin(val_patients)]

    train_list.append(train_subset)
    val_list.append(val_subset)
    test_list.append(test_subset)

# Merge the subpopulations
train_final = pd.concat(train_list).reset_index(drop=True)
val_final = pd.concat(val_list).reset_index(drop=True)
test_final = pd.concat(test_list).reset_index(drop=True)

# Save the sets to separate files
train_final.to_csv("./data/splitted/landmark_evo_train.csv", index=False)
val_final.to_csv("./data/splitted/landmark_evo_vali.csv", index=False)
test_final.to_csv("./data/splitted/landmark_evo_test.csv", index=False)