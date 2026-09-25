import pandas as pd
import matplotlib.pyplot as plt
from icdmappings import Mapper

mapper = Mapper()

diagnoses = pd.read_csv("./data/mimic-iv-3.1/hosp/diagnoses_icd.csv")
descriptions = pd.read_csv("./data/mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz")

diagnoses["diagnoses"] = diagnoses["icd_code"].astype(str) + "_" + diagnoses["icd_version"].astype(str)

# Method 1: Overall diagnosis frequency (most common approach)
def analyze_top_diagnoses(df, diagnosis_col='diagnoses', top_n=20):
    """
    Analyze top diagnoses in the dataset
    
    Parameters:
    df: pandas DataFrame
    diagnosis_col: name of the diagnosis column
    top_n: number of top diagnoses to show
    """
    
    # Count frequency of each diagnosis
    diagnosis_counts = df[diagnosis_col].value_counts()
    
    print(f"Top {top_n} Most Common Diagnoses:")
    print("-" * 40)
    top_diagnoses = diagnosis_counts.head(top_n) # odrdered
    for i, (diagnosis, count) in enumerate(top_diagnoses.items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i:2d}. {diagnosis}: {count} ({percentage:.1f}%)")
    
    return top_diagnoses

analyze_top_diagnoses(diagnoses)


def analyze_top_diagnoses_by_subjects(df, diagnosis_col='diagnoses', top_n=20):
    """
    Find diagnoses that affect the most unique subjects
    """
    
    # Count unique subjects per diagnosis
    subjects_per_diagnosis = df.groupby(diagnosis_col)['subject_id'].nunique().sort_values(ascending=False)
    
    print(f"\nTop {top_n} Diagnoses by Number of Affected Subjects:")
    print("-" * 50)
    N = df["subject_id"].nunique()
    top_by_subjects = subjects_per_diagnosis.head(top_n)
    for i, (diagnosis, subject_count) in enumerate(top_by_subjects.items(), 1):
        total_occurrences = df[df[diagnosis_col] == diagnosis].shape[0]
        percentage = total_occurrences/N*100
        print(f"{i:2d}. {diagnosis}: {subject_count} subjects ({total_occurrences} total occurrences - ({percentage:3f}))")
    
    return top_by_subjects

top_by_subjects = analyze_top_diagnoses_by_subjects(diagnoses)

top_by_subjects.index[0]

def find_name(code: str, descriptions: pd.DataFrame = descriptions) -> str:
    """given code_icd returns description"""
    code, icd = code.split("_")

    desc = descriptions.query(f'icd_version=={int(icd)} & icd_code=="{code}"')["long_title"].item()
    return(desc)

find_name('4019_9')


top_descriptions = [find_name(x) for x in top_by_subjects.index]
 
# For these we want to understant what's the amount of deaths we have in the dataset
## adding an extra-row for subjects depending whether dies or not (problem: we have just death in 90 days)
#diagnoses_death = diagnoses.merge(data[['subject_id', 'hadm_id', 'death_in_90days']], on=['subject_id', 'hadm_id'], how = 'left')

def analyze_top_diagnoses_by_subjects_death(df, diagnosis_col='diagnoses', top_n=20):
    """
    Find diagnoses that affect the most unique subjects
    """
    
    # Count unique subjects per diagnosis
    subjects_per_diagnosis = df.groupby(diagnosis_col)['subject_id'].nunique().sort_values(ascending=False)
    
    print(f"\nTop {top_n} Diagnoses by Number of Affected Subjects:")
    print("-" * 50)
    N = df["subject_id"].nunique()
    top_by_subjects = subjects_per_diagnosis.head(top_n)
    for i, (diagnosis, subject_count) in enumerate(top_by_subjects.items(), 1):
        total_occurrences = df[df[diagnosis_col] == diagnosis].shape[0]
        percentage = total_occurrences/N*100
        n_death = df[df[diagnosis_col]==diagnosis].groupby('subject_id')['death_in_90days'].max().sum()
        text = find_name(diagnosis)
        print(f"{i:2d}. {diagnosis}-{text}: {subject_count} subjects ({total_occurrences} total occurrences - ({percentage:3f}) - Death%: {n_death/total_occurrences*100:3f})")
    
    return top_by_subjects

#top_by_subjects_death = analyze_top_diagnoses_by_subjects_death(diagnoses_death)

# Mapping ICD-9 to ICD-10
mask = diagnoses['icd_version']==9
mapped_series = mapper.map(diagnoses.loc[mask, 'icd_code'], source = 'icd9', target = 'icd10')
diagnoses['mapped_codes'] = diagnoses['icd_code']
diagnoses.loc[mask, 'mapped_codes'] = mapped_series

# Mapping ICD-10 to ICD-9
mask = diagnoses['icd_version']==10
mapped_series = mapper.map(diagnoses.loc[mask, 'icd_code'], source = 'icd10', target = 'icd9')
diagnoses['mapped_codes_9'] = diagnoses['icd_code']
diagnoses.loc[mask, 'mapped_codes_9'] = mapped_series

pd.isna(diagnoses['mapped_codes']).sum().item() # 12442 na's

diagnoses['mapped_ccs'] = diagnoses['icd_code']

# ccs and ccsr not comparable (two different scales)
for icd, ccs in zip(['icd9', 'icd10'], ['ccs', 'ccsr']):
    mask = diagnoses['icd_version']==int(icd.replace("icd", ""))
    ccs_series = mapper.map(diagnoses.loc[mask, 'icd_code'], source = icd, target = ccs)
    diagnoses.loc[mask, 'mapped_ccs'] = ccs_series


# Find all the data with ICD-10 codes I63 - I61 (try as well with I60, I62, I64)
diag_icd10 = diagnoses[diagnoses["icd_code"].str.startswith(("I63", "I61", "I60", "I62", "I64"))]

# Understand where are they mapped (in terms of ICD-9)
counts = diag_icd10.groupby("mapped_codes_9")["icd_code"].value_counts()
counts.to_csv("counts.csv")

# In which ICD-9 codes are mapped the ICD-10 codes corresponding to stroke? (i.e. starting with I63 and I61. To check I60, I62 and I64 (no data))
mask_2_keep = pd.notna(diagnoses["mapped_codes"])
diagnoses_filtered = diagnoses.loc[mask_2_keep]
diag_mapped_icd9 = diagnoses_filtered.loc[diagnoses_filtered["mapped_codes"].str.startswith(("I63", "I61"))]
set(diag_mapped_icd9.loc[diag_mapped_icd9['icd_version']==9, 'icd_code'])

# Let's mark stroke
df_codes_stroke = pd.read_csv("./codes/codes_stroke.csv")
codes_stroke = df_codes_stroke['icd_code']

# check textual description:
codes_2_translate = df_codes_stroke['icd_code']+"_"+df_codes_stroke['icd_version'].astype(str)

for code in codes_2_translate:
    print(find_name(code))

mask = diagnoses[diagnoses['icd_code'].isin(codes_stroke)].index
diagnoses["stroke"] = 0
diagnoses.loc[mask, "stroke"] = 1

# Let's compute some statistics
## Unique patients with stroke
len(set(diagnoses.loc[diagnoses["stroke"]==1, 'subject_id'])) # 12872

## When first appearance occurs for each patient
## First, find patients with stroke
patients_stroke = set(diagnoses.loc[diagnoses["stroke"]==1, 'subject_id'])

# Next steps:
# 1: given hadm_id say which number of visit is for the patient

# 2: given the patient say when he got stroke (visit_id)
len(set(diagnoses.loc[diagnoses['stroke']==1, "hadm_id"])) # 14444

diagnoses_strokes_patients = diagnoses.loc[diagnoses['subject_id'].isin(patients_stroke), ('subject_id','hadm_id', 'stroke')].groupby("subject_id").value_counts()
diagnoses_strokes_patients.to_csv("diagnoses_strokes_patients.csv")

# When python crashes, starts from here
diagnoses_strokes_patients = pd.read_csv("diagnoses_strokes_patients.csv")
patients_stroke = set(diagnoses_strokes_patients['subject_id'])

sub_hadm_stroke = diagnoses_strokes_patients.loc[diagnoses_strokes_patients["stroke"]==1]

# Landmarks
df_landmarks = pd.read_csv("./src/landmark_df_evo_correct.csv", na_values=['', 'None', 'NaN', 'na', 'nan'])
df_landmarks.fillna('', inplace=True)

# Landmarks dataset with patietns with just stroke
df_landmarks = df_landmarks.loc[df_landmarks['subject_id'].isin(patients_stroke)]

# Let's combine stroke information
test = df_landmarks.merge(sub_hadm_stroke, how = 'left', on = ('subject_id', 'hadm_id'))

# In test we have for the same patients different visits where stroke happens. We need to take the first appearance
mask = test[test["stroke"]==1].groupby("subject_id")["landmark_visit"].idxmin()
df_landmarks_strokes_first_appearance = test.loc[mask]

# Save to csv
df_landmarks_strokes_first_appearance.to_csv("df_landmarks_strokes_first_appearance.csv")

plt.cla()
plt.hist(df_landmarks_strokes_first_appearance["landmark_visit"], bins=range(88))
len(set(df_landmarks_strokes_first_appearance["landmark_visit"]))
