# Data Exploration for top diagnoses in the dataset
import pandas as pd
import matplotlib.pyplot as plt
from icdmappings import Mapper

mapper = Mapper()

def mapper_map(code: str) -> str:
    icd_code, icd_version = code.split("_")
    print(f"Description: {find_name(code)}")
    if icd_version == "9":
        source = "icd9"
        target = "icd10"
    else:
        source = "icd10"
        target = "icd9"
    
    return(mapper.map(icd_code, source, target))

diagnoses = pd.read_csv("./data/mimic-iv-3.1/hosp/diagnoses_icd.csv")
descriptions = pd.read_csv("./data/mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz")

diagnoses["diagnoses"] = diagnoses["icd_code"].astype(str) + "_" + diagnoses["icd_version"].astype(str)


def analyze_top_diagnoses_by_subjects(df, diagnosis_col='diagnoses', top_n=20):
    """
    Find diagnoses that affect the most unique subjects
    """
    
    # Count unique subjects per diagnosis
    subjects_per_diagnosis = df.groupby(diagnosis_col)['subject_id'].nunique().sort_values(ascending=False)
    
    print(f"\nTop {top_n} Diagnoses by Number of Affected Subjects:")
    print("-" * 50)
    N = df["subject_id"].nunique()
    print(f"N: {N}")
    top_by_subjects = subjects_per_diagnosis.head(top_n)
    for i, (diagnosis, subject_count) in enumerate(top_by_subjects.items(), 1):
        total_occurrences = df[df[diagnosis_col] == diagnosis].shape[0] # In terms of total diagnoses
        percentage = subject_count/N*100
        print(f"{i:2d}. {diagnosis}: {subject_count} subjects ({total_occurrences} total occurrences - ({percentage:3f}))")
    
    return top_by_subjects

top_by_subjects = analyze_top_diagnoses_by_subjects(diagnoses, diagnosis_col='mapped_codes_9', top_n=13)

def find_name(code: str, descriptions: pd.DataFrame = descriptions) -> str:
    """given code_icd returns description"""
    code, icd = code.split("_")

    desc = descriptions.query(f'icd_version=={int(icd)} & icd_code=="{code}"')["long_title"].item()
    return(desc)

top_descriptions = [find_name(x) for x in top_by_subjects.index]

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

# After mapping all the ICD-9 codes to ICD-10 codes, lets' use this to investigate the top 10 diagnoses 
# for unique patients.

def find_top_diagnoses(data: pd.DataFrame, diagnoses_col: str ='mapped_codes', top: int = 10) -> pd.DataFrame:

    top_diag = pd.DataFrame(data.groupby(diagnoses_col)["subject_id"].nunique().sort_values(ascending=False).head(top))
    top_diag["description"] = [find_name(x+"_10") for x in top_diag.index]
    top_diag.insert(0, "icd_10_code", top_diag.index)
    top_diag=top_diag.reset_index(drop=True)
    return(top_diag)
    
test = find_top_diagnoses(diagnoses)

# SUMMARY OF DATA DISTRIBUTION, GIVEN TOP CODES
def make_summary_top(data: pd.DataFrame, top: pd.DataFrame, diagnoses: str='mapped_codes'):
    codes = top['icd_10_code'] # all the code we have

    for code in codes:
        result = {}
        result["unique_patients"] = len(set(diagnoses.loc[diagnoses[diagnoses]==code, 'subject_id']))



code = "D649" # TO CHANGE


# Let's compute some statistics
## Unique patients with nfmi
len(set(diagnoses.loc[diagnoses["mapped_codes"]==code, 'subject_id'])) 

## When first appearance occurs for each patient
## First, find patients with nfmi
patients_nfmi = set(diagnoses.loc[diagnoses["mapped_codes"]==code, 'subject_id'])

# given the patient say when he got nfmi (visit_id)
len(set(diagnoses.loc[diagnoses['mapped_codes']==code, "hadm_id"])) # 16537

diagnoses_nfmi_patients = diagnoses.loc[diagnoses['subject_id'].isin(patients_nfmi), ('subject_id','hadm_id', 'mapped_codes')].groupby("subject_id").value_counts()
diagnoses_nfmi_patients.to_csv("diagnoses_mapped_codes_patients.csv")

# When python crashes, starts from here
diagnoses_nfmi_patients = pd.read_csv("diagnoses_mapped_codes_patients.csv")
patients_nfmi = set(diagnoses_nfmi_patients['subject_id'])

sub_hadm_nfmi = diagnoses_nfmi_patients.loc[diagnoses_nfmi_patients["mapped_codes"]==code]

# Landmarks
df_landmarks_tot = pd.read_csv("./src/landmark_df_evo_correct.csv", na_values=['', 'None', 'NaN', 'na', 'nan'])
df_landmarks_tot.fillna('', inplace=True)

# Landmarks dataset with patietns with just nfmi
df_landmarks = df_landmarks_tot.loc[df_landmarks_tot['subject_id'].isin(patients_nfmi)]

# Let's combine stroke information
test = df_landmarks.merge(sub_hadm_nfmi, how = 'left', on = ('subject_id', 'hadm_id'))

# In test we have for the same patients different visits where stroke happens. We need to take the first appearance
mask = test[test["mapped_codes"]==code].groupby("subject_id")["landmark_visit"].idxmin()
df_landmarks_nfmi_first_appearance = test.loc[mask]

for i in range(10):
    print(len(df_landmarks_nfmi_first_appearance.loc[df_landmarks_nfmi_first_appearance["landmark_visit"]==i+1]))

# Save to csv
df_landmarks_nfmi_first_appearance.to_csv("df_landmarks_mapped_codes_first_appearance.csv")

plt.cla()
plt.hist(df_landmarks_nfmi_first_appearance["landmark_visit"], bins=range(88))
len(set(df_landmarks_nfmi_first_appearance["landmark_visit"]))
