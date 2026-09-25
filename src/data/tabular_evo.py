
import pandas as pd
import numpy as np

# ------------------------- STEP 1: LOAD DATA -------------------------

# Load patient demographics
patients = pd.read_csv("./data/mimic-iv-3.1/hosp/patients.csv")

# Load hospital admissions
admissions = pd.read_csv("./data/mimic-iv-3.1/hosp/admissions.csv")

# Load ICU stays
icustays = pd.read_csv("./data/mimic-iv-3.1/icu/icustays.csv")

# Load diagnoses (ICD codes assigned to patients)
diagnoses = pd.read_csv("./data/mimic-iv-3.1/hosp/diagnoses_icd.csv")

# Load prescriptions (medications administered)
prescriptions = pd.read_csv("./data/mimic-iv-3.1/hosp/prescriptions.csv")

# Load medical procedures (ICD codes for treatments)
procedures = pd.read_csv("./data/mimic-iv-3.1/hosp/procedures_icd.csv")

# Load ICD descriptions (for both ICD-9 and ICD-10)
icd_diagnoses = pd.read_csv("./data/mimic-iv-3.1/hosp/d_icd_diagnoses.csv.gz")
icd_procedures = pd.read_csv("./data/mimic-iv-3.1/hosp/d_icd_procedures.csv.gz")

# Convert ICU stay timestamps to datetime format
icustays['intime'] = pd.to_datetime(icustays['intime'], errors='coerce')
icustays['outtime'] = pd.to_datetime(icustays['outtime'], errors='coerce')

# Define mapping for admission types
emergency_types = ['DIRECT EMER.', 'EW EMER.', 'URGENT']
normal_types = ['AMBULATORY OBSERVATION', 'DIRECT OBSERVATION', 'ELECTIVE', 
                'EU OBSERVATION', 'OBSERVATION ADMIT', 'SURGICAL SAME DAY ADMISSION']

# Create a new column 'admission_category' based on the mapping
admissions['admission_category'] = admissions['admission_type'].apply(
    lambda x: 'EMERGENCY' if x in emergency_types else 'NORMAL'
)

# Check if mapping worked correctly
print(admissions['admission_category'].value_counts())

# ------------------------- STEP 2: MAP ICD CODES TO DESCRIPTIONS -------------------------

# Merge diagnoses with ICD descriptions
diagnoses = diagnoses.merge(icd_diagnoses, on=["icd_code", "icd_version"], how="left")
diagnoses['diagnosis_description'] = diagnoses['long_title'].fillna("Unknown diagnosis")

# Merge procedures with ICD descriptions
procedures = procedures.merge(icd_procedures, on=["icd_code", "icd_version"], how="left")
procedures['procedure_description'] = procedures['long_title'].fillna("Unknown procedure")

# Keep only necessary columns
diagnoses = diagnoses[['subject_id', 'hadm_id', 'diagnosis_description', 'icd_code']]
procedures = procedures[['subject_id', 'hadm_id', 'procedure_description', 'icd_code']]

# ------------------------- STEP 3: COMPUTE AGE AT EVENTS & MORTALITY -------------------------

# Keep only relevant patient information
patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']]

# Merge patient data into all event tables
admissions = admissions.merge(patients, on='subject_id', how='left')
icustays = icustays.merge(patients, on='subject_id', how='left')
diagnoses = diagnoses.merge(patients, on='subject_id', how='left')
procedures = procedures.merge(patients, on='subject_id', how='left')
prescriptions = prescriptions.merge(patients, on='subject_id', how='left')

# Convert date columns
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
prescriptions['starttime'] = pd.to_datetime(prescriptions['starttime'])
patients['dod'] = pd.to_datetime(patients['dod'], errors='coerce')

# Compute age at event time
admissions['age_at_event'] = admissions['anchor_age'] + (admissions['admittime'].dt.year - admissions['anchor_year'])
prescriptions['age_at_event'] = prescriptions['anchor_age'] + (prescriptions['starttime'].dt.year - prescriptions['anchor_year'])

# Assign age at death for deceased patients
patients['age_at_death'] = patients['anchor_age'] + (patients['dod'].dt.year - patients['anchor_year'])
patients['death_flag'] = patients['dod'].notna().astype(int)

# Merge mortality data into admissions
admissions = admissions.merge(patients[['subject_id', 'death_flag', 'age_at_death']], on='subject_id', how='left')

##### Adding Died During Visit variable 
# Convert discharge time to datetime format
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'], errors='coerce')

# Determine if the patient died during this admission
admissions['died_during_visit'] = (
    (admissions['death_flag'] == 1) & 
    (admissions['dod'] >= admissions['admittime']) & 
    (admissions['dod'] <= admissions['dischtime'])
).astype(int)

##### Creating how many days until the next hospitalization
# Sort admissions by patient ID and admission time
admissions = admissions.sort_values(by=['subject_id', 'admittime'])

# Get the next admission time per patient
admissions['next_admittime'] = admissions.groupby('subject_id')['admittime'].shift(-1)

# Compute the number of days until the next visit
admissions['days_until_next_visit'] = (admissions['next_admittime'] - admissions['dischtime']).dt.days

# Fill NaN values with -1 (for last visit of each patient)
admissions['days_until_next_visit'].fillna(-1, inplace=True)

##### How many days from the last visit to the death?
# Get the last discharge time per patient
last_discharge = admissions.groupby('subject_id')['dischtime'].max().reset_index()
last_discharge.rename(columns={'dischtime': 'last_dischtime'}, inplace=True)

# Merge last discharge time into patients data
patients = patients.merge(last_discharge, on='subject_id', how='left')

# Calculate days from last visit to death (if patient died after last visit)
patients['days_from_last_visit_to_death'] = (patients['dod'] - patients['last_dischtime']).dt.days

# If patient didn't die, set the value to NaN or -1
patients.loc[patients['death_flag'] == 0, 'days_from_last_visit_to_death'] = np.nan

admissions = admissions.merge(
    patients[['subject_id', 'days_from_last_visit_to_death']], 
    on='subject_id', 
    how='left'
)

# ------------------------- STEP 4: BUILD TABULAR DATASET -------------------------

# Select main features
features = admissions[['subject_id',
                       'hadm_id',
                       'age_at_event',
                       'gender',
                       'admission_type',
                       'admission_category',
                       'discharge_location',
                       'death_flag',
                       'age_at_death',
                       'died_during_visit',
                       'days_from_last_visit_to_death',
                       'days_until_next_visit'
                       ]]

# ICU stays: Number of ICU admissions and length of stay per hospitalization
icu_summary = icustays.groupby('hadm_id').agg(
    icu_admissions=('stay_id', 'count'),
    icu_hours=('intime', lambda x: (x.max() - x.min()).total_seconds() / 3600)  #  to hours
).reset_index()

# Diagnoses: Count number of diagnoses per hospitalization
diagnosis_summary = diagnoses.groupby('hadm_id').agg(
    num_diagnoses=('icd_code', 'count'),
    diagnosis_list=('diagnosis_description', lambda x: list(x.unique()))
).reset_index()

# Procedures: Count number of procedures per hospitalization
procedure_summary = procedures.groupby('hadm_id').agg(
    num_procedures=('icd_code', 'count'),
    procedure_list=('procedure_description', lambda x: list(x.unique()))
).reset_index()

# Medications: Count number of prescribed drugs per hospitalization
# Assume 'via' is a fixed value, e.g. 'IV'

# Create the 'dose' column, concatenating this fixed value as well
prescriptions['dose'] = prescriptions.apply(
    lambda row: f"{row['dose_val_rx']} {row['dose_unit_rx']} of {row['formulary_drug_cd']} in {row['form_unit_disp']} through {row['route']}",
    axis=1
)

# Group as before
med_summary = prescriptions.groupby('hadm_id').agg(
    num_medications=('drug', 'count'),
    medication_list=('drug', lambda x: list(x.unique())),
    dose_list=('dose', lambda x: list(x))
).reset_index()

# Merge all features into a single dataset
dataset = features.merge(icu_summary, on='hadm_id', how='left')
dataset = dataset.merge(diagnosis_summary, on='hadm_id', how='left')
dataset = dataset.merge(procedure_summary, on='hadm_id', how='left')
dataset = dataset.merge(med_summary, on='hadm_id', how='left')

# Fill missing values with 0
dataset.fillna({'icu_admissions': 0, 'icu_hours': 0, 'num_diagnoses': 0, 'num_procedures': 0, 'num_medications': 0}, inplace=True)
dataset.fillna({'diagnosis_list': 'None', 'procedure_list': 'None', 'medication_list': 'None', 'dose_list': 'None'}, inplace=True)

# ------------------------- STEP 5: SAVE & DISPLAY DATASET -------------------------

# Save the dataset to a CSV file for ML usage
dataset.to_csv("mimiciv_clinical_dataset_tabular_death_visit_evo.csv", index=False)
