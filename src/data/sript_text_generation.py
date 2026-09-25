import pandas as pd

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
# ------------------------- STEP 2: MAP ICD CODES TO DESCRIPTIONS -------------------------

# Merge diagnoses with ICD descriptions
diagnoses = diagnoses.merge(icd_diagnoses, on=["icd_code", "icd_version"], how="left").rename(columns={'icd_version_x': 'icd_version'})
diagnoses['diagnosis_description'] = diagnoses['long_title'].fillna("Unknown diagnosis")

# Merge procedures with ICD descriptions
procedures = procedures.merge(icd_procedures, on=["icd_code", "icd_version"], how="left").rename(columns={'icd_version_x': 'icd_version'})
procedures['procedure_description'] = procedures['long_title'].fillna("Unknown procedure")# Keep only necessary columns
diagnoses = diagnoses[['subject_id', 'hadm_id', 'icd_version', 'diagnosis_description', 'icd_code']]
procedures = procedures[['subject_id', 'hadm_id', 'icd_version', 'procedure_description', 'icd_code']]# ------------------------- STEP 3: COMPUTE AGE AT EVENTS -------------------------

# Keep only relevant patient information
patients = patients[['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod']]

# Merge patient age into all event tables
admissions = admissions.merge(patients, on='subject_id', how='left', suffixes=None)
icustays = icustays.merge(patients, on='subject_id', how='left', suffixes=None)
diagnoses = diagnoses.merge(patients, on='subject_id', how='left', suffixes=None)
procedures = procedures.merge(patients, on='subject_id', how='left', suffixes=None)
prescriptions = prescriptions.merge(patients, on='subject_id', how='left', suffixes=None)

# Convert date columns
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
icustays['intime'] = pd.to_datetime(icustays['intime'])
prescriptions['starttime'] = pd.to_datetime(prescriptions['starttime'])

# Convert 'dod' (date of death) to datetime format
patients['dod'] = pd.to_datetime(patients['dod'], errors='coerce')

# Compute age at death for deceased patients
# DONE in few lines below
#patients['age_at_death'] = patients['anchor_age'] + (patients['dod'].dt.year - patients['anchor_year'])

# Function to compute patient age at event time
def compute_age_at_event(df, event_col):
    df['event_year'] = df[event_col].dt.year
    df['age_at_event'] = df['anchor_age'] + (df['event_year'] - df['anchor_year'])
    return df

# Apply age computation for events with timestamps
admissions = compute_age_at_event(admissions, 'admittime')
icustays = compute_age_at_event(icustays, 'intime')
prescriptions = compute_age_at_event(prescriptions, 'starttime')
patients = compute_age_at_event(patients, 'dod')

# Directly assign age for diagnoses and procedures (since they don't have exact dates)
diagnoses['age_at_event'] = diagnoses['anchor_age']
procedures['age_at_event'] = procedures['anchor_age']
patient_ids = patients['subject_id'].unique()
len(patient_ids)

# ------------------------- STEP 4: GENERATE STRUCTURED CLINICAL HISTORY -------------------------

def generate_patient_history(subject_id):
    """Generates a structured clinical history for a given patient with enhanced details and time since last visit, including death details."""
    
    print(f"Generating clinical history for patient: {subject_id}\n")

    history = [f"Patient {subject_id} Clinical History:\n"]

    patient_info = patients[patients['subject_id'] == subject_id]
    gender = patient_info['gender'].values[0] if not patient_info.empty else "Unknown"
    died = patient_info['dod'].notna().values[0] if not patient_info.empty else False
    age_at_death = patient_info['age_at_event'].values[0] if died else None
    death_date = patient_info['dod'].values[0] if died else None

    patient_admissions = admissions[admissions['subject_id'] == subject_id].sort_values('admittime')

    last_discharge = None

    visit_n = 1

    for age, adm_group in patient_admissions.groupby('age_at_event'):


        for _, adm in adm_group.iterrows():
            
            if visit_n == 1:
                ordinal = "st"
            elif visit_n == 2:
                ordinal = "nd"
            elif visit_n == 3:
                ordinal = "rd"
            else:
                ordinal = "th"

            event_type = "Emergency" if adm['admission_type'] in ["EMERGENCY", "URGENT"] else "Normal"
            admittime = adm['admittime']
            dischtime = pd.to_datetime(adm['dischtime'])
            stay_duration = dischtime - admittime

            if stay_duration.days >= 1:
                stay_length = f"{stay_duration.days} days"
            else:
                stay_length = f"{int(stay_duration.total_seconds() // 3600)} hours"

            interval_str = f", {(admittime - last_discharge).days} days since the previous one" if last_discharge else None

            history.append(
                "\n {visit_n}-{ordinal} visit{interval_str}: \
                    \n Age: {age} \
                    \n Hospitalization ({event_type}) in {month_of_visit} (code - {event_code}) \
                    \n Duration: {stay_length} - Discharge status: {discharge_location}"
                .format(
                    event_type=event_type,
                    month_of_visit=admittime.strftime('%B'),
                    event_code=adm['hadm_id'],
                    stay_length=stay_length,
                    interval_str=interval_str if interval_str is not None else "",
                    discharge_location=adm['discharge_location'],
                    age = int(age),
                    visit_n = visit_n,
                    ordinal = ordinal
                )
            )

            last_discharge = dischtime

            patient_diagnoses = diagnoses[(diagnoses['subject_id'] == subject_id) & (diagnoses['hadm_id'] == adm['hadm_id'])]
            if not patient_diagnoses.empty:
                history.append("\n  Diagnoses:")
                for _, diag in patient_diagnoses.iterrows():
                    history.append(f"    - {diag['diagnosis_description']}")

            patient_procedures = procedures[(procedures['subject_id'] == subject_id) & (procedures['hadm_id'] == adm['hadm_id'])]
            if not patient_procedures.empty:
                history.append("\n  Procedures:")
                for _, proc in patient_procedures.iterrows():
                    history.append(f"    - {proc['procedure_description']}")

            patient_meds = prescriptions[(prescriptions['subject_id'] == subject_id) & (prescriptions['hadm_id'] == adm['hadm_id'])]
            if not patient_meds.empty:
                history.append("\n  Medications Prescribed:")
                for _, med in patient_meds.iterrows():
                    dose_val = med.get('dose_val_rx', 'Unknown')
                    dose_unit = med.get('dose_unit_rx', '')
                    drug_name = med.get('drug', 'Unknown')
                    form_val = med.get('form_val_disp', '')
                    form_unit = med.get('form_unit_disp', '')
                    route = med.get('route', 'Unknown')

                    history.append(
                        f"    - {dose_val} {dose_unit} of {drug_name} "
                        f"in {form_val} {form_unit} through {route}"
                    )

            patient_icu = icustays[(icustays['subject_id'] == subject_id) & (icustays['hadm_id'] == adm['hadm_id'])]
            for _, icu in patient_icu.iterrows():
                history.append("\n  ICU stay summary:")
                if pd.notna(icu['outtime']):
                    icu_stay_duration = icu['outtime'] - icu['intime']
                    icu_hours = int(icu_stay_duration.total_seconds() // 3600)
                    history.append(f"    - ICU admission lasted {icu_hours} hours")
                else:
                    history.append("    - ICU admission (discharge date unknown)")

            last_discharge = dischtime

            visit_n += 1

    if died:
        death_date = pd.to_datetime(death_date)
        death_month = death_date.strftime('%B')
        if last_discharge and death_date > last_discharge:
            days_since_last = (death_date - last_discharge).days
            death_info = f"died in {death_month}, {days_since_last} days after the last hospitalization"
        else:
            death_info = f"died during hospitalization in {death_month}"

        history.append(f"\n+++ Patient of gender: {gender}, {death_info}, at age: {age_at_death} +++")
    else:
        history.append(f"\n+++ Patient of gender: {gender}, with no evidence of death +++")

    return "\n".join(history)


# ------------------------- STEP 5: GENERATE AND SAVE SUMMARIES -------------------------

# Process multiple patients (limit to 100 patients for testing)
patient_ids = patients['subject_id'].unique()
summaries = {pid: generate_patient_history(pid) for pid in patient_ids[:100]}

# Save all summaries to a file
with open("clinical_histories.txt", "w") as file:
    for summary in summaries.values():
        file.write(summary + "\n\n")
