import random
import ast
import pandas as pd

from collections import Counter
from num2words import num2words

# def no_narrative_prompt(row):
#     narrative = "Based on this information, what is the probability of mortality within 90 days?"
#     if pd.notna(row['diag_text']) and row['diag_text'].strip():
#         narrative += f"Diagnoses: {set(row['diag_text'])}"
#     if pd.notna(row['med_text']) and row['med_text'].strip():
#         narrative += f"Medications: {set(row['med_text'])}"
#     if pd.notna(row['proc_text']) and row['proc_text'].strip():
#         narrative += f"Procedures: {set(row['proc_text'])}"

#     return narrative

def no_narrative_prompt(row, to_split='\n'):
    information_set = set()
    narrative = 'Based on this information, what is the probability of mortality within 90 days?'
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        information_set = information_set.union(set(row["diag_text"].split(f"{to_split}")))
    if pd.notna(row['med_text']) and row['med_text'].strip():
        information_set = information_set.union(set(row["med_text"].split(f"{to_split}")))
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        information_set = information_set.union(set(row["proc_text"].split(f"{to_split}")))
    
    # shuffling the set
    information_list = list(information_set)
    random.shuffle(information_list)

    narrative += '; '.join(information_list).strip()

    return narrative

def naive_narrative_prompt(row):
    narrative = 'Based on this information, what is the probability of mortality within 90 days?'
    narrative += f"\nPatient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the patient history till visit {row['landmark_visit']}."

    max_visit = int(row['landmark_visit'])

    for visit in range(1, max_visit+1):
        narrative += f"\nDuring visit {visit}:"
        narrative += f"\n\tDiagnosis: {'; '.join(ast.literal_eval(row['diag_per_visit'])[visit])}"
        narrative += f"\n\tMedications: {'; '.join(ast.literal_eval(row['meds_per_visit'])[visit])}"
        narrative += f"\n\tProcedures: {'; '.join(ast.literal_eval(row['proc_per_visit'])[visit])}"
    
    return narrative

def compact_narrative_prompt(row):
    narrative = f"What is the probability of death in the next 90 days for {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    current_visit = row['landmark_visit']
    type = row['admission_category']
    narrative += f"Visit number {current_visit} - {type} \n"

    if row['days_since_previous_visit'] != -1:
        narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    # --- Diagnoses ---
    narrative += "\nDiagnosis history:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])
    
    # Frequency count of diagnoses
    all_diags = []
    for diags in diag_per_visit.values():
        all_diags.extend(diags)
    diag_counts = Counter(all_diags)

    # Chronic diagnoses are those that appear in at least 2 visits
    chronic_diags = [d for d, c in diag_counts.items() if c >= 2]

    # New diagnoses are those that appear only in the current visit
    current_diags = diag_per_visit[int(current_visit)]
    new_diags = [d for d in current_diags if diag_counts[d] == 1]

    if chronic_diags:
        narrative += "\n\t"+f"Chronic diagnoses: {'; '.join(chronic_diags)}".strip()
    if new_diags:
        narrative += "\n\t"+f"New diagnoses in this visit: {'; '.join(new_diags)}".strip()
    if not chronic_diags and not new_diags:
        narrative += "\n\tNo diagnoses recorded"

    # --- Medicines ---
    narrative += "\nPrescriptions history:"
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])
    
    all_meds = []
    for meds in meds_per_visit.values():
        all_meds.extend(meds)
    med_counts = Counter(all_meds)

    chronic_meds = [m for m, c in med_counts.items() if c >= 2]
    current_meds = meds_per_visit[int(current_visit)]
    new_meds = [m for m in current_meds if med_counts[m] == 1]

    if chronic_meds:
        narrative += "\n\t"+f"Chronic medications: {'; '.join(chronic_meds)}".strip()
    if new_meds:
        narrative += "\n\t"+f"New medications in this visit: {'; '.join(new_meds)}".strip()
    if not chronic_meds and not new_meds:
        narrative += "\n\t"+"No medications recorded"

    # --- Procedure ---
    narrative += "\nProcedures history:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])

    all_proc = []
    for procs in proc_per_visit.values():
        all_proc.extend(procs)
    proc_counts = Counter(all_proc)

    chronic_proc = [p for p, c in proc_counts.items() if c >= 2]
    current_proc = proc_per_visit[int(current_visit)]
    new_proc = [p for p in current_proc if proc_counts[p] == 1]

    if chronic_proc:
        narrative += "\n\t"+f"Chronic procedures: {'; '.join(chronic_proc)}".strip()
    if new_proc:
        narrative += "\n\t"+f"New procedures in this visit: {'; '.join(new_proc)}".strip()
    if not chronic_proc and not new_proc:
        narrative += "\n\t"+"No procedures recorded"

    return narrative

def compact_no_time_prompt(row):
    narrative = f"What is the probability of death in the next 90 days for {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    narrative += f"Visit type: {row['admission_category']}\n"

    current_visit = int(row['landmark_visit'])

    # --- Diagnoses ---
    narrative += "Diagnosis history:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])

    # Events from previous visits
    past_diags = [diag for key, values in diag_per_visit.items() if int(key) != current_visit for diag in values]
    past_diag_counts = Counter(past_diags)
    chronic_diags = [diag for diag, count in past_diag_counts.items() if count >= 2]

    # Events from the last visit
    current_diags = diag_per_visit.get(current_visit, [])

    # Union and final count
    final_diags = chronic_diags + current_diags
    diag_summary = Counter(final_diags)
    if diag_summary:
        diag_str = '; '.join([f"{d}" for d, _ in diag_summary.items()])
        narrative += "\n\t"+f"{diag_str}".strip()
    else:
        narrative += "No diagnoses recorded"

    # --- Medications ---
    narrative += "\nPrescriptions history: "
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])

    past_meds = [med.replace(";", "").strip() for key, values in meds_per_visit.items() if int(key) != current_visit for med in values]
    past_meds_counts = Counter(past_meds)
    chronic_meds = [med.replace(";", "").strip() for med, count in past_meds_counts.items() if count >= 2]

    current_meds = meds_per_visit.get(current_visit, [])
    final_meds = chronic_meds + current_meds
    med_summary = Counter(final_meds)
    
    if med_summary:
        med_str = '; '.join([f"{m}" for m, _ in med_summary.items()])
        narrative += "\n\t"+f"{med_str}".strip()
    else:
        narrative += "No medications recorded."

    # --- Procedures ---
    narrative += "\nProcedures history:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])

    past_procs = [proc for key, values in proc_per_visit.items() if int(key) != current_visit for proc in values]
    past_proc_counts = Counter(past_procs)
    chronic_procs = [proc for proc, count in past_proc_counts.items() if count >= 2]

    current_procs = proc_per_visit.get(current_visit, [])
    final_procs = chronic_procs + current_procs
    proc_summary = Counter(final_procs)
    if proc_summary:
        proc_str = '; '.join([f"{p}" for p, _ in proc_summary.items()])
        narrative += "\n\t"+f"{proc_str}".strip()
    else:
        narrative += "\n\t"+"No procedures recorded"

    return narrative

def compact_no_time_prompt_rnd(row):
    narrative = f"What is the probability of death in the next 90 days for {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    narrative += f"Visit type: {row['admission_category']}"

    current_visit = int(row['landmark_visit'])

    # --- Diagnoses ---
    narrative += "\nDiagnosis history:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])

    past_diags = [diag for key, values in diag_per_visit.items() if int(key) != current_visit for diag in values]
    past_diag_counts = Counter(past_diags)
    chronic_diags = [diag for diag, count in past_diag_counts.items() if count >= 2]

    current_diags = diag_per_visit.get(current_visit, [])
    final_diags = chronic_diags + current_diags

    if final_diags:
        random.shuffle(final_diags) 
        diag_str = '; '.join(final_diags)
        narrative += "\n\t"+f"{diag_str}".strip()
    else:
        narrative += "\n\t"+"No diagnoses recorded"

    # --- Medications ---
    narrative += "\nPrescriptions history:"
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])

    past_meds = [med for key, values in meds_per_visit.items() if int(key) != current_visit for med in values]
    past_meds_counts = Counter(past_meds)
    chronic_meds = [med for med, count in past_meds_counts.items() if count >= 2]

    current_meds = meds_per_visit.get(current_visit, [])
    final_meds = chronic_meds + current_meds

    if final_meds:
        random.shuffle(final_meds)  # <-- SHUFFLING
        med_str = '; '.join(final_meds)
        narrative += "\n\t"+f"{med_str}".strip()
    else:
        narrative += "\n\t"+"No medications recorded"

    # --- Procedures ---
    narrative += "\nProcedures history:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])

    past_procs = [proc for key, values in proc_per_visit.items() if int(key) != current_visit for proc in values]
    past_proc_counts = Counter(past_procs)
    chronic_procs = [proc for proc, count in past_proc_counts.items() if count >= 2]

    current_procs = proc_per_visit.get(current_visit, [])
    final_procs = chronic_procs + current_procs

    if final_procs:
        random.shuffle(final_procs)  # <-- SHUFFLING
        proc_str = '; '.join(final_procs)
        narrative += "\n\t"+f"{proc_str}".strip()
    else:
        narrative += "\n\t"+"No procedures recorded".strip()

    return narrative

def full_narrative(row):
    narrative = f"What is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    current_visit = row['landmark_visit']
    type = row['admission_category']
    narrative += f"Today is the {current_visit} visit and the visit type is: {type}"
    max_visit = int(row['landmark_visit'])

    # if row['days_since_previous_visit'] != -1:
    #     narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    narrative += "\nDiagnosis history:"
    for past_visit, diags in reversed(list(ast.literal_eval(row['diag_per_visit']).items())):
        if past_visit == max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(diags)}".strip()
        else:
            narrative += "\n\t" + f"{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit -1])} days ago: {'; '.join(diags)}".strip()

    narrative += "\nPrescriptions history:"
    for past_visit, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
        if past_visit ==  max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(meds)}\n".strip()
        else:
            narrative += "\n\t" + f"{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1])} days ago: {'; '.join(meds)}".strip()

    narrative += "\nProcedures history:"
    for past_visit, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
        if past_visit == max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(proc)}\n".strip()
        else:
            narrative += "\n\t" + f"{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1])} days ago: {'; '.join(proc)}".strip()
    
    return narrative

def full_narrative_no_time(row):
    narrative = f"What is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?"

    max_visit = int(row['landmark_visit'])

    # if row['days_since_previous_visit'] != -1:
    #     narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    narrative += "\nDiagnosis history:"
    subnarrative = "\n\t"
    for _, diags in reversed(list(ast.literal_eval(row['diag_per_visit']).items())):
        subnarrative += f"{'; '.join(diags)}".strip()
    narrative += subnarrative

    subnarrative = "\n\t"
    narrative += "\nPrescriptions history:"
    for _, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
        subnarrative += f"{'; '.join(meds)}".strip()
    narrative += subnarrative

    subnarrative = "\n\t"
    narrative += "\nProcedures history:"
    for _, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
        subnarrative += f"{'; '.join(proc)}".strip()
    narrative += subnarrative

    return narrative

def full_narrative_no_time_rnd(row):
    narrative = f"What is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?"

    # Diagnoses
    all_diags = []
    diag_dict = ast.literal_eval(row['diag_per_visit'])
    for diags in diag_dict.values():
        all_diags.extend(diags)
    random.shuffle(all_diags)
    narrative += "\nDiagnosis history:"
    narrative += "\n\t"+'; '.join(all_diags).strip()

    # Prescriptions
    all_meds = []
    meds_dict = ast.literal_eval(row['meds_per_visit'])
    for meds in meds_dict.values():
        all_meds.extend(meds)
    random.shuffle(all_meds)
    narrative += "\nPrescriptions history:" 
    narrative += "\n\t" + '; '.join(all_meds).strip()

    # Procedures
    all_procs = []
    proc_dict = ast.literal_eval(row['proc_per_visit'])
    for procs in proc_dict.values():
        all_procs.extend(procs)
    random.shuffle(all_procs)
    narrative += "\nProcedures history:" 
    narrative += "\n\t" + '; '.join(all_procs).strip()

    return narrative

def compact_narrative_humanstyle_prompt(row):
    
    narrative = f"A {num2words(row['age_at_landmark'])}-years-old {row['gender']} patient is currently hospitalized for a {row['admission_category'].lower()} admission (visit number {num2words(row['landmark_visit'])})."
    narrative += "\nGiven the following clinical history, what is the probability of death in the next ninety days?"

    # Temporal hint if available
    if row['days_since_previous_visit'] != -1:
        narrative += f"\nThe previous visit occurred {num2words(row['days_since_previous_visit'])} days ago."

    current_visit = int(row['landmark_visit'])

    # --- Diagnoses ---
    narrative += "\nDiagnosis history:"
    diag_per_visit = ast.literal_eval(row['diag_per_visit'])
    all_diags = [d for visit_diags in diag_per_visit.values() for d in visit_diags]
    diag_counts = Counter(all_diags)

    chronic_diags = [d for d, c in diag_counts.items() if c >= 2]
    current_diags = diag_per_visit.get(current_visit, [])
    new_diags = [d for d in current_diags if diag_counts[d] == 1]

    if chronic_diags:
        narrative += "\n\t"+f"The patient has a chronic history of: {'; '.join(chronic_diags)}\n".strip()
    if new_diags:
        narrative += "\n\t"+f"During the current visit, new diagnoses include: {'; '.join(new_diags)}\n".strip()
    if not chronic_diags and not new_diags:
        narrative += "\n\t"+"There are no recorded diagnoses so far"

    # --- Medications ---
    narrative += "\nMedication history:"
    meds_per_visit = ast.literal_eval(row['meds_per_visit'])
    all_meds = [m for visit_meds in meds_per_visit.values() for m in visit_meds]
    med_counts = Counter(all_meds)

    chronic_meds = [m for m, c in med_counts.items() if c >= 2]
    current_meds = meds_per_visit.get(current_visit, [])
    new_meds = [m for m in current_meds if med_counts[m] == 1]

    if chronic_meds:
        narrative += "\n\t"+f"This patient was repeatedly prescribed these medications: {'; '.join(chronic_meds)}\n".strip()
    if new_meds:
        narrative += "\n\t"+f"New medications prescribed in this visit are: {'; '.join(new_meds)}\n".strip()
    if not chronic_meds and not new_meds:
        narrative += "\n\t"+"No medications have been prescribed so far"

    # --- Procedures ---
    narrative += "\nProcedures history:"
    proc_per_visit = ast.literal_eval(row['proc_per_visit'])
    all_procs = [p for visit_procs in proc_per_visit.values() for p in visit_procs]
    proc_counts = Counter(all_procs)

    chronic_procs = [p for p, c in proc_counts.items() if c >= 2]
    current_procs = proc_per_visit.get(current_visit, [])
    new_procs = [p for p in current_procs if proc_counts[p] == 1]

    if chronic_procs:
        narrative += "\n\t"+f"The patient has undergone repeated procedures such as: {'; '.join(chronic_procs)}\n".strip()
    if new_procs:
        narrative += "\n\t"+f"New procedures during this visit include: {'; '.join(new_procs)}\n".strip()
    if not chronic_procs and not new_procs:
        narrative += "\n\t"+"No procedures have been recorded"

    return narrative

def semi_full_narrative(row, to_split='\n'):
    """Diagnoses not for all visit, but just the set of them. All the rest is the same as full_narrative"""
    narrative = f"What is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    current_visit = row['landmark_visit']
    type = row['admission_category']
    narrative += f"Today is the {current_visit} visit and the visit type is: {type}"
    max_visit = int(row['landmark_visit'])

    # if row['days_since_previous_visit'] != -1:
    #     narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    narrative += "\nDiagnosis history:"
    for past_visit, diags in reversed(list(ast.literal_eval(row['diag_per_visit']).items())):
        if past_visit == max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(diags)}".strip()
        else:
            narrative += "\n\t" + f"{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit -1])} days ago: {'; '.join(diags)}".strip()

    narrative += "\nPrescriptions history:"
    # for past_visit, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
    #     if past_visit ==  max_visit:
    #         narrative += "\n\t" + f"Today: {'; '.join(meds)}\n".strip()
    #     else:
    #         narrative += "\n\t" + f"{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1])} days ago: {'; '.join(meds)}".strip()
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f'\n{"; ".join(set(row["med_text"].split(f"{to_split}")))}'

    narrative += "\nProcedures history:"
    # for past_visit, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
    #     if past_visit == max_visit:
    #         narrative += "\n\t" + f"Today: {'; '.join(proc)}\n".strip()
    #     else:
    #         narrative += "\n\t" + f"{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1])} days ago: {'; '.join(proc)}".strip()
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f'\n{"; ".join(set(row["proc_text"].split(f"{to_split}")))}'
    
    return narrative

def reversed_naive_narrative_prompt(row):
    narrative = 'Based on this information, what is the probability of mortality within 90 days?'
    narrative += f"\nPatient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the patient history till visit {row['landmark_visit']}."

    max_visit = int(row['landmark_visit'])

    for visit in reversed(range(1, max_visit+1)):
        narrative += f"\nVisit {visit}:"
        narrative += f"\n\tDiagnosis: {'; '.join(ast.literal_eval(row['diag_per_visit'])[visit])}"
        narrative += f"\n\tMedications: {'; '.join(ast.literal_eval(row['meds_per_visit'])[visit])}"
        narrative += f"\n\tProcedures: {'; '.join(ast.literal_eval(row['proc_per_visit'])[visit])}"
    
    return narrative

def reversed_time_naive_narrative_prompt(row):
    narrative = 'Based on this information, what is the probability of mortality within 90 days?'
    narrative += f"\nPatient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the patient history till visit {row['landmark_visit']}."

    max_visit = int(row['landmark_visit'])

    for visit in reversed(range(1, max_visit+1)):
        if visit == max_visit:
            narrative += "\nToday:"
        else:
            narrative += f"\n{int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[visit - 1])} days ago"
        
        narrative += f"\n\tDiagnosis: {'; '.join(ast.literal_eval(row['diag_per_visit'])[visit])}"
        narrative += f"\n\tMedications: {'; '.join(ast.literal_eval(row['meds_per_visit'])[visit])}"
        narrative += f"\n\tProcedures: {'; '.join(ast.literal_eval(row['proc_per_visit'])[visit])}"
    
    return narrative

def last_info_prompt(row):
    narrative = 'Based on this information, what is the probability of mortality within 90 days?'
    narrative += f"\nPatient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the patient history till visit {row['landmark_visit']}."

    visit = int(row['landmark_visit'])
        
    narrative += "\nToday:"
    
    narrative += f"\n\tDiagnosis: {'; '.join(ast.literal_eval(row['diag_per_visit'])[visit])}"
    narrative += f"\n\tMedications: {'; '.join(ast.literal_eval(row['meds_per_visit'])[visit])}"
    narrative += f"\n\tProcedures: {'; '.join(ast.literal_eval(row['proc_per_visit'])[visit])}"

    return narrative

def full_narrative_num2words(row):
    narrative = f"What is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?\n"
    current_visit = row['landmark_visit']
    type = row['admission_category']
    narrative += f"Today is the {num2words(current_visit)} visit and the visit type is: {type}"
    max_visit = int(row['landmark_visit'])

    # if row['days_since_previous_visit'] != -1:
    #     narrative += f"Last visit happened {row['days_since_previous_visit']} days ago."

    narrative += "\nDiagnosis history:"
    for past_visit, diags in reversed(list(ast.literal_eval(row['diag_per_visit']).items())):
        if past_visit == max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(diags)}".strip()
        else:
            narrative += "\n\t" + f"{num2words(int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit -1]))} days ago: {'; '.join(diags)}".strip()

    narrative += "\nPrescriptions history:"
    for past_visit, meds in reversed(ast.literal_eval(row['meds_per_visit']).items()):
        if past_visit ==  max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(meds)}\n".strip()
        else:
            narrative += "\n\t" + f"{num2words(int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1]))} days ago: {'; '.join(meds)}".strip()

    narrative += "\nProcedures history:"
    for past_visit, proc in reversed(ast.literal_eval(row['proc_per_visit']).items()):
        if past_visit == max_visit:
            narrative += "\n\t" + f"Today: {'; '.join(proc)}\n".strip()
        else:
            narrative += "\n\t" + f"{num2words(int(ast.literal_eval(row['days_since_last_visit_cumulate_sum'])[past_visit-1]))} days ago: {'; '.join(proc)}".strip()
    
    return narrative

def temporal_causal_prompt(row):
    """
    Prompt optimized for temporal causal learning
    Based on naive_narrative_prompt but structured for next-token prediction
    """
    prompt = f"Patient: {row['age_at_landmark']}-year-old {row['gender']}."
    prompt += f" Medical history over {row['landmark_visit']} visits:"
    
    max_visit = int(row['landmark_visit'])
    
    for visit in range(1, max_visit + 1):
        prompt += f" Visit-{visit}:"
        
        # Diagnoses
        diagnoses = ast.literal_eval(row['diag_per_visit'])[visit]
        if diagnoses:
            prompt += f" diagnoses: {', '.join(diagnoses)}."
        
        # Medications  
        medications = ast.literal_eval(row['meds_per_visit'])[visit]
        if medications:
            prompt += f" medications: {', '.join(medications)}."
            
        # Procedures
        procedures = ast.literal_eval(row['proc_per_visit'])[visit]
        if procedures:
            prompt += f" procedures: {', '.join(procedures)}."
    
    # Final target the model must learn to predict
    prompt += " Clinical assessment: 90-day mortality risk is"
    
    return prompt
