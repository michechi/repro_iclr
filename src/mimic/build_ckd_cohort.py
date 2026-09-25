"""
Build CKD → ESRD cohort from MIMIC-IV for sequential prediction experiments.

Cohort definition:
  - Inclusion: patients with at least one CKD code (ICD-10 N18.1–N18.5, ICD-9 585.1–585.5)
  - Label Y=1: ESRD codes appear AFTER the first CKD code (progressor)
  - Label Y=0: no ESRD codes ever (non-progressor)
  - Sequences: all diagnosis codes ordered by (admittime, seq_num), truncated before first ESRD for Y=1
  - Filtering: ≥10 codes after flattening; exclude same-admission CKD+ESRD onset

Output: data/processed/ckd_cohort.parquet
        data/processed/ckd_cohort_stats.txt
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mimic-iv" / "hosp"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Load data ────────────────────────────────────────────────────────────

print("Loading tables...")
diag = pd.read_csv(RAW / "diagnoses_icd.csv", dtype={"icd_code": str, "icd_version": int})
adm = pd.read_csv(RAW / "admissions.csv", parse_dates=["admittime", "dischtime"])
pat = pd.read_csv(RAW / "patients.csv")

print(f"  Diagnoses: {len(diag):,} rows")
print(f"  Admissions: {len(adm):,} rows")
print(f"  Patients: {len(pat):,} unique patients")


# ── 2. Define code sets ─────────────────────────────────────────────────────

def is_ckd(code, version):
    """CKD stages 1-5 (excludes stage 6 / ESRD)."""
    if version == 10:
        return bool(pd.notna(code) and code.startswith("N18") and code not in ("N186", "N189"))
    elif version == 9:
        return bool(pd.notna(code) and code.startswith("585") and code not in ("5856", "5859"))
    return False

def is_esrd(code, version):
    """ESRD / dialysis / transplant codes."""
    if version == 10:
        if code.startswith("N186") or code.startswith("Z992") or code.startswith("Z49"):
            return True
    elif version == 9:
        if code.startswith("5856") or code.startswith("V451") or code.startswith("V56"):
            return True
    return False

# Vectorized flags
diag["is_ckd"] = [is_ckd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
diag["is_esrd"] = [is_esrd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]

ckd_patients = set(diag.loc[diag["is_ckd"], "subject_id"])
esrd_patients = set(diag.loc[diag["is_esrd"], "subject_id"])
print(f"\nCKD patients: {len(ckd_patients):,}")
print(f"ESRD patients (all): {len(esrd_patients):,}")


# ── 3. Merge admission times & sort ─────────────────────────────────────────

diag = diag.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
diag = diag.sort_values(["subject_id", "admittime", "seq_num"]).reset_index(drop=True)


# ── 4. For each CKD patient, build sequence and assign label ────────────────

print("\nBuilding patient sequences...")

records = []
excluded_same_admission = 0
excluded_short = 0

for sid in sorted(ckd_patients):
    pdf = diag[diag["subject_id"] == sid].copy()

    # Find first CKD admission
    ckd_rows = pdf[pdf["is_ckd"]]
    first_ckd_hadm = ckd_rows.iloc[0]["hadm_id"]

    # Find first ESRD row (if any)
    esrd_rows = pdf[pdf["is_esrd"]]

    if len(esrd_rows) > 0 and sid in esrd_patients:
        # Progressor candidate
        first_esrd_hadm = esrd_rows.iloc[0]["hadm_id"]
        first_esrd_admittime = esrd_rows.iloc[0]["admittime"]

        # Exclusion: ESRD at same admission as first CKD
        if first_esrd_hadm == first_ckd_hadm:
            excluded_same_admission += 1
            continue

        # Must have ESRD AFTER first CKD (by admittime)
        first_ckd_admittime = ckd_rows.iloc[0]["admittime"]
        if first_esrd_admittime <= first_ckd_admittime:
            excluded_same_admission += 1
            continue

        # Truncate sequence before first ESRD admission
        seq = pdf[pdf["admittime"] < first_esrd_admittime]
        label = 1
    else:
        # Non-progressor: keep all codes
        seq = pdf
        label = 0

    if len(seq) < 10:
        excluded_short += 1
        continue

    # Build 3-character ICD-10 level codes (or raw for ICD-9 with prefix)
    codes = []
    hadm_ids = []
    for _, row in seq.iterrows():
        c = row["icd_code"]
        v = row["icd_version"]
        if v == 10:
            token = c[:3]  # 3-character ICD-10
        else:
            token = "ICD9_" + c[:3]  # prefix ICD-9 to distinguish
        codes.append(token)
        hadm_ids.append(int(row["hadm_id"]))

    records.append({
        "subject_id": int(sid),
        "label": label,
        "seq_len": len(codes),
        "codes": json.dumps(codes),
        "hadm_ids": json.dumps(hadm_ids),
        "n_admissions": len(set(hadm_ids)),
    })

print(f"  Excluded (same-admission CKD+ESRD): {excluded_same_admission}")
print(f"  Excluded (< 10 codes): {excluded_short}")


# ── 5. Build DataFrame and save ─────────────────────────────────────────────

cohort = pd.DataFrame(records)
print(f"\n{'='*60}")
print(f"FINAL COHORT: {len(cohort):,} patients")
print(f"  Progressors (Y=1): {(cohort['label']==1).sum():,}")
print(f"  Non-progressors (Y=0): {(cohort['label']==0).sum():,}")
print(f"  Class balance: {cohort['label'].mean():.3f}")

print(f"\nSequence length distribution:")
print(cohort["seq_len"].describe().to_string())

print(f"\nAdmissions per patient:")
print(cohort["n_admissions"].describe().to_string())

# Alphabet size
all_codes = set()
for c in cohort["codes"]:
    all_codes.update(json.loads(c))
print(f"\nAlphabet size (unique 3-char codes): {len(all_codes)}")

# Save
cohort.to_csv(OUT / "ckd_cohort.csv", index=False)
print(f"\nSaved to {OUT / 'ckd_cohort.csv'}")

# Also save stats to text file
stats_lines = [
    f"CKD → ESRD Cohort Statistics",
    f"{'='*40}",
    f"Total patients: {len(cohort):,}",
    f"Progressors (Y=1): {(cohort['label']==1).sum():,}",
    f"Non-progressors (Y=0): {(cohort['label']==0).sum():,}",
    f"Class balance (frac Y=1): {cohort['label'].mean():.4f}",
    f"",
    f"Sequence lengths:",
    cohort["seq_len"].describe().to_string(),
    f"",
    f"Admissions per patient:",
    cohort["n_admissions"].describe().to_string(),
    f"",
    f"Alphabet size: {len(all_codes)}",
    f"",
    f"Excluded (same-admission onset): {excluded_same_admission}",
    f"Excluded (< 10 codes): {excluded_short}",
]
stats_text = "\n".join(stats_lines)
(OUT / "ckd_cohort_stats.txt").write_text(stats_text)
print(f"Saved stats to {OUT / 'ckd_cohort_stats.txt'}")
