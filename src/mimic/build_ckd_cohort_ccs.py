"""
Build CKD → ESRD cohort with CCS-mapped codes.

Same cohort logic as build_ckd_cohort.py but tokens are CCS category codes
(~293 categories) instead of 3-char ICD codes (~2198).
Unmapped codes → "OTHER" token.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "mimic-iv" / "hosp"
OUT = ROOT / "data" / "processed"

# ── 1. Load data ────────────────────────────────────────────────────────────

print("Loading tables...")
diag = pd.read_csv(RAW / "diagnoses_icd.csv", dtype={"icd_code": str, "icd_version": int})
adm = pd.read_csv(RAW / "admissions.csv", parse_dates=["admittime", "dischtime"])
ccs_df = pd.read_csv(ROOT / "codes" / "CCS_DX_mapping.csv")

print(f"  Diagnoses: {len(diag):,} rows")

# ── 2. Build CCS lookup ────────────────────────────────────────────────────

vocab_map = {9: "ICD9CM", 10: "ICD10CM"}
ccs_lookup = {}
for _, row in ccs_df.iterrows():
    ccs_lookup[(str(row["code"]), row["vocabulary_id"])] = f"CCS_{int(row['category_code'])}"

def get_ccs(icd_code, icd_version):
    key = (icd_code, vocab_map[icd_version])
    return ccs_lookup.get(key, "OTHER")

# ── 3. Code classification (same as original) ──────────────────────────────

def is_ckd(code, version):
    if version == 10:
        return bool(pd.notna(code) and code.startswith("N18") and code not in ("N186", "N189"))
    elif version == 9:
        return bool(pd.notna(code) and code.startswith("585") and code not in ("5856", "5859"))
    return False

def is_esrd(code, version):
    if version == 10:
        return code.startswith("N186") or code.startswith("Z992") or code.startswith("Z49")
    elif version == 9:
        return code.startswith("5856") or code.startswith("V451") or code.startswith("V56")
    return False

diag["is_ckd"] = [is_ckd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
diag["is_esrd"] = [is_esrd(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]
diag["ccs"] = [get_ccs(c, v) for c, v in zip(diag["icd_code"], diag["icd_version"])]

ckd_patients = set(diag.loc[diag["is_ckd"], "subject_id"])
esrd_patients = set(diag.loc[diag["is_esrd"], "subject_id"])
print(f"CKD patients: {len(ckd_patients):,}")

# ── 4. Merge admission times & sort ─────────────────────────────────────────

diag = diag.merge(adm[["hadm_id", "admittime"]], on="hadm_id", how="left")
diag = diag.sort_values(["subject_id", "admittime", "seq_num"]).reset_index(drop=True)

# ── 5. Build sequences ──────────────────────────────────────────────────────

print("Building patient sequences...")

records = []
excluded_same_admission = 0
excluded_short = 0

for sid in sorted(ckd_patients):
    pdf = diag[diag["subject_id"] == sid]

    ckd_rows = pdf[pdf["is_ckd"]]
    first_ckd_hadm = ckd_rows.iloc[0]["hadm_id"]

    esrd_rows = pdf[pdf["is_esrd"]]

    if len(esrd_rows) > 0 and sid in esrd_patients:
        first_esrd_hadm = esrd_rows.iloc[0]["hadm_id"]
        first_esrd_admittime = esrd_rows.iloc[0]["admittime"]

        if first_esrd_hadm == first_ckd_hadm:
            excluded_same_admission += 1
            continue

        first_ckd_admittime = ckd_rows.iloc[0]["admittime"]
        if first_esrd_admittime <= first_ckd_admittime:
            excluded_same_admission += 1
            continue

        seq = pdf[pdf["admittime"] < first_esrd_admittime]
        label = 1
    else:
        seq = pdf
        label = 0

    if len(seq) < 10:
        excluded_short += 1
        continue

    codes = seq["ccs"].tolist()
    hadm_ids = seq["hadm_id"].astype(int).tolist()

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

# ── 6. Stats ─────────────────────────────────────────────────────────────────

cohort = pd.DataFrame(records)

all_codes = set()
other_count = 0
total_count = 0
for c in cohort["codes"]:
    tokens = json.loads(c)
    all_codes.update(tokens)
    total_count += len(tokens)
    other_count += sum(1 for t in tokens if t == "OTHER")

print(f"\n{'='*60}")
print(f"FINAL COHORT (CCS): {len(cohort):,} patients")
print(f"  Progressors (Y=1): {(cohort['label']==1).sum():,}")
print(f"  Non-progressors (Y=0): {(cohort['label']==0).sum():,}")
print(f"  Class balance: {cohort['label'].mean():.4f}")
print(f"\nAlphabet size: {len(all_codes)} CCS categories")
print(f"  (includes OTHER for unmapped codes)")
print(f"  OTHER tokens: {other_count:,} / {total_count:,} ({other_count/total_count*100:.2f}%)")
print(f"\nSequence length distribution:")
print(cohort["seq_len"].describe().to_string())

# ── 7. Save ──────────────────────────────────────────────────────────────────

cohort.to_csv(OUT / "ckd_cohort_ccs.csv", index=False)
print(f"\nSaved to {OUT / 'ckd_cohort_ccs.csv'}")

stats_lines = [
    "CKD → ESRD Cohort Statistics (CCS-mapped)",
    "=" * 45,
    f"Total patients: {len(cohort):,}",
    f"Progressors (Y=1): {(cohort['label']==1).sum():,}",
    f"Non-progressors (Y=0): {(cohort['label']==0).sum():,}",
    f"Class balance (frac Y=1): {cohort['label'].mean():.4f}",
    "",
    f"Alphabet size: {len(all_codes)} CCS categories",
    f"OTHER tokens: {other_count:,} / {total_count:,} ({other_count/total_count*100:.2f}%)",
    "",
    "Sequence lengths:",
    cohort["seq_len"].describe().to_string(),
    "",
    "Admissions per patient:",
    cohort["n_admissions"].describe().to_string(),
    "",
    f"Excluded (same-admission onset): {excluded_same_admission}",
    f"Excluded (< 10 codes): {excluded_short}",
]
(OUT / "ckd_cohort_ccs_stats.txt").write_text("\n".join(stats_lines))
print(f"Saved stats to {OUT / 'ckd_cohort_ccs_stats.txt'}")
