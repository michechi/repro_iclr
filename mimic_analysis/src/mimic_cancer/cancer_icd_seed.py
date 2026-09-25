from __future__ import annotations

import pandas as pd


def icd_cancer_seed(code: str | None, version: int) -> bool:
    c = (code or "").strip()
    if not c:
        return False
    if version == 10:
        return c.startswith("C") or c.startswith("D0")
    if version == 9 and len(c) >= 3 and c[:3].isdigit():
        return 140 <= int(c[:3]) <= 239
    return False


def icd_cancer_seed_mask(df: pd.DataFrame) -> pd.Series:
    """Vectorized seed mask. Expects columns `icd_code`, `icd_version`.

    MIMIC-IV codes are space-padded; we strip before checking.
    """
    code = df["icd_code"].fillna("").astype("string").str.strip()
    version = df["icd_version"].astype("int8")

    icd10_mask = (version == 10) & (code.str.startswith("C") | code.str.startswith("D0"))

    icd9_first3 = code.where(version == 9, other="").str[:3]
    icd9_numeric = icd9_first3.str.isdigit().fillna(False)
    # Convert the first 3 digits to int where they are numeric
    as_int = pd.to_numeric(icd9_first3.where(icd9_numeric, other="0"), errors="coerce")
    icd9_mask = (version == 9) & icd9_numeric & as_int.between(140, 239)

    return (icd10_mask | icd9_mask).fillna(False)
