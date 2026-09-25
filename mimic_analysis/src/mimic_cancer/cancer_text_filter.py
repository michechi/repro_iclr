from __future__ import annotations

import re

import pandas as pd


# Word-boundary aware so the prefilter does not match substrings
# (e.g. "protect" -> ct, "mrs" -> mr, "petechiae" -> pet).
CANCER_TERM = re.compile(
    r"\b(?:cancer|cancers|restaging|malignan\w*)\b",
    flags=re.IGNORECASE,
)

MODALITY_TERM = re.compile(
    r"\b(?:ct|mri?|pet(?:/ct)?|nm|mammo\w*)\b",
    flags=re.IGNORECASE,
)


def text_cancer_seed(text: str | None) -> bool:
    if not text:
        return False
    return bool(CANCER_TERM.search(text) and MODALITY_TERM.search(text))


def text_cancer_seed_series(s: pd.Series) -> pd.Series:
    """Vectorized version — much faster than `s.map(text_cancer_seed)`."""
    has_cancer = s.str.contains(CANCER_TERM, na=False, regex=True)
    has_modality = s.str.contains(MODALITY_TERM, na=False, regex=True)
    return has_cancer & has_modality
