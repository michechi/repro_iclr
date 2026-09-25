from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .dfci_imaging_model import LABEL_ORDER


FLAG_COLS = [f"flag_{label}" for label in LABEL_ORDER]
PROB_COLS = [f"prob_{label}" for label in LABEL_ORDER]


def compile_modality_map(raw: Iterable[dict]) -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for entry in raw:
        compiled.append((re.compile(entry["pattern"], re.IGNORECASE), entry["group"]))
    return compiled


def map_modality(exam_name, modality_map: list[tuple[re.Pattern[str], str]]) -> str:
    if exam_name is None or pd.isna(exam_name) or not exam_name:
        return "OTHER"
    name = str(exam_name)
    for pattern, group in modality_map:
        if pattern.search(name):
            return group
    return "OTHER"


def merge_addenda(
    preds: pd.DataFrame,
    detail_wide: pd.DataFrame,
) -> pd.DataFrame:
    """OR child addendum flags into parent predictions, take elementwise max of probs.

    Adds `has_addendum` and `is_orphan_addendum` boolean columns.
    Keeps orphan addenda (parent_note_id points outside `preds`) as standalone rows.
    Deduplicates on note_id at the end.
    """
    preds = preds.copy()
    if "has_addendum" not in preds.columns:
        preds["has_addendum"] = False
    if "is_orphan_addendum" not in preds.columns:
        preds["is_orphan_addendum"] = False

    if "parent_note_id" not in detail_wide.columns:
        return preds

    children = detail_wide[detail_wide["parent_note_id"].notna()][
        ["note_id", "parent_note_id"]
    ]
    if children.empty:
        return preds

    pred_ids = set(preds["note_id"].tolist())
    # Split children into merge-able and orphans.
    children = children.copy()
    children["parent_in_preds"] = children["parent_note_id"].isin(pred_ids)
    children["child_in_preds"] = children["note_id"].isin(pred_ids)

    merge_links = children[children["parent_in_preds"] & children["child_in_preds"]]
    orphan_links = children[~children["parent_in_preds"] & children["child_in_preds"]]

    if not merge_links.empty:
        preds_idx = preds.set_index("note_id")
        child_preds = preds_idx.loc[merge_links["note_id"].tolist()]
        parents_idx = merge_links["parent_note_id"].tolist()

        for i, parent_id in enumerate(parents_idx):
            child = child_preds.iloc[i]
            for col in FLAG_COLS:
                if col in preds_idx.columns:
                    preds_idx.at[parent_id, col] = int(
                        max(int(preds_idx.at[parent_id, col]), int(child[col]))
                    )
            for col in PROB_COLS:
                if col in preds_idx.columns:
                    preds_idx.at[parent_id, col] = max(
                        float(preds_idx.at[parent_id, col]), float(child[col])
                    )
            preds_idx.at[parent_id, "has_addendum"] = True

        # Drop the child rows that have been merged into parents.
        preds_idx = preds_idx.drop(index=merge_links["note_id"].tolist(), errors="ignore")
        preds = preds_idx.reset_index()

    if not orphan_links.empty:
        orphan_ids = set(orphan_links["note_id"].tolist())
        preds.loc[preds["note_id"].isin(orphan_ids), "is_orphan_addendum"] = True

    preds = preds.drop_duplicates(subset="note_id", keep="first").reset_index(drop=True)
    return preds


def build_note_level_table(
    preds: pd.DataFrame,
    radiology_meta: pd.DataFrame,
    detail_wide: pd.DataFrame,
    admissions: pd.DataFrame,
    modality_map: list[tuple[re.Pattern[str], str]],
    min_retained_notes_per_patient: int = 2,
) -> pd.DataFrame:
    """Join predictions to radiology metadata, compute sequence features,
    keep only cancer-positive notes, and drop patients below the minimum count.

    `preds` is expected to already carry `subject_id`, `hadm_id`, `event_time`,
    `charttime`, `storetime` from step 3. From `radiology_meta` we only add
    `note_type` (the rest of the metadata columns would collide).
    """
    meta_small = radiology_meta[["note_id", "note_type"]].drop_duplicates("note_id")
    merged = preds.merge(meta_small, on="note_id", how="left")

    if "event_time" not in merged.columns:
        merged["event_time"] = merged["charttime"].fillna(merged["storetime"])
    merged = merged.dropna(subset=["event_time"])

    # Keep only notes with any-cancer positive
    merged = merged[merged["flag_any_cancer"].astype(int) == 1].copy()
    if merged.empty:
        return merged

    # Attach modality via radiology_detail.exam_name
    detail_small = detail_wide[["note_id", "exam_name"]].drop_duplicates("note_id")
    merged = merged.merge(detail_small, on="note_id", how="left")
    merged["modality_group"] = merged["exam_name"].map(
        lambda name: map_modality(name, modality_map)
    )

    # Sort within patient and compute sequence features
    merged = merged.sort_values(
        ["subject_id", "event_time", "storetime", "note_id"]
    ).reset_index(drop=True)
    merged["seq_time_rank"] = merged.groupby("subject_id").cumcount()
    deltas = (
        merged.groupby("subject_id")["event_time"]
        .diff()
        .dt.total_seconds()
        .div(86400.0)
        .fillna(0.0)
    )
    merged["delta_days_from_prev_note"] = deltas.astype(np.float32)

    # Admission index within patient (chronological order of admissions)
    adm_small = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    adm_small = adm_small.sort_values(["subject_id", "admittime"])
    adm_small["admission_index"] = (
        adm_small.groupby("subject_id").cumcount().astype("int32")
    )
    merged = merged.merge(
        adm_small[["hadm_id", "admission_index", "dischtime"]],
        on="hadm_id",
        how="left",
    )

    merged["n_positive_labels"] = merged[FLAG_COLS].sum(axis=1).astype("int16")
    merged["source"] = "radiology"

    counts = merged.groupby("subject_id").size()
    keep_subjects = counts[counts >= min_retained_notes_per_patient].index
    merged = merged[merged["subject_id"].isin(keep_subjects)].reset_index(drop=True)

    return merged


def _patient_first_event(notes: pd.DataFrame, flag_col: str) -> pd.DataFrame:
    mask = notes[flag_col].astype(int) == 1
    sub = notes.loc[mask, ["subject_id", "event_time"]]
    return (
        sub.groupby("subject_id", as_index=False)
        .min()
        .rename(columns={"event_time": f"t_{flag_col}"})
    )


def build_dataset_a(
    notes: pd.DataFrame,
    strict: bool = True,
) -> dict[str, pd.DataFrame]:
    """Dataset A — order-only cancer trajectory task.

    When `strict=True` (the primary strict cohort) each patient must
    have *exactly* one progression-positive note and one response-positive
    note before `t_cut`. When `strict=False` the relaxed Sensitivity A cohort
    allows multiple progression/response notes, but the label is still based
    on the first-occurrence order.

    Returns a dict with keys: labels, seq_context, seq_core, bag_context, last_event.
    """
    if notes.empty:
        empty = pd.DataFrame()
        return {
            "labels": empty,
            "seq_context": empty,
            "seq_core": empty,
            "bag_context": empty,
            "last_event": empty,
        }

    t_prog = _patient_first_event(notes, "flag_progression").rename(
        columns={"t_flag_progression": "t_prog1"}
    )
    t_resp = _patient_first_event(notes, "flag_response").rename(
        columns={"t_flag_response": "t_resp1"}
    )

    both = t_prog.merge(t_resp, on="subject_id", how="inner")
    both = both[both["t_prog1"] != both["t_resp1"]].copy()
    both["t_cut"] = both[["t_prog1", "t_resp1"]].max(axis=1)
    both["y_order"] = (both["t_prog1"] < both["t_resp1"]).astype("int8")

    # Restrict to events <= t_cut per patient
    sub = notes.merge(both[["subject_id", "t_cut"]], on="subject_id", how="inner")
    sub = sub[sub["event_time"] <= sub["t_cut"]].copy()

    # Count progression/response notes per patient in the window.
    counts_prog = (
        sub.assign(
            _p=(sub["flag_progression"].astype(int) == 1).astype(int),
            _r=(sub["flag_response"].astype(int) == 1).astype(int),
        )
        .groupby("subject_id")[["_p", "_r"]]
        .sum()
        .rename(columns={"_p": "n_progression_notes_up_to_cut", "_r": "n_response_notes_up_to_cut"})
        .reset_index()
    )

    labels = both.merge(counts_prog, on="subject_id", how="left").fillna(
        {"n_progression_notes_up_to_cut": 0, "n_response_notes_up_to_cut": 0}
    )
    labels["strict_primary_subset"] = (
        (labels["n_progression_notes_up_to_cut"] == 1)
        & (labels["n_response_notes_up_to_cut"] == 1)
    ).astype("int8")
    if strict:
        labels = labels[labels["strict_primary_subset"] == 1].copy()

    sub = sub[sub["subject_id"].isin(labels["subject_id"])].copy()
    n_ctx = sub.groupby("subject_id").size().rename("n_notes_used_context").reset_index()
    labels = labels.merge(n_ctx, on="subject_id", how="left")
    labels["split"] = "unassigned"

    # Build seq_context
    seq_context = sub.sort_values(
        ["subject_id", "event_time", "storetime", "note_id"]
    ).copy()
    seq_context["seq_idx"] = seq_context.groupby("subject_id").cumcount()
    seq_context = seq_context[
        ["subject_id", "seq_idx", "note_id", "hadm_id", "event_time", "delta_days_from_prev_note", "modality_group", "n_positive_labels"]
        + FLAG_COLS
        + PROB_COLS
    ]

    # Build seq_core: exactly the two strict notes
    core_rows = []
    for _, row in labels.iterrows():
        subj = row["subject_id"]
        sp = sub[
            (sub["subject_id"] == subj)
            & (sub["flag_progression"].astype(int) == 1)
        ].head(1)
        sr = sub[
            (sub["subject_id"] == subj)
            & (sub["flag_response"].astype(int) == 1)
        ].head(1)
        if not sp.empty:
            core_rows.append(sp.iloc[0])
        if not sr.empty:
            core_rows.append(sr.iloc[0])
    seq_core = pd.DataFrame(core_rows) if core_rows else sub.iloc[0:0].copy()
    if not seq_core.empty:
        seq_core = seq_core.sort_values(["subject_id", "event_time", "note_id"]).reset_index(drop=True)
        seq_core["seq_idx"] = seq_core.groupby("subject_id").cumcount()
        seq_core = seq_core[
            ["subject_id", "seq_idx", "note_id", "hadm_id", "event_time", "delta_days_from_prev_note", "modality_group", "n_positive_labels"]
            + FLAG_COLS
            + PROB_COLS
        ]

    # Bag-of-events baseline
    bag = sub.groupby("subject_id").agg(
        n_notes=("note_id", "count"),
        **{f"cnt_{label}": (f"flag_{label}", "sum") for label in LABEL_ORDER},
        **{f"mean_{label}": (f"prob_{label}", "mean") for label in LABEL_ORDER},
        **{f"max_{label}": (f"prob_{label}", "max") for label in LABEL_ORDER},
    ).reset_index()

    last_event = (
        sub.sort_values(["subject_id", "event_time", "storetime", "note_id"])
        .groupby("subject_id")
        .tail(1)
        .reset_index(drop=True)
    )
    last_event_cols = ["subject_id", "note_id", "event_time", "modality_group", "n_positive_labels"] + FLAG_COLS + PROB_COLS
    last_event = last_event[last_event_cols].copy()

    return {
        "labels": labels,
        "seq_context": seq_context,
        "seq_core": seq_core,
        "bag_context": bag,
        "last_event": last_event,
    }


def build_dataset_b(
    notes: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Dataset B — 1-year mortality anchored on last cancer-related note's admission."""
    if notes.empty:
        return {
            "labels": pd.DataFrame(),
            "seq": pd.DataFrame(),
            "bag": pd.DataFrame(),
            "last_event": pd.DataFrame(),
        }

    adm_small = admissions[["hadm_id", "dischtime"]].drop_duplicates("hadm_id")
    notes = notes.merge(adm_small, on="hadm_id", how="left", suffixes=("", "_adm"))
    if "dischtime_adm" in notes.columns:
        notes = notes.drop(columns=["dischtime"]).rename(columns={"dischtime_adm": "dischtime"})

    # Leakage guard: drop notes whose event_time is after the admission's dischtime
    notes = notes[notes["dischtime"].notna()].copy()
    notes = notes[notes["event_time"] <= notes["dischtime"]].copy()

    # Pick each patient's last retained note and set hadm_last, t0 = dischtime(hadm_last)
    sorted_notes = notes.sort_values(
        ["subject_id", "event_time", "storetime", "note_id"]
    )
    last = sorted_notes.groupby("subject_id").tail(1).reset_index(drop=True)
    last = last[["subject_id", "hadm_id", "dischtime"]].rename(
        columns={"hadm_id": "hadm_last", "dischtime": "t0"}
    )

    pat_small = patients[["subject_id", "dod"]].copy()
    labels = last.merge(pat_small, on="subject_id", how="left")
    # Filter out rows with missing t0 (already guarded, but defensive).
    labels = labels.dropna(subset=["t0"]).copy()
    horizon = labels["t0"] + pd.Timedelta(days=365)
    labels["y_death_365"] = (
        labels["dod"].notna() & (labels["dod"] <= horizon)
    ).astype("int8")
    days_to_event = (labels["dod"] - labels["t0"]).dt.days
    labels["days_to_death_or_censor"] = days_to_event.where(
        labels["dod"].notna(), other=365
    ).clip(upper=365).astype("int32")
    labels = labels.drop(columns=["dod"])

    seq = notes.merge(labels[["subject_id", "t0"]], on="subject_id", how="inner")
    seq = seq[seq["event_time"] <= seq["t0"]].copy()
    seq = seq.sort_values(["subject_id", "event_time", "storetime", "note_id"])
    seq["seq_idx"] = seq.groupby("subject_id").cumcount()
    seq_cols = [
        "subject_id",
        "seq_idx",
        "note_id",
        "hadm_id",
        "event_time",
        "delta_days_from_prev_note",
        "modality_group",
        "n_positive_labels",
    ] + FLAG_COLS + PROB_COLS
    seq = seq[seq_cols]

    n_notes_used = seq.groupby("subject_id").size().rename("n_notes_used").reset_index()
    n_adm_used = (
        seq.groupby("subject_id")["hadm_id"].nunique().rename("n_admissions").reset_index()
    )
    first_last = seq.groupby("subject_id").agg(
        first_note_time=("event_time", "min"),
        last_note_time=("event_time", "max"),
    ).reset_index()

    labels = (
        labels.merge(n_notes_used, on="subject_id", how="left")
        .merge(n_adm_used, on="subject_id", how="left")
        .merge(first_last, on="subject_id", how="left")
    )
    labels["observation_window_days"] = (
        (labels["last_note_time"] - labels["first_note_time"]).dt.days.fillna(0).astype("int32")
    )
    labels["split"] = "unassigned"

    # Bag features
    bag = seq.groupby("subject_id").agg(
        n_notes=("note_id", "count"),
        **{f"cnt_{label}": (f"flag_{label}", "sum") for label in LABEL_ORDER},
        **{f"mean_{label}": (f"prob_{label}", "mean") for label in LABEL_ORDER},
        **{f"max_{label}": (f"prob_{label}", "max") for label in LABEL_ORDER},
    ).reset_index()

    last_event_cols = ["subject_id", "note_id", "event_time", "modality_group", "n_positive_labels"] + FLAG_COLS + PROB_COLS
    # `seq` already has `seq_idx` in chronological order, so we don't need storetime here.
    last_event = (
        seq.sort_values(["subject_id", "seq_idx"])
        .groupby("subject_id")
        .tail(1)
        .reset_index(drop=True)[last_event_cols]
    )

    return {
        "labels": labels,
        "seq": seq,
        "bag": bag,
        "last_event": last_event,
    }
