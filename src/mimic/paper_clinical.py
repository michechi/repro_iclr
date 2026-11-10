"""MIMIC-IV CKD to ESRD cohort construction, summary statistics, and shortcut controls (Section 4.4 and Appendix F)."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean, median

PROTOCOL = {
    "id": "mimic_ckd_v1", "paper": "section 4.4 and Appendix F, Tables 13-15",
    "dataset": "MIMIC-IV v3.1", "minimum_codes": 10,
    "ckd": {"10": "N18.1-N18.5 including descendants", "9": "585.1-585.5"},
    "esrd": {"10": ["N18.6", "Z99.2", "Z49.*"], "9": ["585.6", "V45.1", "V56.*"]},
    "positive": "first ESRD admission time strictly after first CKD admission time",
    "negative": "CKD and no ESRD endpoint anywhere in observed record",
    "truncate_positive": "exclude entire first ESRD admission and all later admissions",
    "sequence_order": ["admittime", "hadm_id (tie-break)", "seq_num", "source_row (tie-break)"],
    "representation": "CCS tokens with repeated diagnoses retained",
    "kgram": {"k": list(range(1, 8)), "splits": 5, "train_test": [.8, .2],
              "score": "mean of training conditional probabilities over contiguous k-gram occurrences",
              "unseen_probability": .5, "f1_threshold": .5},
    "bag_of_codes": {"features": "counts, not binary presence", "split": [.6, .2, .2],
                     "models": ["LogisticRegression", "XGBoost"],
                     "threshold": "validation F1 on .10..89 by .01"},
    "shuffle": "permute every patient's complete code list, preserving multiplicities and split membership",
    "settings": {
        "split_seed": 42, "kgram_split_seeds": [42, 43, 44, 45, 46],
        "bag_logreg": {"C": 1., "class_weight": "balanced", "max_iter": 1000},
        "bag_xgboost": {"n_estimators": 200, "max_depth": 6, "learning_rate": .1,
                        "scale_pos_weight": "training negative/positive count"},
    },
}


def normalize_code(code):
    return str(code or "").strip().upper().replace(".", "")


def is_ckd(code, version):
    """Match CKD stages 1-5 (ICD-10 N18.1-N18.5 and ICD-9 585.1-585.5)."""
    code = normalize_code(code)
    return bool(re.fullmatch(r"N18[1-5][0-9]*", code) if int(version) == 10 else
                re.fullmatch(r"585[1-5]", code) if int(version) == 9 else False)


def is_esrd(code, version):
    code = normalize_code(code)
    prefixes = ("N186", "Z992", "Z49") if int(version) == 10 else (
        ("5856", "V451", "V56") if int(version) == 9 else ())
    return code.startswith(prefixes)


def read_csv(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def ccs_mapping(path, *, conflict_policy="error"):
    if conflict_policy not in {"error", "first", "last"}:
        raise ValueError("unknown CCS mapping conflict policy")
    result = {}
    for row in read_csv(path):
        key = (normalize_code(row["code"]), {"ICD9CM": 9, "ICD10CM": 10}[row["vocabulary_id"]])
        value = "CCS_" + str(int(row["category_code"]))
        if key in result and result[key] != value:
            if conflict_policy == "error":
                raise ValueError(f"ambiguous CCS mapping for {key}; select first/last explicitly")
            if conflict_policy == "first":
                continue
        result[key] = value
    return result


def build_cohort(diagnosis_rows, admission_rows, mapping, *, minimum_codes=10, unmapped="error"):
    """Build the CKD-to-ESRD cohort from diagnosis and admission iterables."""
    if minimum_codes != 10 or unmapped not in {"error", "other", "drop"}:
        raise ValueError("cohort requires minimum_codes=10 and an explicit unmapped policy")
    admissions = {}
    for row in admission_rows:
        hadm, subject = int(row["hadm_id"]), int(row["subject_id"])
        value = (subject, datetime.fromisoformat(str(row["admittime"])))
        if hadm in admissions and admissions[hadm] != value:
            raise ValueError("conflicting admission identity or time")
        admissions[hadm] = value

    def identify(row):
        hadm, subject = int(row["hadm_id"]), int(row["subject_id"])
        if hadm not in admissions or admissions[hadm][0] != subject:
            raise ValueError("diagnosis has no matching patient/admission timestamp")
        return subject, hadm, admissions[hadm][1], int(row["icd_version"])

    first_ckd, first_esrd = {}, {}
    for row in diagnosis_rows():
        subject, hadm, time, version = identify(row)
        for matched, target in ((is_ckd(row["icd_code"], version), first_ckd),
                                (is_esrd(row["icd_code"], version), first_esrd)):
            if matched and (subject not in target or (time, hadm) < target[subject]):
                target[subject] = (time, hadm)
    eligible, exclusions = {}, Counter()
    for subject, (ckd_time, ckd_hadm) in first_ckd.items():
        endpoint = first_esrd.get(subject)
        if endpoint is None:
            eligible[subject] = (0, None)
        elif endpoint[1] == ckd_hadm:
            exclusions["same_admission_onset"] += 1
        elif endpoint[0] <= ckd_time:
            exclusions["esrd_not_strictly_after_ckd"] += 1
        else:
            eligible[subject] = (1, endpoint[0])
    histories, unmapped_codes = defaultdict(list), 0
    for source_row, row in enumerate(diagnosis_rows()):
        subject, hadm, time, version = identify(row)
        if subject not in eligible:
            continue
        label, endpoint_time = eligible[subject]
        if endpoint_time is not None and time >= endpoint_time:
            continue
        token = mapping.get((normalize_code(row["icd_code"]), version))
        if token is None:
            unmapped_codes += 1
            if unmapped == "error":
                raise ValueError("unmapped diagnosis: choose other or drop")
            if unmapped == "drop":
                continue
            token = "OTHER"
        histories[subject].append((time, hadm, int(row["seq_num"]), source_row, token))
    records = []
    for subject in sorted(eligible):
        history = sorted(histories[subject])
        if len(history) < minimum_codes:
            exclusions["fewer_than_10_codes_after_mapping"] += 1
            continue
        codes = [row[-1] for row in history]
        hadm_ids = [row[1] for row in history]
        records.append({"subject_id": subject, "label": eligible[subject][0], "seq_len": len(codes),
                        "codes": codes, "hadm_ids": hadm_ids, "n_admissions": len(set(hadm_ids))})
    summary = {"patients": len(records), "progressors": sum(r["label"] for r in records),
               "observed_vocabulary_size": len({c for r in records for c in r["codes"]}),
               "unmapped_retained_history_codes": unmapped_codes, "unmapped_policy": unmapped,
               "exclusions": dict(exclusions)}
    return records, summary


def read_cohort(path):
    records = []
    for row in read_csv(path):
        record = {"subject_id": int(row["subject_id"]), "label": int(row["label"]),
                  "codes": json.loads(row["codes"])}
        if row.get("n_admissions"):
            record["n_admissions"] = int(row["n_admissions"])
        records.append(record)
    if len({r["subject_id"] for r in records}) != len(records):
        raise ValueError("cohort must have one row per patient")
    if any(r["label"] not in (0, 1) or len(r["codes"]) < 10 for r in records):
        raise ValueError("invalid binary label or short clinical history")
    return records


def cohort_statistics(records):
    """Compute Table 13 cohort summary statistics and Table 15 enriched unigrams."""
    if not records or {r["label"] for r in records} != {0, 1}:
        raise ValueError("cohort summaries require both outcome groups")
    groups = {label: [r for r in records if r["label"] == label] for label in (0, 1)}
    occurrences = {label: Counter(c for r in group for c in r["codes"]) for label, group in groups.items()}
    prevalence = {label: Counter(c for r in group for c in set(r["codes"])) for label, group in groups.items()}
    vocab = sorted(set(occurrences[0]) | set(occurrences[1]))
    a, b = ([occurrences[label][c] for c in vocab] for label in (0, 1))
    cosine = sum(x * y for x, y in zip(a, b)) / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

    def ranks(values):
        result = [0.] * len(values)
        order = sorted(range(len(values)), key=lambda i: values[i])
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and values[order[j]] == values[order[i]]:
                j += 1
            for index in order[i:j]:
                result[index] = (i + j - 1) / 2 + 1
            i = j
        return result

    rank_a, rank_b = ranks(a), ranks(b)
    mean_a, mean_b = mean(rank_a), mean(rank_b)
    centered_a = [x - mean_a for x in rank_a]
    centered_b = [x - mean_b for x in rank_b]
    denominator = math.sqrt(sum(x * x for x in centered_a) * sum(y * y for y in centered_b))
    spearman = sum(x * y for x, y in zip(centered_a, centered_b)) / denominator if denominator else None
    lengths = [len(r["codes"]) for r in records]
    common = set(occurrences[0]) & set(occurrences[1])
    summary = {"patients": len(records), "progressors": len(groups[1]), "non_progressors": len(groups[0]),
        "progressor_prevalence": len(groups[1]) / len(records), "observed_vocabulary_size": len(vocab),
        "length_median": median(lengths), "length_mean": mean(lengths), "length_max": max(lengths),
        "length_median_by_label": {str(label): median(len(r["codes"]) for r in group) for label, group in groups.items()},
        "shared_code_categories": len(common), "vocabulary_jaccard": len(common) / len(vocab),
        "positive_exclusive_categories": len(set(occurrences[1]) - set(occurrences[0])),
        "code_frequency_cosine": cosine, "code_frequency_spearman": spearman,
        "observed_vocabulary_includes_unmapped_token": "OTHER" in vocab}
    if all("n_admissions" in r for r in records):
        admissions = [r["n_admissions"] for r in records]
        summary.update(admissions_median=median(admissions), admissions_mean=mean(admissions),
                       fraction_at_least_2_admissions=mean(n >= 2 for n in admissions),
                       fraction_at_least_3_admissions=mean(n >= 3 for n in admissions))
    selected = []
    for code in ("CCS_156", "CCS_87", "CCS_215", "CCS_158"):
        p1, p0 = (prevalence[label][code] / len(groups[label]) for label in (1, 0))
        selected.append({"code": code, "progressor_patient_prevalence": p1,
                         "nonprogressor_patient_prevalence": p0,
                         "log2_prevalence_ratio": math.log2(p1 / p0) if p1 and p0 else None})
    return summary, selected


def paired_histories(records, seed=42):
    rng = random.Random(seed)
    paired = []
    for row in records:
        shuffled = list(row["codes"])
        rng.shuffle(shuffled)
        if Counter(shuffled) != Counter(row["codes"]):
            raise ValueError("shuffle changed the patient code multiset")
        paired.append({**row, "ordered": list(row["codes"]), "shuffled": shuffled})
    return paired


def neural_split_indices(records, seed=42):
    from sklearn.model_selection import train_test_split
    labels = [r["label"] for r in records]
    train, remaining = train_test_split(list(range(len(records))), test_size=.4,
                                        stratify=labels, random_state=seed)
    val, test = train_test_split(remaining, test_size=.5,
                                 stratify=[labels[i] for i in remaining], random_state=seed)
    return {"train": train, "val": val, "test": test}


def count_features(histories, vocabulary):
    """Counts of diagnosis occurrences."""
    counts = [Counter(history) for history in histories]
    return [[count[token] for token in vocabulary] for count in counts]


def kgrams(tokens, k):
    return [tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def kgram_probabilities(histories, labels, k):
    counts = defaultdict(lambda: [0, 0])
    for history, label in zip(histories, labels, strict=True):
        for gram in kgrams(history, k):
            counts[gram][int(label)] += 1
    return {gram: positive / (negative + positive) for gram, (negative, positive) in counts.items()}


def kgram_score(history, probabilities, k):
    grams = kgrams(history, k)
    return sum(probabilities.get(g, .5) for g in grams) / len(grams) if grams else .5


def baseline_runs(records, seed=42):
    """Run Appendix F k-gram and bag-of-codes baseline controls."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    def metrics(y, p, threshold):
        binary = np.asarray(p) >= threshold
        return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, binary, zero_division=0)),
                "precision": float(precision_score(y, binary, zero_division=0)),
                "recall": float(recall_score(y, binary, zero_division=0)), "threshold": threshold}

    labels = np.array([r["label"] for r in records])
    rows = []
    for split_seed in range(seed, seed + 5):
        train, test = train_test_split(np.arange(len(records)), test_size=.2,
                                       stratify=labels, random_state=split_seed)
        for k in range(1, 8):
            probs = kgram_probabilities([records[i]["codes"] for i in train], labels[train], k)
            scores = [kgram_score(records[i]["codes"], probs, k) for i in test]
            rows.append({"model": f"{k}-gram", "condition": "ordered", "split_seed": split_seed,
                         "train_rows": len(train), "test_rows": len(test), "threshold_source": "fixed",
                         **metrics(labels[test], scores, .5)})
    splits = neural_split_indices(records, seed)
    vocab = sorted({c for i in splits["train"] for c in records[i]["codes"]})
    features = {key: np.asarray(count_features([records[i]["codes"] for i in indices], vocab),
                                dtype=np.float32) for key, indices in splits.items()}
    paired = paired_histories(records, seed)
    for key, indices in splits.items():
        if not np.array_equal(features[key], count_features([paired[i]["shuffled"] for i in indices], vocab)):
            raise ValueError("count features changed under patient permutation")
    train_y = labels[splits["train"]]
    models = {
        "LogisticRegression_counts": LogisticRegression(C=1., max_iter=1000, class_weight="balanced", random_state=seed),
        "XGBoost_counts": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=.1,
                    scale_pos_weight=float((train_y == 0).sum() / (train_y == 1).sum()),
                    random_state=seed, eval_metric="logloss", verbosity=0),
    }
    for name, model in models.items():
        model.fit(features["train"], train_y)
        val_prob = model.predict_proba(features["val"])[:, 1]
        grid = np.arange(10, 90) / 100
        threshold = float(grid[np.argmax([f1_score(labels[splits["val"]], val_prob >= t, zero_division=0) for t in grid])])
        test_prob = model.predict_proba(features["test"])[:, 1]
        for condition in ("ordered", "shuffled"):
            rows.append({"model": name, "condition": condition, "split_seed": seed,
                         "train_rows": len(splits["train"]), "val_rows": len(splits["val"]),
                         "test_rows": len(splits["test"]), "threshold_source": "validation",
                         **metrics(labels[splits["test"]], test_prob, threshold)})
    return rows


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()
