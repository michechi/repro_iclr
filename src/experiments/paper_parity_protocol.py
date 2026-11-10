"""Appendix E.1-E.5 parity experiment protocol and summary aggregation."""
from __future__ import annotations

import hashlib
import json

PROTOCOL_ID = "paper_parity_v1"
MODELS = ("transformer", "lstm", "bert")
MAIN_KEYS = (22, 3, 16, 9, 23, 13)  # W,D,Q,J,X,N; zero-based letters


def gpc_label(sequence, key_ids):
    """Def. 3.7: even number of keys with even occurrence count, including zero."""
    keys = frozenset(key_ids)
    return int(sum(token in keys for token in sequence) % 2 == len(keys) % 2)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "big")


def recipe(section, model):
    if section not in {"E1", "E2", "E3", "E4", "E5"} or model not in MODELS:
        raise ValueError("unknown section or model")
    if section == "E1" and model != "transformer":
        raise ValueError("E.1 only specifies the Transformer")
    result = {
        "model": model, "loss": "two_class_cross_entropy", "head_classes": 2,
        "threshold_selection": "validation F1 on 0.10..0.89; absent in E1",
    }
    if model == "bert":
        result.update(model_name="bert-base-uncased", initialization="pretrained",
                      optimizer="AdamW", lr=2e-5, weight_decay=.01, batch_size=32,
                      max_epochs=20, stopping="validation_loss", patience=3,
                      schedule="linear_6_percent_warmup", max_length=200,
                      lora={"r": 8, "alpha": 16, "dropout": .1,
                            "targets": ["query", "key", "value"]})
    else:
        result.update(initialization="random", optimizer="Adam" if section in {"E1", "E5"} else "AdamW",
                      lr=1e-3, weight_decay=0.0 if section in {"E1", "E5"} else .01,
                      batch_size=1024 if section == "E1" else 64,
                      max_epochs=None if section == "E1" else 30,
                      steps=1000 if section == "E1" else None,
                      stopping="fixed_steps" if section == "E1" else ("validation_loss" if section == "E5" else "validation_f1"),
                      patience=None if section == "E1" else (3 if section == "E5" else 5),
                      schedule="none")
        if model == "transformer":
            result.update(hidden_size=64, heads=8, layers=2, feedforward_size=512,
                          dropout=.3, activation="gelu", pooling="cls" if section == "E1" else "masked_mean",
                          positional_embedding="learned_standard_normal", final_layer_norm=True)
        else:
            result.update(embedding_size=32, hidden_size=64, layers=1, dropout=.3, pooling="final_hidden")
    return result


def build_plan(section="all"):
    if section not in {"all", "E1", "E2", "E3", "E4", "E5"}:
        raise ValueError("section must be all or E1..E5")
    jobs, references = [], []

    def add(sec, model, ell, m, n, seed, *, regime="fixed", stage=None):
        key = f"{sec}_{model}_l{ell}_m{m}_n{n}_{regime}_seed{seed}"
        if stage:
            key += f"_stage{stage}"
        keys = list(MAIN_KEYS) if sec == "E5" else list(range(m))
        data_id = f"{sec}_l{ell}_m{m}_n{n}"
        config = recipe(sec, model)
        job = {"id": key, "section": sec, "table": {"E1": 8, "E2": 9, "E3": 10, "E4": 11, "E5": 12}[sec],
               "model": model, "seed": seed, "alphabet_size": ell, "n_keys": m,
               "sequence_length": n, "keys": keys, "regime": regime, "stage": stage,
               "data_id": data_id, "data_seed": stable_seed(data_id, seed if sec == "E1" else "shared"),
               "rows": {"train": 100000, "val": 0 if sec == "E1" else 20000,
                        "test": 10000 if sec == "E1" else 20000},
               "label_rule": "gpc_definition_3_7_even_number_of_even_key_counts",
               "config": config, "depends_on": None}
        if stage == 2:
            job["depends_on"] = f"E5_{model}_l26_m6_n10_fixed_seed{seed}_stage1"
        jobs.append(job)

    if section in {"all", "E1"}:
        for n in (10, 20):
            for regime in ("streaming", "fixed"):
                for seed in range(10):
                    add("E1", "transformer", 2, 1, n, seed, regime=regime)
    if section in {"all", "E2"}:
        for ell in (2, 4, 8, 16, 26):
            for model in MODELS:
                for seed in range(3):
                    add("E2", model, ell, max(1, ell // 2), 20, seed)
    if section in {"all", "E3", "E4"}:
        for m in ((2, 4, 6, 8, 10, 13) if section != "E4" else (13,)):
            for model in MODELS:
                for seed in range(3):
                    add("E3", model, 26, m, 20, seed)
    if section in {"all", "E4"}:
        for model in MODELS:
            for seed in range(3):
                references.append({"section": "E4", "table": 11, "n": 20, "model": model, "seed": seed,
                                   "source_job": f"E3_{model}_l26_m13_n20_fixed_seed{seed}"})
        for n in (10, 15, 30):
            for model in MODELS:
                for seed in range(3):
                    add("E4", model, 26, 13, n, seed)
    if section in {"all", "E5"}:
        for stage, n in ((1, 10), (2, 20)):
            for model in MODELS:
                for seed in range(3):
                    add("E5", model, 26, 6, n, seed, stage=stage)
    return {"protocol_id": PROTOCOL_ID, "status": PROTOCOL_ID, "section": section,
            "jobs": jobs, "reused_cells": references,
            "data_policy": "iid uniform; independently seeded splits",
            "initialization_policy": "model seed independent of data RNG; E1 streaming/fixed share initialization"}


def draw_sequences(rows, n, ell, keys, rng):
    import numpy as np
    if not 1 <= len(keys) <= ell <= 26 or n < 1 or rows < 1:
        raise ValueError("invalid sequence population")
    if len(set(keys)) != len(keys) or any(k < 0 or k >= ell for k in keys):
        raise ValueError("keys must be distinct alphabet indices")
    raw = rng.integers(0, ell, size=(rows, n), dtype=np.int64)
    labels = (np.isin(raw, keys).sum(1) % 2 == len(keys) % 2).astype(np.int64)
    return raw + 1, labels


def generate_split(job, split):
    import numpy as np
    rows = job["rows"][split]
    if not rows:
        return None
    rng = np.random.default_rng(stable_seed(job["data_seed"], split))
    return draw_sequences(rows, job["sequence_length"], job["alphabet_size"], job["keys"], rng)


def validate_parent_job(source, job, protocol_id):
    if protocol_id != PROTOCOL_ID or source["id"] != job["depends_on"]:
        raise ValueError("checkpoint is not the declared stage-one dependency")
    fields = ("model", "seed", "alphabet_size", "n_keys", "keys", "config")
    if any(source[k] != job[k] for k in fields) or source["stage"] != 1 or source["sequence_length"] != 10:
        raise ValueError("curriculum requires the matched seed/model/rule/config from stage one")
    if job["stage"] != 2 or job["sequence_length"] != 20:
        raise ValueError("destination must be the n=20 second stage")


def summarize_run_root(plan, run_root):
    from collections import defaultdict
    from pathlib import Path
    import statistics
    from src.experiments.paper_data import sha256
    groups = defaultdict(list)
    for job in plan["jobs"]:
        cell = (job["section"], job["model"], job["alphabet_size"], job["n_keys"],
                job["sequence_length"], job["regime"], job["stage"])
        path = Path(run_root) / job["id"] / "metrics.json"
        result = None
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get("protocol_id") != PROTOCOL_ID or record.get("job") != job or record.get("status") != "completed":
                raise ValueError(f"result identity or protocol mismatch: {path}")
            expected_selection = "fixed_1000_steps" if job["section"] == "E1" else job["config"]["stopping"]
            if record.get("selected_by") != expected_selection:
                raise ValueError(f"checkpoint selection mismatch: {path}")
            for name, field in (("manifest.json", "manifest_sha256"), ("selected.pt", "checkpoint_sha256"),
                                ("test_predictions.npz", "predictions_sha256")):
                artifact = path.with_name(name)
                if not artifact.is_file() or sha256(artifact) != record.get(field):
                    raise ValueError(f"artifact SHA256 mismatch: {artifact}")
            manifest = json.loads(path.with_name("manifest.json").read_text())
            if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("job") != job:
                raise ValueError(f"manifest identity mismatch: {path}")
            source_hashes = manifest.get("source_sha256")
            if not isinstance(source_hashes, dict) or not source_hashes:
                raise ValueError(f"source-code hashes absent: {path}")
            if len({seed for seed, _ in groups[cell]} | {job["seed"]}) != len(groups[cell]) + 1:
                raise ValueError(f"duplicate seed in plan cell: {path}")
            for _, prev in groups[cell]:
                if prev is not None and prev["source_sha256"] != source_hashes:
                    raise ValueError(f"source-code versions differ between seeds: {path}")
            value = record.get("test", {}).get("auc")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"invalid test AUC: {path}")
            result = {"seed": job["seed"], "auc": value, "source": str(path),
                      "source_sha256": source_hashes,
                      "verified_artifacts": {field: record[field] for field in
                          ("manifest_sha256", "checkpoint_sha256", "predictions_sha256")}}
        groups[cell].append((job["seed"], result))
    rows = []
    for cell, items in groups.items():
        observed = [record for _, record in items if record is not None]
        values = [record["auc"] for record in observed]
        rows.append({"section": cell[0], "model": cell[1], "ell": cell[2], "m": cell[3],
                     "n": cell[4], "regime": cell[5], "stage": cell[6],
                     "required_seeds": [seed for seed, _ in items],
                     "missing_seeds": [seed for seed, record in items if record is None],
                     "n_observed": len(values), "complete": len(values) == len(items),
                     "auc_mean": statistics.mean(values) if values else None,
                     "auc_sample_sd": statistics.stdev(values) if len(values) > 1 else None,
                     "success_rate_auc_gt_0_9": sum(v > .9 for v in values) / len(values) if values else None,
                     "per_seed": observed})
    if plan["reused_cells"]:
        for model in MODELS:
            source = next(row for row in rows if row["section"] == "E3" and row["model"] == model
                          and row["ell"] == 26 and row["m"] == 13 and row["n"] == 20)
            rows.append({**source, "section": "E4", "reused_from_section": "E3",
                         "source_job_ids": [r["source_job"] for r in plan["reused_cells"] if r["model"] == model]})
    return {"protocol_id": PROTOCOL_ID, "cells": rows, "reused_cells": plan["reused_cells"]}
