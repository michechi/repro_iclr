"""Experiment routing and pretrained experiment configurations."""
from __future__ import annotations

from pathlib import Path
from .paper_protocol import CONFIG, SEEDS, TRAIN_SIZES, model_spec

RECIPES = {
    "bert_lora": ("bert", "lora"), "bert_full": ("bert", "full"),
    "roberta_lora": ("roberta", "lora"), "roberta_full": ("roberta", "full"),
    "llama1b_lora": ("llama1b", "lora"), "llama1b_full": ("llama1b", "full"),
    "llama8b_lora": ("llama8b", "lora"), "llama8b_full": ("llama8b", "full"),
    "qwen4b_lora": ("qwen4b", "lora"), "qwen14b_lora": ("qwen14b", "lora"),
    "qwen27b_qlora": ("qwen27b", "qlora"), "llama70b_qlora": ("llama70b", "qlora"),
}
# Appendix G attention fits (distinct from Table 1 / main-result recipes).
ATTENTION_RECIPES = {
    "bert_full": ("bert", "full"),
    "roberta_base_full": ("roberta_base", "full"),
    "roberta_base_lora": ("roberta_base", "lora"),
    "llama1b_full": ("llama1b", "full"),
    "llama8b_lora": ("llama8b", "lora"),
    "qwen14b_qlora": ("qwen14b", "qlora"),
    "qwen27b_qlora": ("qwen27b", "qlora"),
}
# Table 4 / Section 6: 70B and Qwen3-4B are not in the KIP printed cells.
KIP_RECIPES = [key for key in RECIPES if key not in {"qwen4b_lora", "llama70b_qlora"}]
TABLE_ROWS = {
    "ocdet": ["bert_lora", "roberta_lora", "llama1b_lora", "llama1b_full",
              "llama8b_lora", "qwen4b_lora", "qwen27b_qlora"],
    "ocnoise": ["bert_lora", "roberta_lora", "llama1b_lora", "llama1b_full",
                "llama8b_lora", "llama8b_full", "qwen4b_lora", "qwen14b_lora", "qwen27b_qlora"],
    "kip_m4": KIP_RECIPES, "kip_m6": KIP_RECIPES, "test_just_pair": list(RECIPES),
}

# Published paper instances used only when regenerating the reported datasets.
# Loaders never read these: they always take the rule from paper_dataset_manifest.json.
PAPER_INSTANCES = {
    "ocdet": {"task": "oc", "key_letters": "WDQJXU", "lag": 7, "noise": 0.0, "n": 20, "m": 6},
    "ocnoise": {"task": "oc", "key_letters": "WDQJXU", "lag": 7, "noise": 0.3, "n": 20, "m": 6},
    "kip_m4": {"task": "kip", "key_letters": "COTA", "noise": 0.0, "n": 20, "m": 4},
    "kip_m6": {"task": "kip", "key_letters": "GHLYIZ", "noise": 0.0, "n": 20, "m": 6},
    "test_just_pair": {"task": "gpc", "key_letters": "WDQJXN", "noise": 0.0, "n": 20, "m": 6},
}
DATA_DIRS = {tag: f"data/paper/{tag}" for tag in PAPER_INSTANCES}
ROUTES = {
    "pretrained": {"paper": "4.2-4.3, B.1-B.2, Tables 1/3/4/5", "entrypoint": "scripts/run_paper_pretrained.py", "training": True},
    "kip-sweep": {"paper": "4.3, Table 4", "entrypoint": "scripts/run_kip_data_efficiency.py", "training": True},
    "baselines": {"paper": "B.1-B.2, Table 2, D.1-D.3", "entrypoint": "scripts/run_paper_baselines.py", "training": True},
    "core": {"paper": "4.3, B, D.1-D.4", "entrypoint": "scripts/paper_core.py", "training": False},
    "data": {"paper": "3, A, 4.1, D.1", "entrypoint": "scripts/run_paper_data.py", "training": False},
    "heldout-rule": {"paper": "D.4", "entrypoint": "scripts/run_paper_heldout_rule.py", "training": True},
    "parity": {"paper": "E.1-E.5, Tables 8-12", "entrypoint": "scripts/run_paper_parity.py", "training": True},
    "gpc-helpers": {"paper": "5, Appendix E opening helper ladder", "entrypoint": "scripts/run_paper_gpc_helpers.py", "training": True},
    "clinical": {"paper": "4.4, F, Tables 13-15", "entrypoint": "scripts/run_paper_clinical.py", "training": True},
}


def data_generation_command(tag, data_root="data/paper", seed=None, *, use_paper_keys=True):
    """Build a ``run_paper_data.py`` command for one synthetic instance."""
    if tag not in PAPER_INSTANCES:
        raise ValueError(f"unknown paper instance {tag!r}; choose {tuple(PAPER_INSTANCES)}")
    instance = PAPER_INSTANCES[tag]
    cmd = ["python", "-B", "scripts/run_paper_data.py",
           "--task", instance["task"], "--tag", tag,
           "--keys", str(instance["m"]), "--noise", str(instance["noise"]),
           "--length", str(instance["n"]),
           "--seed", str(SEEDS[0] if seed is None else seed),
           "--output", str(Path(data_root) / tag)]
    if instance["task"] == "oc":
        cmd.extend(["--lag", str(instance["lag"])])
    if use_paper_keys:
        cmd.extend(["--key-letters", instance["key_letters"]])
    return cmd + ["--execute"]


def pretrained_plan(root, *, scope="all-core", sizes=TRAIN_SIZES, seeds=SEEDS, snapshots=None,
                    data_root="data/paper", use_paper_keys=True):
    """Build the Section 4.3 pretrained experiment schedule."""
    if scope not in {"all-core", "table-cells"}:
        raise ValueError("scope must be all-core or table-cells")
    if not sizes or any(size not in TRAIN_SIZES for size in sizes):
        raise ValueError("use the paper training sizes")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    root, snapshots = Path(root), snapshots or {}
    jobs = []
    for tag, rows in TABLE_ROWS.items():
        for recipe in (RECIPES if scope == "all-core" else rows):
            model, tuning = RECIPES[recipe]
            spec = model_spec(model, tuning)
            for seed in seeds:
                for size in sizes:
                    identity = f"{tag}/{recipe}/n{size}/seed{seed}"
                    snapshot = snapshots.get(model)
                    cmd = ["python", "-B", "scripts/run_paper_pretrained.py", "--model", model,
                           "--tuning", tuning, "--model-path", str(snapshot or f"<LOCAL_{model.upper()}_SNAPSHOT>"),
                           "--tag", tag, "--data-dir", str(Path(data_root) / tag), "--train-size", str(size),
                           "--seed", str(seed), "--output-dir", str(root / identity)]
                    jobs.append({"id": identity, "tag": tag, "recipe": recipe, "model": spec,
                                 "seed": seed, "train_rows": size, "val_rows": 50000, "test_rows": 50000,
                                 "printed_table_cell": recipe in rows,
                                 "status": tag, "snapshot_configured": snapshot is not None,
                                 "inspect_command": cmd + ["--dry-run"],
                                 "execute_command": cmd + ["--execute"]})
    return {"schema_version": 1, "scope": scope,
            "protocol": CONFIG.to_dict(), "job_count": len(jobs),
            "seeds": list(seeds),
            "paper_instances": PAPER_INSTANCES,
            "data_generation_commands": [
                data_generation_command(tag, data_root, seeds[0], use_paper_keys=use_paper_keys)
                for tag in PAPER_INSTANCES
            ],
            "jobs": jobs}
