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
KIP_RECIPES = [key for key in RECIPES if key not in {"qwen4b_lora", "llama70b_qlora"}]
TABLE_ROWS = {
    "ocdet": ["bert_lora", "roberta_lora", "llama1b_lora", "llama1b_full",
              "llama8b_lora", "qwen4b_lora", "qwen27b_qlora"],
    "ocnoise": ["bert_lora", "roberta_lora", "llama1b_lora", "llama1b_full",
                "llama8b_lora", "llama8b_full", "qwen4b_lora", "qwen14b_lora", "qwen27b_qlora"],
    "kip_m4": KIP_RECIPES, "kip_m6": KIP_RECIPES, "test_just_pair": list(RECIPES),
}
DATA_RECIPES = {
    "ocdet": ("oc", 6, 0.0), "ocnoise": ("oc", 6, 0.3),
    "kip_m4": ("kip", 4, 0.0), "kip_m6": ("kip", 6, 0.0),
    "test_just_pair": ("gpc", 6, 0.0),
}
DATA_DIRS = {tag: f"data/paper/{tag}" for tag in DATA_RECIPES}
ROUTES = {
    "pretrained": {"paper": "4.2-4.3, B.1-B.2, Tables 1/3/4/5", "entrypoint": "scripts/run_paper_pretrained.py", "training": True},
    "kip-sweep": {"paper": "4.3, Table 4", "entrypoint": "scripts/run_kip_data_efficiency.py", "training": True},
    "baselines": {"paper": "B.1-B.2, Table 2, D.1-D.3", "entrypoint": "scripts/run_paper_baselines.py", "training": True},
    "core": {"paper": "4.3, B, D.1-D.4", "entrypoint": "scripts/paper_core.py", "training": False},
    "data": {"paper": "3, A, 4.1, D.1", "entrypoint": "scripts/run_paper_data.py", "training": False},
    "heldout-rule": {"paper": "D.4", "entrypoint": "scripts/run_paper_heldout_rule.py", "training": True},
    "parity": {"paper": "E.1-E.5, Tables 8-12", "entrypoint": "scripts/run_paper_parity.py", "training": True},
    "gpc-helpers": {"paper": "5, Appendix E opening helper ladder", "entrypoint": "scripts/run_paper_gpc_helpers.py", "training": True},
    "run-summary": {"paper": "Tables 3-5; Figures 1-2 data", "entrypoint": "scripts/summarize_paper_runs.py", "training": False},
    "clinical": {"paper": "4.4, F, Tables 13-15", "entrypoint": "scripts/run_paper_clinical.py", "training": True},
    "attention": {"paper": "G, Figures 3-5", "entrypoint": "scripts/reproduce_figures.py", "training": False},
}


def pretrained_plan(root, *, scope="all-core", sizes=TRAIN_SIZES, seeds=SEEDS, snapshots=None,
                    data_root="data/paper"):
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
            "data_generation_commands": [
                ["python", "-B", "scripts/run_paper_data.py", "--task", task,
                 "--tag", tag, "--keys", str(keys), "--noise", str(noise),
                 "--seed", str(SEEDS[0]), "--output", str(Path(data_root) / tag), "--execute"]
                for tag, (task, keys, noise) in DATA_RECIPES.items()
            ],
            "jobs": jobs}
