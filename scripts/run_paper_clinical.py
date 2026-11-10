#!/usr/bin/env python3
"""Appendix F cohort, summaries, and shortcut controls."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mimic import paper_clinical as protocol


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("no cohort/result rows to write")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["protocol", "cohort", "prepare", "summarize", "baselines"], default="protocol")
    parser.add_argument("--diagnoses", type=Path)
    parser.add_argument("--admissions", type=Path)
    parser.add_argument("--mapping", type=Path, default=ROOT / "codes/CCS_DX_mapping.csv")
    parser.add_argument("--unmapped-policy", choices=["error", "other", "drop"])
    parser.add_argument("--mapping-conflicts", choices=["error", "first", "last"], default="error")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if args.mode != "protocol" and not args.output_dir:
        parser.error("execution requires --output-dir")
    if args.mode == "cohort" and not (args.diagnoses and args.admissions and args.unmapped_policy):
        parser.error("cohort requires --diagnoses, --admissions and --unmapped-policy")
    if args.mode in {"prepare", "summarize", "baselines"} and not args.cohort:
        parser.error("prepare/summarize/baselines require --cohort")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = {"protocol": protocol.PROTOCOL, "mode": args.mode, "seed": args.seed,
            "private_data_required": args.mode != "protocol"}
    if args.mode == "protocol":
        print(json.dumps(plan, indent=2))
        return plan
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    sources = {"entrypoint": protocol.sha256(__file__), "protocol": protocol.sha256(protocol.__file__)}
    inputs = ({"diagnoses": args.diagnoses, "admissions": args.admissions, "mapping": args.mapping}
              if args.mode == "cohort" else {"cohort": args.cohort})
    input_hashes = {key: protocol.sha256(path) for key, path in inputs.items()}
    manifest = {**plan, "status": "running", "source_sha256": sources, "input_sha256": input_hashes,
                "local_patient_level_output": args.mode in {"cohort", "prepare"},
                "mapping_conflict_policy": args.mapping_conflicts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.mode == "cohort":
        records, summary = protocol.build_cohort(lambda: protocol.read_csv(args.diagnoses),
            protocol.read_csv(args.admissions), protocol.ccs_mapping(args.mapping, conflict_policy=args.mapping_conflicts),
            unmapped=args.unmapped_policy)
        write_csv(output / "ckd_cohort_ccs.csv", records)
        manifest["cohort_summary"] = summary
    else:
        records = protocol.read_cohort(args.cohort)
        if args.mode == "prepare":
            paired = protocol.paired_histories(records, args.seed)
            splits = protocol.neural_split_indices(records, args.seed)
            for split, indices in splits.items():
                for condition in ("ordered", "shuffled"):
                    write_csv(output / f"X_{split}_{condition}.csv",
                              [{"Sequences": "\x1f".join(paired[i][condition])} for i in indices])
                    write_csv(output / f"y_{split}_{condition}.csv",
                              [{"Outcome": paired[i]["label"]} for i in indices])
                write_csv(output / f"patients_{split}.csv", [{"subject_id": paired[i]["subject_id"]} for i in indices])
            vocab = sorted({c for row in records for c in row["codes"]})
            (output / "vocab.json").write_text(json.dumps({"code_to_idx": {c: i + 1 for i, c in enumerate(vocab)},
                    "vocab_size": len(vocab) + 1}, indent=2))
            manifest["split_rows"] = {key: len(value) for key, value in splits.items()}
        elif args.mode == "summarize":
            summary, selected = protocol.cohort_statistics(records)
            (output / "cohort_statistics.json").write_text(json.dumps(summary, indent=2) + "\n")
            write_csv(output / "selected_ccs_prevalence.csv", selected)
        else:
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
            rows = protocol.baseline_runs(records, args.seed)
            write_csv(output / "baseline_results.csv", rows)
            manifest["result_rows"] = len(rows)
    if input_hashes != {key: protocol.sha256(path) for key, path in inputs.items()}:
        raise RuntimeError("clinical inputs changed during execution")
    manifest["output_sha256"] = {p.name: protocol.sha256(p) for p in output.iterdir()
                                  if p.is_file() and p.name != "manifest.json"}
    manifest["status"] = "completed"
    manifest["environment"] = {name: importlib.metadata.version(name) for name in
                                (["numpy", "scikit-learn", "xgboost"] if args.mode == "baselines" else
                                 ["scikit-learn"] if args.mode == "prepare" else [])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "completed", "mode": args.mode, "output_dir": str(output)}, indent=2))
    return manifest


if __name__ == "__main__":
    main()
