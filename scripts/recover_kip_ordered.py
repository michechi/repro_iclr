#!/usr/bin/env python3
"""Recover original KIP rows by exactly inverting their recorded shuffle.

The historical source is absent, but the existing Python 3.12 bytecode at
repro/src/data/__pycache__/kip_shuffles.cpython-312.pyc records the algorithm:
random.Random(f"{seed}:{tag}:{split}:{row}").shuffle(tokens).
This script does not execute the bytecode. Each recovered row is checked using
the rule sidecar, and its forward shuffle is checked against the source row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


SEP = "\x1f"
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parity(tokens: list[str], kappa: dict[str, int]) -> int:
    ranks = [kappa[token] for token in tokens if token in kappa]
    inversions = sum(a > b for i, a in enumerate(ranks) for b in ranks[i + 1 :])
    return int(inversions % 2 == 0)


def recover_sequence(raw: str, *, seed: int, tag: str, split: str, row: int) -> list[str]:
    shuffled_tokens = raw.split(SEP)
    permutation = list(range(len(shuffled_tokens)))
    random.Random(f"{seed}:{tag}:{split}:{row}").shuffle(permutation)
    original_tokens = [""] * len(shuffled_tokens)
    for shuffled_position, original_position in enumerate(permutation):
        original_tokens[original_position] = shuffled_tokens[shuffled_position]
    # Independent forward application of the historical operation.
    roundtrip_tokens = original_tokens.copy()
    random.Random(f"{seed}:{tag}:{split}:{row}").shuffle(roundtrip_tokens)
    if roundtrip_tokens != shuffled_tokens:
        raise ValueError(f"Shuffle roundtrip failed for {tag}/{split}/{row}")
    return original_tokens


def recover(source: Path, output: Path) -> dict:
    shuffle_manifest_path = source / "shuffle_manifest.json"
    shuffle_manifest = json.loads(shuffle_manifest_path.read_text())
    seed = int(shuffle_manifest["shuffle_seed"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Recovery output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_bytecode = REPO_ROOT / "repro/src/data/__pycache__/kip_shuffles.cpython-312.pyc"
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Exact inverse of historical deterministic per-row shuffle",
        "shuffle_algorithm": 'random.Random(f"{seed}:{tag}:{split}:{row}").shuffle(tokens)',
        "shuffle_seed": seed,
        "source_dir": str(source.resolve()),
        "historical_ordered_dir": shuffle_manifest["source"],
        "output_dir": str(output.resolve()),
        "python_version": sys.version,
        "algorithm_evidence": {
            "path": str(source_bytecode),
            "sha256": sha256(source_bytecode) if source_bytecode.exists() else None,
        },
        "recovery_script_sha256": sha256(Path(__file__)),
        "source_files_sha256": {"shuffle_manifest.json": sha256(shuffle_manifest_path)},
        "output_files_sha256": {},
        "splits": {},
    }
    for tag in ("kip_m4", "kip_m6"):
        rule_path = source / f"{tag}_rule.json"
        rule = json.loads(rule_path.read_text())
        kappa = rule["kappa"]
        for split in ("train", "val", "test"):
            x_name, y_name = f"X_{split}_{tag}.csv", f"y_{split}_{tag}.csv"
            rows = positives = label_preserved = 0
            forward_hasher = hashlib.sha256(b"Sequences\n")
            with (
                (source / x_name).open(newline="") as x_handle,
                (source / y_name).open(newline="") as y_handle,
                (output / x_name).open("x", newline="") as out_handle,
            ):
                x_reader, y_reader = csv.DictReader(x_handle), csv.DictReader(y_handle)
                if x_reader.fieldnames != ["Sequences"] or y_reader.fieldnames != ["Outcome"]:
                    raise ValueError(f"Unexpected columns in {tag}/{split}")
                writer = csv.writer(out_handle, lineterminator="\n")
                writer.writerow(["Sequences"])
                for row, (x_record, y_record) in enumerate(zip(x_reader, y_reader, strict=True)):
                    raw = x_record["Sequences"]
                    tokens = recover_sequence(raw, seed=seed, tag=tag, split=split, row=row)
                    label_float = float(y_record["Outcome"])
                    if label_float not in (0.0, 1.0):
                        raise ValueError(f"Invalid label in {tag}/{split}/{row}")
                    label = int(label_float)
                    if len(tokens) != rule["n"] or any(token not in rule["alphabet"] for token in tokens):
                        raise ValueError(f"Invalid tokens in {tag}/{split}/{row}")
                    if any(tokens.count(key) != 1 for key in kappa):
                        raise ValueError(f"Key multiplicity failure in {tag}/{split}/{row}")
                    if parity(tokens, kappa) != label:
                        raise ValueError(f"Recovered label mismatch in {tag}/{split}/{row}")
                    label_preserved += parity(raw.split(SEP), kappa) == label
                    writer.writerow([SEP.join(tokens)])
                    # Together with the row-level forward check, this verifies
                    # the forward-recovered CSV equals the input byte for byte.
                    forward_hasher.update((raw + "\n").encode())
                    positives += label
                    rows += 1
            expected = shuffle_manifest["splits"][f"{tag}/{split}"]
            if rows != expected["rows"]:
                raise ValueError(f"Unexpected row count for {tag}/{split}: {rows}")
            if round(label_preserved / rows, 5) != expected["label_preserved_rate"]:
                raise ValueError(f"Shuffle manifest parity mismatch for {tag}/{split}")
            if forward_hasher.hexdigest() != sha256(source / x_name):
                raise ValueError(f"Forward CSV byte check failed for {tag}/{split}")
            shutil.copyfile(source / y_name, output / y_name)
            for name in (x_name, y_name):
                manifest["source_files_sha256"][name] = sha256(source / name)
                manifest["output_files_sha256"][name] = sha256(output / name)
            manifest["splits"][f"{tag}/{split}"] = {
                "rows": rows,
                "positive_labels": positives,
                "positive_rate": positives / rows,
                "all_recovered_labels_match_rule": True,
                "all_rows_forward_shuffle_match_source": True,
                "forward_csv_sha256_matches_source": True,
                "labels_byte_identical": sha256(source / y_name) == sha256(output / y_name),
                "shuffle_label_preservation_rate": label_preserved / rows,
            }
            print(f"Recovered {tag}/{split}: {rows:,} rows, every label and shuffle verified", flush=True)
        shutil.copyfile(rule_path, output / rule_path.name)
        manifest["source_files_sha256"][rule_path.name] = sha256(rule_path)
        manifest["output_files_sha256"][rule_path.name] = sha256(output / rule_path.name)
    (output / "recovery_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data/simulation/kip_shuffled")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/simulation/kip_ordered_recovered")
    args = parser.parse_args()
    recover(args.source, args.output)


if __name__ == "__main__":
    main()
