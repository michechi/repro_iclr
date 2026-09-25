#!/usr/bin/env python3
"""Check every canonical KIP row and corrected OC clean label (standard library)."""
import ast
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def values(path):
    with path.open(newline="") as handle:
        rows = csv.reader(handle)
        next(rows)
        for row in rows:
            yield row[0]


def main():
    report = {}
    directory = ROOT / "data/simulation/kip_ordered_recovered"
    recorded = json.loads((directory / "recovery_manifest.json").read_text())["output_files_sha256"]
    for tag, keys in [("kip_m4", "COTA"), ("kip_m6", "GHLYIZ")]:
        for split, size in [("train", 400000), ("val", 50000), ("test", 50000)]:
            xp, yp = [directory / f"{prefix}_{split}_{tag}.csv" for prefix in ("X", "y")]
            for path in (xp, yp):
                assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded[path.name], path.name
            n = positives = 0
            for sequence, label in zip(values(xp), values(yp), strict=True):
                letters = sequence.split("\x1f")
                assert len(letters) == 20 and all(len(c) == 1 and "A" <= c <= "Z" for c in letters)
                assert all(letters.count(c) == 1 for c in keys)
                ranks = [keys.index(c) for c in letters if c in keys]
                inversions = sum(a > b for i, a in enumerate(ranks) for b in ranks[i + 1:])
                outcome = float(label)
                assert outcome in (0., 1.) and outcome == int(inversions % 2 == 0)
                n += 1
                positives += int(outcome)
            assert n == size and 0 < positives < n
            report[f"{tag}/{split}"] = {"rows": n, "positives": positives, "oracle": "passed"}
    # Load the canonical pure rule without importing optional training dependencies.
    # The exact module/function is shared with the existing attention oracle.
    oracle_module = ROOT / "src/analysis/oc_attention_core.py"
    oracle_text = oracle_module.read_text()
    module = ast.parse(oracle_text)
    imports = [node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
               and not (isinstance(node, ast.Import) and any(x.name in ("numpy", "torch") for x in node.names))]
    selected = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.Assign))
                and (getattr(node, "name", "") in ("canonical_rule", "label")
                     or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("ROOT", "ORACLE_SOURCE", "KEYS", "LAG") for t in node.targets))]
    namespace = {"__file__": str(oracle_module)}
    exec(compile(ast.Module(body=imports + selected, type_ignores=[]), str(oracle_module), "exec"), namespace)
    oracle = namespace["label"]
    directory = ROOT / "results/qwen25_14b_fewshot_20260918/data/oc_corrected"
    for split, size in [("train", 400000), ("val", 50000), ("test", 50000)]:
        paths = [directory / f"{prefix}_{split}_{tag}.csv"
                 for prefix, tag in [("X", "ocdet"), ("y", "ocdet"), ("X", "ocnoise"), ("y", "ocnoise")]]
        n = flips = 0
        for sequence, label, noisy_sequence, noisy_label in zip(*(values(path) for path in paths), strict=True):
            assert sequence == noisy_sequence and int(float(label)) == oracle(sequence)
            assert float(noisy_label) in (0., 1.)
            flips += float(label) != float(noisy_label)
            n += 1
        assert n == size
        report[f"corrected_oc/{split}"] = {"rows": n, "clean_oracle": "passed", "observed_noise_rate": flips / n}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
