#!/usr/bin/env python3
"""Entrypoint catalog and schedule generator for the paper experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_catalog import ROUTES, pretrained_plan
from src.experiments.paper_protocol import CONFIG, MODELS, SEEDS, TRAIN_SIZES


def check_entrypoints():
    sources = {}
    for name, route in ROUTES.items():
        path = ROOT / route["entrypoint"]
        sources[name] = {**route, "exists": path.is_file(),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}
    return {"sources_present": all(row["exists"] for row in sources.values()),
            "protocol": CONFIG.to_dict(),
            "routes": sources}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["catalog", "pretrained", "check"], nargs="?", default="catalog")
    parser.add_argument("--scope", choices=["all-core", "table-cells"], default="all-core")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(TRAIN_SIZES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--run-root", type=Path, default=Path("runs/paper_pretrained"))
    parser.add_argument("--data-root", type=Path, default=Path("data/paper"))
    parser.add_argument("--model-path", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, help="write the JSON to this path")
    args = parser.parse_args(argv)
    snapshots = {}
    for entry in args.model_path:
        model, sep, path = entry.partition("=")
        if not sep or model not in MODELS or not path:
            parser.error("--model-path requires a known MODEL=PATH")
        snapshots[model] = str(Path(path).expanduser().resolve())
    if args.action == "catalog":
        result = {"routes": ROUTES}
    elif args.action == "pretrained":
        try:
            result = pretrained_plan(args.run_root, scope=args.scope, sizes=args.sizes,
                                     seeds=args.seeds, snapshots=snapshots, data_root=args.data_root)
        except ValueError as error:
            parser.error(str(error))
    else:
        result = check_entrypoints()
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
