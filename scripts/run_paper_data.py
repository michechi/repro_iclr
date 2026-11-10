#!/usr/bin/env python3
"""Write the paper synthetic splits. Generation requires --execute."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_data import sample_rule, generate_rows, sha256, oc_prevalence, oracle_metrics


def plan(args):
    configurations = [(m, lag) for m in (3, 6, 10) for lag in (1, 3, 5, 7, 9, 10)] if args.sensitivity else [(args.keys, args.lag)]
    jobs = []
    for i, (m, lag) in enumerate(configurations):
        task = "oc" if args.sensitivity else args.task
        rule = sample_rule(task, args.seed + i, n=args.length, m=m, lag=lag, noise=.3 if args.sensitivity else args.noise)
        tag = f"oc_m{m}_lag{lag}" if args.sensitivity else args.tag
        if not tag:
            raise ValueError("--tag is required outside --sensitivity")
        job = {"protocol": "paper_data_v1", "tag": tag, "rule": rule,
               "counts": {"train": 400000, "val": 50000, "test": 50000},
               "output": str(args.output / tag if args.sensitivity else args.output),
               "sequence_seed": args.seed + 1000 + i,
               "noise_seed": args.seed + 100000 + i,
               "input_distribution": "iid_uniform" if task != "kip" else "uniform_exactly_once_keys",
               "prevalence_resampling": False}
        if task == "oc":
            rho = oc_prevalence(rule["n"], m, lag, len(rule["alphabet"]))
            job["theoretical_latent_prevalence"] = rho
            job["noise_oracle"] = oracle_metrics(rho, rule["noise_probability"])
        jobs.append(job)
    return jobs


def execute(job):
    output = Path(job["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    stream = generate_rows(job["rule"], sum(job["counts"].values()), job["sequence_seed"], job["noise_seed"])
    hashes = {}
    summary = {}
    for split, count in job["counts"].items():
        paths = [output / f"{kind}_{split}_{job['tag']}.csv" for kind in ("X", "y", "y_latent")]
        with paths[0].open("w", newline="") as xf, paths[1].open("w", newline="") as yf, paths[2].open("w", newline="") as lf:
            writers = [csv.writer(f) for f in (xf, yf, lf)]
            for w, h in zip(writers, ["Sequences", "Outcome", "Outcome"]):
                w.writerow([h])
            latent_sum = observed_sum = flips = 0
            for _ in range(count):
                sequence, latent, observed = next(stream)
                for w, v in zip(writers, [sequence, observed, latent]):
                    w.writerow([v])
                latent_sum += latent
                observed_sum += observed
                flips += latent != observed
        hashes.update({p.name: sha256(p) for p in paths})
        summary[split] = {"rows": count, "latent_prevalence": latent_sum / count,
                          "observed_prevalence": observed_sum / count, "realized_flip_rate": flips / count}
    manifest = {**job, "files_sha256": hashes, "split_summary": summary,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), ROOT / "src/experiments/paper_data.py"]}}
    (output / "paper_dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["oc", "gpc", "kip"], default="oc")
    p.add_argument("--tag")
    p.add_argument("--keys", type=int, default=6)
    p.add_argument("--length", type=int, default=20)
    p.add_argument("--lag", type=int, default=7)
    p.add_argument("--noise", type=float, default=0.)
    p.add_argument("--seed", type=int, default=9550)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--execute", action="store_true", help="write the 500K-row datasets; the default prints the configuration")
    args = p.parse_args(argv)
    try:
        jobs = plan(args)
    except ValueError as e:
        p.error(str(e))
    if args.execute:
        for job in jobs:
            execute(job)
    else:
        print(json.dumps({"jobs": jobs}, indent=2))


if __name__ == "__main__":
    main()
