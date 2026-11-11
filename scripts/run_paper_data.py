#!/usr/bin/env python3
"""Write the paper synthetic splits. Generation requires --execute."""
from __future__ import annotations
import argparse, csv, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.paper_data import ALPHABET, sample_rule, generate_rows, sha256, oc_prevalence, oracle_metrics


def plan(args):
    configurations = [(m, lag) for m in args.sensitivity_m for lag in args.sensitivity_lags] if args.sensitivity else [(args.keys, args.lag)]
    counts = {"train": args.train_rows, "val": args.val_rows, "test": args.test_rows}
    if any(v < 1 for v in counts.values()):
        raise ValueError("split row counts must be positive integers")
    jobs = []
    for i, (m, lag) in enumerate(configurations):
        task = "oc" if args.sensitivity else args.task
        noise = args.sensitivity_noise if args.sensitivity else args.noise
        explicit_keys = None if args.sensitivity else args.key_letters
        rule = sample_rule(task, args.seed + i, n=args.length, m=m, alphabet=args.alphabet,
                           lag=lag, noise=noise, keys=explicit_keys)
        tag = f"oc_m{m}_lag{lag}" if args.sensitivity else args.tag
        if not tag:
            raise ValueError("--tag is required outside --sensitivity")
        job = {"protocol": "paper_data_v1", "tag": tag, "rule": rule,
               "counts": dict(counts),
               "output": str(args.output / tag if args.sensitivity else args.output),
               "sequence_seed": args.seed + 1000 + i,
               "noise_seed": args.seed + 100000 + i,
               "input_distribution": "iid_uniform" if task != "kip" else "uniform_exactly_once_keys",
               # Remark A.4: splits are stratified on latent Y* before noise is applied.
               "prevalence_resampling": False,
               "split_stratification": "latent_before_noise"}
        if task == "oc":
            rho = oc_prevalence(rule["n"], len(rule["keys"]), lag, len(rule["alphabet"]))
            job["theoretical_latent_prevalence"] = rho
            job["noise_oracle"] = oracle_metrics(rho, rule["noise_probability"])
        jobs.append(job)
    return jobs


def _stratified_split_indices(latents, counts, seed):
    """Assign generated rows to splits by latent label (Remark A.4).

    Quotas are a 2×|splits| contingency table with fixed class and split margins,
    so independent per-class rounding cannot overfill one split.
    """
    by_label = {0: [], 1: []}
    for index, latent in enumerate(latents):
        by_label[int(latent)].append(index)
    rng = random.Random(seed)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    split_names = list(counts)
    total = sum(counts.values())
    if total != len(latents):
        raise ValueError("latent pool size must match requested split counts")
    class_sizes = {label: len(bucket) for label, bucket in by_label.items()}
    if sum(class_sizes.values()) != total or any(size < 0 for size in class_sizes.values()):
        raise ValueError("invalid latent class counts")
    # Floor of the independence table, then allocate remainders jointly.
    quotas = {label: {} for label in by_label}
    remainders = []
    for label, size in class_sizes.items():
        for name, count in counts.items():
            exact = size * count / total
            floor = int(exact)
            quotas[label][name] = floor
            remainders.append((exact - floor, label, name))
    class_left = {label: class_sizes[label] - sum(quotas[label].values()) for label in by_label}
    split_left = {name: counts[name] - sum(quotas[label][name] for label in by_label)
                  for name in split_names}
    if any(value < 0 for value in list(class_left.values()) + list(split_left.values())):
        raise ValueError("floor quotas already exceed class or split margins")
    for _, label, name in sorted(remainders, key=lambda item: (-item[0], item[1], item[2])):
        if class_left[label] > 0 and split_left[name] > 0:
            quotas[label][name] += 1
            class_left[label] -= 1
            split_left[name] -= 1
    # Any residual cells (ties / zero remainders) fill by stable (label, split) order.
    while sum(class_left.values()) > 0:
        progress = False
        for label in sorted(by_label):
            for name in split_names:
                if class_left[label] > 0 and split_left[name] > 0:
                    quotas[label][name] += 1
                    class_left[label] -= 1
                    split_left[name] -= 1
                    progress = True
        if not progress:
            raise ValueError("cannot complete joint stratified assignment")
    if any(class_left.values()) or any(split_left.values()):
        raise ValueError("joint stratified assignment left residual margin")
    assigned = {name: [] for name in split_names}
    for label, bucket in by_label.items():
        offset = 0
        for name in split_names:
            take = quotas[label][name]
            assigned[name].extend(bucket[offset:offset + take])
            offset += take
        if offset != len(bucket):
            raise ValueError("class quota does not exhaust its bucket")
    for name, count in counts.items():
        if len(assigned[name]) != count:
            raise ValueError(f"stratified assignment failed for {name}")
        assigned[name].sort()
    return assigned


def execute(job):
    output = Path(job["output"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    # Generate clean (latent) rows first; apply noise only after split assignment.
    clean_rule = {**job["rule"], "noise_probability": 0.}
    total = sum(job["counts"].values())
    clean_rows = list(generate_rows(clean_rule, total, job["sequence_seed"], job["noise_seed"]))
    latents = [latent for _, latent, _ in clean_rows]
    assignment = _stratified_split_indices(latents, job["counts"], job["sequence_seed"])
    noise_rng = random.Random(job["noise_seed"])
    noise = job["rule"]["noise_probability"]
    hashes = {}
    summary = {}
    for split, indices in assignment.items():
        paths = [output / f"{kind}_{split}_{job['tag']}.csv" for kind in ("X", "y", "y_latent")]
        with paths[0].open("w", newline="") as xf, paths[1].open("w", newline="") as yf, paths[2].open("w", newline="") as lf:
            writers = [csv.writer(f) for f in (xf, yf, lf)]
            for w, h in zip(writers, ["Sequences", "Outcome", "Outcome"]):
                w.writerow([h])
            latent_sum = observed_sum = flips = 0
            for index in indices:
                sequence, latent, _ = clean_rows[index]
                observed = latent ^ int(noise_rng.random() < noise)
                for w, v in zip(writers, [sequence, observed, latent]):
                    w.writerow([v])
                latent_sum += latent
                observed_sum += observed
                flips += latent != observed
        hashes.update({p.name: sha256(p) for p in paths})
        count = len(indices)
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
    p.add_argument("--key-letters", help="Explicit key symbols (overrides random sampling of --keys symbols)")
    p.add_argument("--alphabet", default=ALPHABET, help="Alphabet string (default: A-Z)")
    p.add_argument("--length", type=int, default=20)
    p.add_argument("--lag", type=int, default=7)
    p.add_argument("--noise", type=float, default=0.)
    p.add_argument("--seed", type=int, default=9550)
    p.add_argument("--train-rows", type=int, default=400000)
    p.add_argument("--val-rows", type=int, default=50000)
    p.add_argument("--test-rows", type=int, default=50000)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sensitivity", action="store_true")
    p.add_argument("--sensitivity-m", type=int, nargs="+", default=[3, 6, 10])
    p.add_argument("--sensitivity-lags", type=int, nargs="+", default=[1, 3, 5, 7, 9, 10])
    p.add_argument("--sensitivity-noise", type=float, default=0.3)
    p.add_argument("--execute", action="store_true", help="write the datasets; the default prints the configuration")
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
