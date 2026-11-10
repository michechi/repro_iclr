#!/usr/bin/env python3
"""Aggregate multi-seed summaries of main-experiment runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

PRETRAINED_PROTOCOL = "paper_pretrained_v1"
BASELINE_PROTOCOL = "paper_core_baselines_v1"
DEFAULT_SEEDS = (9550, 9551, 9552)
METRICS = ("auc", "f1", "precision", "recall", "accuracy", "loss")
HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_json(path):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key}")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(), parse_constant=reject_constant,
                       object_pairs_hook=unique_keys)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def hash_map(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"nonempty {name} required")
    if any(not isinstance(key, str) or not isinstance(digest, str) or not HEX.fullmatch(digest)
           for key, digest in value.items()):
        raise ValueError(f"invalid SHA256 in {name}")
    return value


def same(left, right, field):
    if left.get(field) != right.get(field) or field not in left or field not in right:
        raise ValueError(f"metrics/manifest mismatch: {field}")


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"positive integer {name} required")
    return value


def data_identity(data, label_convention):
    """Group on source dataset files rather than seed-specific training subsamples."""
    if not isinstance(data, dict) or set(data) != {"train", "val", "test"}:
        raise ValueError("train/val/test data metadata required")
    result = {}
    for split, evidence in data.items():
        if not isinstance(evidence, dict) or evidence.get("split") != split:
            raise ValueError(f"invalid {split} data metadata")
        if evidence.get("label_convention") != label_convention:
            raise ValueError(f"{split} label convention disagrees with metrics")
        rule = evidence.get("rule")
        if not isinstance(rule, dict) or rule.get("task") not in {"oc", "kip", "gpc"}:
            raise ValueError(f"{split} rule identity missing")
        if not isinstance(rule.get("keys"), (str, list)) or not rule["keys"]:
            raise ValueError(f"{split} key identity missing")
        result[split] = {
            "source_sha256": hash_map(evidence.get("source_sha256"), f"{split} source_sha256"),
            "rows": positive_integer(evidence.get("rows"), f"{split} source rows"),
            "source_rows": evidence.get("source_rows", evidence.get("rows")),
            "returned_rows": evidence.get("returned_rows", evidence.get("rows")),
            "source_selection": evidence.get("source_selection"),
            "task": evidence.get("task"), "rule": rule,
            "label_convention": label_convention,
            "dataset_manifest_sha256": evidence.get("dataset_manifest_sha256"),
        }
    if any(item["rule"] != result["train"]["rule"] for item in result.values()):
        raise ValueError("rules differ across data splits")
    return result


def model_identity(model):
    """Extract invariant model metadata."""
    if not isinstance(model, dict):
        raise ValueError("model metadata required")
    keys = ("model_key", "model_name", "family", "tuning", "model_revision",
            "model_config_sha256", "tokenizer_files_sha256", "actual_architecture",
            "pretrained_parameter_count", "objective", "decoder_head", "lora",
            "quantization", "compute_dtype", "head_dtype", "gradient_checkpointing",
            "lm_reduction", "environment_versions", "weight_files_sha256",
            "model_weights_sha256", "snapshot_files_sha256", "pretrained_weights_sha256",
            "pretrained_weight_files_sha256")
    return {key: model[key] for key in keys if key in model}


def normalize(metrics_path):
    """Validate one run and return (record, exclusion_reason)."""
    metrics_path = Path(metrics_path)
    metrics = load_json(metrics_path)
    protocol = metrics.get("protocol_id")
    if protocol not in {PRETRAINED_PROTOCOL, BASELINE_PROTOCOL}:
        return None, "unsupported protocol"
    if metrics.get("is_smoke") is True:
        return None, "smoke diagnostic"
    if metrics.get("status") != "completed":
        return None, "run not completed"
    if metrics.get("is_smoke") is not False:
        raise ValueError("explicit is_smoke=false required")
    if metrics.get("threshold_selected_on") != "validation":
        raise ValueError("validation-only metric threshold required")
    seed = metrics.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("integer seed required")
    tag = metrics.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("task tag required")
    rows = metrics.get("rows")
    if not isinstance(rows, dict) or set(rows) != {"train", "val", "test"}:
        raise ValueError("split row counts required")
    for split, count in rows.items():
        positive_integer(count, f"{split} rows")
    if rows["val"] != 50000 or rows["test"] != 50000:
        raise ValueError("core summaries require fixed 50K validation and test sets")
    manifest_path = metrics_path.with_name("manifest.json")
    expected_hash = metrics.get("manifest_sha256")
    if metrics.get("protocol") == "paper_v2":
        expected_hash = metrics.get("artifact_sha256", {}).get("manifest.json")
    if not isinstance(expected_hash, str) or not HEX.fullmatch(expected_hash):
        raise ValueError("metrics must authenticate manifest.json with SHA256")
    if sha256(manifest_path) != expected_hash:
        raise ValueError("manifest SHA256 mismatch")
    manifest = load_json(manifest_path)
    sources = hash_map(manifest.get("source_sha256"), "trainer source_sha256")
    if metrics.get("protocol") == "paper_v2":
        writer = "kip_data_efficiency"
        config = manifest.get("config")
        if not isinstance(config, dict) or config.get("protocol") != "paper_v2":
            raise ValueError("KIP paper_v2 config required")
        for field in ("protocol_id", "arm", "tag", "seed", "fraction", "is_smoke", "rows", "label_convention"):
            same(metrics, config, field)
        if metrics.get("selected_by") != "validation_loss" or metrics.get("stopping_rule") != "validation_loss_patience":
            raise ValueError("KIP validation-loss stopping config required")
        if config.get("effective_batch_size") != 16:
            raise ValueError("KIP effective batch must be 16")
        fraction = metrics["fraction"]
        if fraction not in (.01, .1, .3, .5, .75, 1.) or rows["train"] != round(400000 * fraction):
            raise ValueError("KIP fraction/row-count mismatch")
        model = model_identity(manifest.get("model"))
        original_model = config.get("model_source")
        if not isinstance(original_model, dict):
            raise ValueError("KIP pretrained checkpoint identity required")
        model["checkpoint_source"] = {key: value for key, value in original_model.items() if key != "snapshot"}
        excluded = {"seed", "tag", "rows", "data", "model_source", "index_sha256", "fraction", "is_smoke"}
        recipe = {key: value for key, value in config.items() if key not in excluded}
        selected_indices = config.get("index_sha256")
        data = config.get("data")
    elif protocol == PRETRAINED_PROTOCOL:
        writer = "paper_pretrained_training"
        for field in ("protocol_id", "tag", "seed", "is_smoke", "rows"):
            same(metrics, manifest, field)
        if metrics.get("selected_by") != "validation_loss" or manifest.get("stopping_statistic") != "validation_loss":
            raise ValueError("pretrained validation-loss stopping config required")
        model = model_identity(manifest.get("model"))
        if not isinstance(metrics.get("model"), dict):
            raise ValueError("pretrained metrics model specification required")
        for field in ("model_key", "model_name", "family", "tuning"):
            same(metrics["model"], model, field)
        if manifest.get("requested_train_size") != rows["train"]:
            raise ValueError("requested training size differs from completed run")
        excluded = {"seed", "tag", "rows", "data", "model", "source_sha256", "selected_indices_sha256",
                    "is_smoke", "requested_train_size"}
        recipe = {key: value for key, value in manifest.items() if key not in excluded}
        config = {"recipe": recipe, "model": model}
        selected_indices = manifest.get("selected_indices_sha256")
        data = manifest.get("data")
    else:
        writer = "paper_baselines"
        for field in ("protocol_id", "model", "tag", "seed", "train_size"):
            same(metrics, manifest, field)
        if manifest.get("threshold_selected_on") != "validation" or manifest["train_size"] != rows["train"]:
            raise ValueError("baseline selection/size metadata mismatch")
        config = manifest.get("config")
        if not isinstance(config, dict) or config.get("model") != metrics["model"]:
            raise ValueError("baseline model configuration required")
        model = {"model_key": metrics["model"]}
        if metrics["model"] == "xgboost-llm":
            model["frozen_checkpoint"] = manifest.get("frozen_snapshot_files_sha256") or manifest.get("frozen_model")
            if model["frozen_checkpoint"] is None:
                model["model_path"] = manifest.get("model_path")
        recipe = {"config": config, "feature_family": manifest.get("feature_family"), "k": manifest.get("k")}
        if "xgboost_parameters" in manifest:
            recipe["xgboost_parameters"] = {key: value for key, value in manifest["xgboost_parameters"].items()
                                            if key not in {"random_state", "scale_pos_weight"}}
            recipe["xgboost_parameters"]["scale_pos_weight"] = "training Nnegative/Npositive, recomputed per seed"
        selected_indices = manifest.get("selected_indices_sha256")
        data = manifest.get("data")
    if "config_sha256" in metrics and metrics["config_sha256"] != canonical_hash(config):
        raise ValueError("config SHA256 mismatch")
    if selected_indices is not None:
        hash_map(selected_indices, "selected-index hashes")
        if set(selected_indices) != {"train", "val", "test"}:
            raise ValueError("selected-index hashes must cover all splits")
        for split, digest in selected_indices.items():
            if sha256(metrics_path.with_name(f"{split}_indices.npy")) != digest:
                raise ValueError(f"selected-index SHA256 mismatch: {split}")
    label = metrics.get("label_convention")
    if label not in {"oc", "gpc", "odd_inversions"}:
        raise ValueError("label convention missing")
    dataset = data_identity(data, label)
    if any(dataset[split]["rows"] != expected for split, expected in (("train", 400000), ("val", 50000), ("test", 50000))):
        raise ValueError("core source dataset must have 400K/50K/50K rows")
    test = metrics.get("test")
    if not isinstance(test, dict):
        raise ValueError("test metrics missing")
    values = {}
    for name in METRICS:
        value = test.get(name)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid test {name}")
            if name != "loss" and not 0 <= value <= 1:
                raise ValueError(f"test {name} must lie in [0,1]")
            if name == "loss" and value < 0:
                raise ValueError("test loss cannot be negative")
        values[name] = value
    if any(values[name] is None for name in ("auc", "f1", "precision", "recall", "accuracy")):
        raise ValueError("main classification metrics incomplete")
    group = {"writer": writer, "protocol_id": protocol, "task_tag": tag,
             "task": dataset["train"]["rule"]["task"], "train_rows": rows["train"],
             "validation_rows": rows["val"], "test_rows": rows["test"],
             "label_convention": label, "model": model, "recipe": recipe,
             "data_identity": dataset, "source_sha256": sources}
    return {"group": group, "group_sha256": canonical_hash(group), "seed": seed,
            "test": values, "metrics_path": str(metrics_path.resolve()),
            "metrics_sha256": sha256(metrics_path), "manifest_sha256": expected_hash,
            "config_sha256": canonical_hash(config), "selected_indices_sha256": selected_indices}, None


def summarize(run_roots, seeds=DEFAULT_SEEDS):
    seeds = tuple(seeds)
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
        raise ValueError("three distinct integer seeds required")
    paths = set()
    for root in run_roots:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"run root is not a directory: {root}")
        paths.update(path.resolve() for path in root.rglob("metrics.json"))
    groups, excluded, rejected = {}, [], []
    for path in sorted(paths):
        try:
            record, reason = normalize(path)
        except (OSError, ValueError, TypeError, KeyError) as error:
            rejected.append({"metrics_path": str(path), "reason": str(error)})
            continue
        if reason:
            excluded.append({"metrics_path": str(path), "reason": reason})
            continue
        if record["seed"] not in seeds:
            excluded.append({"metrics_path": str(path), "reason": "seed is outside the requested three"})
            continue
        group = groups.setdefault(record["group_sha256"], {"identity": record["group"], "runs": {}})
        if record["seed"] in group["runs"]:
            previous = group["runs"][record["seed"]]["metrics_path"]
            raise ValueError(f"duplicate seed {record['seed']} in one group: {previous} and {path}")
        group["runs"][record["seed"]] = {key: value for key, value in record.items() if key not in {"group", "group_sha256"}}
    output = []
    for digest, group in sorted(groups.items()):
        runs = group["runs"]
        complete = set(runs) == set(seeds)
        aggregates = {}
        for metric in METRICS:
            values = [runs[seed]["test"][metric] for seed in seeds
                      if seed in runs and runs[seed]["test"][metric] is not None]
            aggregates[metric] = {
                "n_observed": len(values),
                "observed_mean": statistics.mean(values) if values else None,
                "sample_sd": statistics.stdev(values) if len(values) >= 2 else None,
                "three_seed_mean": statistics.mean(values) if complete and len(values) == 3 else None,
                "three_seed_sample_sd": statistics.stdev(values) if complete and len(values) == 3 else None,
            }
        output.append({"group_sha256": digest, **group["identity"], "complete": complete,
                       "expected_seeds": list(seeds), "missing_seeds": [seed for seed in seeds if seed not in runs],
                       "per_seed": {str(seed): runs.get(seed) for seed in seeds}, "metrics": aggregates})
    return {"schema_version": 1, "expected_seeds": list(seeds),
            "groups": output, "excluded": excluded, "rejected": rejected,
            "group_count": len(output), "complete_group_count": sum(row["complete"] for row in output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, action="append", required=True,
                        help="repeat for multiple roots; overlapping roots are de-duplicated")
    parser.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS)
    parser.add_argument("--output", type=Path, help="optional new JSON file")
    args = parser.parse_args(argv)
    try:
        result = summarize(args.run_root, args.seeds)
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                handle.write(encoded)
        else:
            print(encoded, end="")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 1 if result["rejected"] else 0


if __name__ == "__main__":
    sys.exit(main())
