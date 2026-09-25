#!/usr/bin/env python3
"""Full-data ordinal and frozen-Llama XGBoost KIP baselines, three seeds.

Matches XGBoost_fraction_experiment.py's full-data tree settings and prompts,
with explicit split identity to avoid its validation/test cache-name collision.
Embeddings are kept in host RAM, shared across seeds, and never cached on disk.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/simulation/kip_ordered_recovered"
OUT = ROOT / "results/kip_xgboost_full_20260918"
SOURCE = ROOT / "src/experiments/XGBoost_fraction_experiment.py"
MODEL = ROOT / Path("data/model_cache/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08")
SEEDS = (9550, 9551, 9552)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load(tag, smoke):
    data, identities = {}, {}
    for split in ("train", "val", "test"):
        xp, yp = DATA / f"X_{split}_{tag}.csv", DATA / f"y_{split}_{tag}.csv"
        x, y = pd.read_csv(xp), pd.read_csv(yp).Outcome.to_numpy()
        expected = 400000 if split == "train" else 50000
        assert len(x) == len(y) == expected
        if smoke:
            count = 1000 if split == "train" else 256
            x, y = x.iloc[:count], y[:count]
        letters = x.Sequences.str.split("\x1f").tolist()
        assert all(len(row) == 20 for row in letters)
        assert all(len(letter) == 1 and "A" <= letter <= "Z" for row in letters for letter in row)
        assert np.isin(y, [0, 1]).all()
        data[split] = {
            "ordinal": np.array([[ord(letter) - 64 for letter in row] for row in letters], dtype=np.int32),
            "prompts": [f'Sequential events: {" ".join(row)}\nOutcome (0 or 1):' for row in letters],
            "labels": y.astype(np.int8),
        }
        identities[split] = {"features_path": str(xp), "features_sha256": sha(xp), "labels_path": str(yp), "labels_sha256": sha(yp), "rows": len(y), "positive_rate": float(y.mean())}
    assert len({record["features_sha256"] for record in identities.values()}) == 3
    for i, split in enumerate(data):
        for other in list(data)[i + 1:]:
            assert not set(data[split]["prompts"]).intersection(data[other]["prompts"])
    # Compare against the two original encoding functions, without running its driver/cache.
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in ("ordinal_encoding_1to26", "standard_narrative_prompt")]
    context = {"np": np, "pd": pd, "tqdm": lambda iterable, **kwargs: iterable}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), context)
    sample = pd.read_csv(DATA / f"X_train_{tag}.csv", nrows=16)
    np.testing.assert_array_equal(context["ordinal_encoding_1to26"](sample).to_numpy(), data["train"]["ordinal"][:16])
    assert sample.apply(context["standard_narrative_prompt"], axis=1).tolist() == data["train"]["prompts"][:16]
    return data, identities


def fit(tag, encoding, features, data, metadata, args, out):
    feature_evidence = {}
    for split, matrix in features.items():
        assert matrix.shape[0] == len(data[split]["labels"])
        assert np.isfinite(matrix).all()
        feature_evidence[split] = {"shape": list(matrix.shape), "dtype": str(matrix.dtype), "sha256_raw_array": hashlib.sha256(memoryview(np.ascontiguousarray(matrix))).hexdigest()}
    assert feature_evidence["val"]["sha256_raw_array"] != feature_evidence["test"]["sha256_raw_array"]
    y = data["train"]["labels"]
    for seed in SEEDS:
        directory = out / "runs" / encoding / tag / f"seed{seed}"
        if (directory / "metrics.json").exists():
            print(f"SKIP completed {encoding}/{tag}/{seed}", flush=True)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        parameters = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=1, gamma=0.1, reg_alpha=0.5, reg_lambda=1.5, tree_method="hist", device="cpu", n_jobs=args.threads, scale_pos_weight=float((y == 0).sum() / (y == 1).sum()), random_state=seed, verbosity=0, enable_categorical=False, early_stopping_rounds=10, eval_metric="logloss")
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "tag": tag, "encoding": encoding, "seed": seed, "smoke": args.smoke, "parameters": parameters, "features": feature_evidence, **metadata}
        write(directory / "manifest.json", manifest)
        write(out / f"status_{args.mode}.json", {"status": "fitting", "pid": os.getpid(), "encoding": encoding, "tag": tag, "seed": seed, "updated_utc": datetime.now(timezone.utc).isoformat()})
        print(f"FIT {encoding}/{tag}/seed{seed} on {len(y):,} rows, {features['train'].shape[1]} features", flush=True)
        model = xgb.XGBClassifier(**parameters)
        started = time.monotonic()
        model.fit(features["train"], y, eval_set=[(features["val"], data["val"]["labels"])], verbose=50)
        probabilities = model.predict_proba(features["test"])[:, 1]
        labels = data["test"]["labels"]
        result = dict(tag=tag, encoding=encoding, seed=seed, n_train=len(y), n_val=len(data["val"]["labels"]), n_test=len(labels), test_auc=float(roc_auc_score(labels, probabilities)), test_accuracy_at_half=float(accuracy_score(labels, probabilities >= .5)), test_f1_at_half=float(f1_score(labels, probabilities >= .5)), best_iteration=int(model.best_iteration), best_validation_logloss=float(model.best_score), fitted_rounds=int(model.get_booster().num_boosted_rounds()), elapsed_seconds=time.monotonic() - started, smoke=args.smoke)
        model.save_model(directory / "model.ubj")
        write(directory / "validation_history.json", model.evals_result())
        (directory / "booster_config.json").write_text(model.get_booster().save_config() + "\n")
        np.savez_compressed(directory / "test_predictions.npz", labels=labels, probabilities=probabilities)
        reloaded = xgb.XGBClassifier(n_jobs=args.threads)
        reloaded.load_model(directory / "model.ubj")
        np.testing.assert_array_equal(probabilities[:256], reloaded.predict_proba(features["test"][:256])[:, 1])
        result["model_reload_check"] = "passed"
        write(directory / "metrics.json", result)
        print(json.dumps(result), flush=True)
        summarize(out)
        del model, reloaded
        gc.collect()


def summarize(out):
    records = [json.loads(path.read_text()) for path in sorted((out / "runs").glob("*/*/seed*/metrics.json"))]
    if not records:
        return
    frame = pd.DataFrame(records)
    frame.to_csv(out / "per_seed.csv", index=False)
    summary = frame.groupby(["encoding", "tag"]).test_auc.agg(n="count", auc_mean="mean", auc_sample_sd="std").reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    if len(frame) == 12 and not frame.smoke.any():
        assert (summary.n == 3).all()
        table = (ROOT / "results/core_table_gap_review_20260918/table.tex").read_text()
        for encoding, label in (("basic", "XGBoost (basic)"), ("llm", "XGBoost (LLM)")):
            values = []
            for tag in ("kip_m4", "kip_m6"):
                row = summary[(summary.encoding == encoding) & (summary.tag == tag)].iloc[0]
                values.append(f"{row.auc_mean:.3f} ({row.auc_sample_sd:.3f})".replace("0.", "."))
            lines = table.splitlines()
            for i, line in enumerate(lines):
                if line.startswith(label + " &"):
                    fields = line.removesuffix(r"\\").split("&")
                    fields[-2:] = [" " + values[0] + " ", " " + values[1]]
                    lines[i] = "&".join(fields) + r"\\"
            table = "\n".join(lines) + "\n"
        (out / "table.tex").write_text(table)


def model_setup(args):
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run the LLM stage with GPU access outside the sandbox.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(9550)
    torch.set_float32_matmul_precision(args.matmul_precision)
    free, total = torch.cuda.mem_get_info()
    if free < 9 * 1024**3:
        raise RuntimeError(f"Only {free/1024**3:.1f} GiB GPU memory free; need 9 GiB to coexist safely.")
    torch.cuda.set_per_process_memory_fraction(8 * 1024**3 / total)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa").to("cuda")
    model.eval().requires_grad_(False)
    assert model.config.hidden_size == 2048
    info = {"name": "meta-llama/Llama-3.2-1B", "snapshot": str(MODEL), "files_sha256": {path.name: sha(path) for path in sorted(MODEL.iterdir()) if path.is_file()}, "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(), "weights_and_outputs_dtype": "float32", "float32_matmul_precision": args.matmul_precision, "attention_implementation": "sdpa", "gpu_allocator_limit_gib": 8, "frozen": True, "pooling": "arithmetic mean of final hidden states over all input tokens, including template and BOS; no padding", "cache": "host RAM only; distinct split identities and hashes"}
    return model, tokenizer, info


def encode(prompts, model, tokenizer, tag, split, args, out):
    import torch
    # Tokenize every row without truncation, and establish padding-free batches.
    tokenized = []
    for start in range(0, len(prompts), 4096):
        tokenized.extend(tokenizer(prompts[start:start + 4096], padding=False, truncation=False)["input_ids"])
    lengths = {len(ids) for ids in tokenized}
    if len(lengths) != 1:
        raise ValueError(f"Unexpected variable token lengths {lengths}; mean-pooling semantics need review.")
    length = lengths.pop()
    assert length <= 512
    ids = np.asarray(tokenized, dtype=np.int64)
    del tokenized
    embeddings = np.empty((len(prompts), 2048), dtype=np.float32)
    started = last = time.monotonic()
    for start in range(0, len(prompts), args.batch_size):
        batch = torch.from_numpy(ids[start:start + args.batch_size]).to("cuda")
        with torch.inference_mode():
            states = model(input_ids=batch, attention_mask=torch.ones_like(batch), use_cache=False).last_hidden_state
            pooled = states.mean(dim=1)
        end = start + len(batch)
        embeddings[start:end] = pooled.cpu().numpy()
        del states, pooled, batch
        now = time.monotonic()
        if now - last >= 30 or end == len(prompts):
            state = {"status": "embedding", "pid": os.getpid(), "tag": tag, "split": split, "completed": end, "total": len(prompts), "rows_per_second": end / (now - started), "token_length": length, "gpu_allocated_gib": torch.cuda.memory_allocated()/1024**3, "gpu_reserved_gib": torch.cuda.memory_reserved()/1024**3, "updated_utc": datetime.now(timezone.utc).isoformat()}
            write(out / f"status_{args.mode}.json", state)
            print(json.dumps(state), flush=True)
            last = now
    # Independently tokenize reference examples, and compare matching batch
    # shapes. TF32 legitimately changes arithmetic kernels for batch size one.
    checks = []
    with torch.inference_mode():
        for i in (0, min(17, len(ids) - 1)):
            one = tokenizer(prompts[i], return_tensors="pt", padding=False, truncation=False)["input_ids"].to("cuda")
            np.testing.assert_array_equal(one.cpu().numpy()[0], ids[i])
            single = model(input_ids=one, attention_mask=torch.ones_like(one), use_cache=False).last_hidden_state[0].mean(0).cpu().numpy()
            repeat = one.repeat(min(args.batch_size, len(prompts)), 1)
            matched = model(input_ids=repeat, attention_mask=torch.ones_like(repeat), use_cache=False).last_hidden_state[0].mean(0).cpu().numpy()
            np.testing.assert_allclose(embeddings[i], matched, rtol=1e-4, atol=1e-4)
            relative_l2 = float(np.linalg.norm(embeddings[i] - single) / max(np.linalg.norm(single), 1e-12))
            assert relative_l2 < 1e-3, relative_l2
            checks.append({"row": i, "matched_batch_max_absolute_difference": float(np.max(np.abs(embeddings[i] - matched))), "single_example_max_absolute_difference": float(np.max(np.abs(embeddings[i] - single))), "single_example_relative_l2_difference": relative_l2})
    return embeddings, {"token_length": length, "rows": len(prompts), "elapsed_seconds": time.monotonic() - started, "batch_vs_single_check": "passed: independent tokenization; matched batch allclose 1e-4; single-example relative L2 < 1e-3", "numerical_checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("basic", "llm", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--matmul-precision", choices=("highest", "high"), default="highest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = OUT / "smoke" / args.matmul_precision if args.smoke else OUT
    out.mkdir(parents=True, exist_ok=True)
    metadata = {"versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgb.__version__}, "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__), SOURCE)}, "min_child_weight_note": "1, matching original source; manuscript says 2", "device_note": "XGBoost hist uses CPU; original script automatically selected CUDA when available", "seeds": list(SEEDS), "selection": "validation logloss, early stopping 10; no test-based tuning", "full_data": not args.smoke}
    protocol = out / "protocol" / metadata["source_sha256"][str(Path(__file__).relative_to(ROOT))][:12]
    protocol.mkdir(parents=True, exist_ok=True)
    for path in (Path(__file__), SOURCE):
        shutil.copyfile(path, protocol / path.name)
    try:
        if args.mode in ("basic", "all"):
            for tag in ("kip_m4", "kip_m6"):
                data, identities = load(tag, args.smoke)
                fit(tag, "basic", {split: value["ordinal"] for split, value in data.items()}, data, {**metadata, "data": identities}, args, out)
        if args.mode in ("llm", "all"):
            model, tokenizer, model_info = model_setup(args)
            write(out / "llama_model.json", model_info)
            for tag in ("kip_m4", "kip_m6"):
                data, identities = load(tag, args.smoke)
                features, embedding_checks = {}, {}
                for split in ("train", "val", "test"):
                    features[split], embedding_checks[split] = encode(data[split]["prompts"], model, tokenizer, tag, split, args, out)
                fit(tag, "llm", features, data, {**metadata, "data": identities, "llama": model_info, "embedding_checks": embedding_checks}, args, out)
                del features, data
                gc.collect()
        summarize(out)
        write(out / f"status_{args.mode}.json", {"status": "completed", "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        write(out / f"status_{args.mode}.json", {"status": "failed", "pid": os.getpid(), "error": repr(error), "updated_utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
