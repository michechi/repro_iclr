#!/usr/bin/env python3
"""Complete historical ordered BERT-LoRA seeds 9551/9552 for KIP M4/M6.

Retains historical fixed padding (M4=512, M6=64), batch 16, FP32, optimizer,
schedule, and validation-loss early stopping. Epoch checkpoints are resumable.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results/kip_bert_repeats_20260918"
DATA = ROOT / "data/simulation/kip_ordered_recovered"
MODEL = ROOT / Path("data/model_cache/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594")
SOURCE = ROOT / "src/experiments/BERT_fraction_experiment.py"
JOBS = [(tag, seed) for tag in ("kip_m6", "kip_m4") for seed in (9551, 9552)]


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def checkpoint(path, payload):
    import torch
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def trainable(model):
    return {k: p.detach().cpu().clone() for k, p in model.named_parameters() if p.requires_grad}


def restore(model, state):
    expected = {k for k, p in model.named_parameters() if p.requires_grad}
    assert set(state) == expected
    assert not model.load_state_dict(state, strict=False).unexpected_keys


def rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


class FixedTokens:
    """Pre-tokenization changes no model inputs; padding matches old dataset."""
    def __init__(self, sequences, labels, tokenizer, width):
        import numpy as np
        import torch
        from src.experiments import BERT_fraction_experiment as old
        self.width = width
        self.pad = tokenizer.pad_token_id
        self.ids = []
        self.labels = np.asarray(labels, dtype=np.int64)
        for start in range(0, len(sequences), 2048):
            prompts = [old.standard_narrative_prompt({"Sequences": s}) for s in sequences[start:start + 2048]]
            self.ids.extend(tokenizer(prompts, padding=False, truncation=False)["input_ids"])
        assert max(map(len, self.ids)) <= width
        reference = old.SequenceClassificationDataset(
            [old.standard_narrative_prompt({"Sequences": s}) for s in sequences[:16]],
            self.labels[:16], tokenizer, max_length=width)
        for i in range(min(16, len(self.ids))):
            for key, value in self[i].items():
                torch.testing.assert_close(value, reference[i][key], rtol=0, atol=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        import torch
        tokens = self.ids[index]
        ids = torch.full((self.width,), self.pad, dtype=torch.long)
        mask = torch.zeros(self.width, dtype=torch.long)
        ids[:len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        mask[:len(tokens)] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": torch.tensor(self.labels[index], dtype=torch.long)}


def progress(directory, **values):
    state = {"pid": os.getpid(), "updated_unix": time.time(), **values}
    write(directory / "status.json", state)
    print(json.dumps(state), flush=True)


def evaluate(model, loader, directory, label):
    import numpy as np
    import torch
    model.eval()
    predictions, labels, losses = [], [], []
    started = last = time.monotonic()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            output = model(**{k: v.to("cuda") for k, v in batch.items()})
            losses.append(output.loss.item())
            predictions.append(torch.softmax(output.logits, dim=-1)[:, 1].cpu().numpy())
            labels.append(batch["labels"].numpy())
            if time.monotonic() - last >= 45:
                progress(directory, status="evaluating", split=label, batches=step + 1, total_batches=len(loader))
                last = time.monotonic()
    probabilities, truths = np.concatenate(predictions), np.concatenate(labels)
    assert np.isfinite(probabilities).all()
    return sum(losses) / len(losses), probabilities, truths, time.monotonic() - started


def arguments(tag, seed, smoke):
    return argparse.Namespace(model_name=str(MODEL), cache_dir=str(ROOT / "data/model_cache"),
        tiny=False, cold_start=False, peft=True, seed=seed, batch_size=16,
        gradient_accumulation_steps=1, epochs=2 if smoke else 20, patience=3,
        max_length=512 if tag == "kip_m4" else 64, lr=2e-5, early="loss")


def run_one(tag, seed, output, smoke=False):
    import numpy as np
    import pandas as pd
    import peft
    import sklearn
    import torch
    import transformers
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from src.experiments import BERT_fraction_experiment as old

    directory = output / "runs" / tag / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metrics.json").exists():
        print(f"SKIP completed {tag}/{seed}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: this run needs GPU access outside the sandbox")
    torch.set_num_threads(4)
    free, total = torch.cuda.mem_get_info()
    if free < 10 * 1024**3:
        raise RuntimeError(f"Need at least 10 GiB free GPU memory; found {free/1024**3:.2f}")
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / total)
    args = arguments(tag, seed, smoke)
    old.set_seed(seed)
    tokenizer = old.load_tokenizer(str(MODEL), args.cache_dir)
    loaders, provenance = {}, {}
    sequence_sets = {}
    for split, full_count in (("train", 400000), ("val", 50000), ("test", 50000)):
        xp, yp = DATA / f"X_{split}_{tag}.csv", DATA / f"y_{split}_{tag}.csv"
        sequences = pd.read_csv(xp).Sequences.tolist()
        labels = pd.read_csv(yp).Outcome.to_numpy()
        assert len(sequences) == len(labels) == full_count
        assert np.isin(labels, [0, 1]).all()
        assert all(len(s.split("\x1f")) == 20 and all(len(c) == 1 and "A" <= c <= "Z" for c in s.split("\x1f")) for s in sequences)
        if smoke:
            count = 128 if split == "train" else 64
            sequences, labels = sequences[:count], labels[:count]
        sequence_sets[split] = set(sequences)
        dataset = FixedTokens(sequences, labels, tokenizer, args.max_length)
        loaders[split] = DataLoader(dataset, batch_size=16, shuffle=split == "train", pin_memory=True, num_workers=0)
        provenance[split] = dict(rows=len(labels), features_sha256=sha(xp), labels_sha256=sha(yp),
            max_unpadded_tokens=max(map(len, dataset.ids)), input_encoding_reference_check="passed")
    assert not sequence_sets["train"] & sequence_sets["val"]
    assert not sequence_sets["train"] & sequence_sets["test"]
    assert not sequence_sets["val"] & sequence_sets["test"]
    del sequence_sets
    model = old.load_model(args, tokenizer).to("cuda")
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 443906
    assert all(p.dtype == torch.float32 for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loaders["train"])
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=int(.06 * total_steps), num_training_steps=total_steps)
    manifest = dict(tag=tag, seed=seed, smoke=smoke, arguments=vars(args), data=provenance,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__, sklearn=sklearn.__version__, numpy=np.__version__),
        gpu=torch.cuda.get_device_name(), source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), SOURCE)},
        pretrained_files_sha256={p.name: sha(p) for p in MODEL.iterdir() if p.is_file()},
        dtype="float32", autocast=False, float32_matmul_precision=torch.get_float32_matmul_precision(),
        padding="fixed max_length; M4=512, M6=64, same as respective historical seed9550",
        optimizer=dict(name="AdamW", lr=args.lr, weight_decay=.01, betas=[.9, .999], eps=1e-8),
        schedule=dict(total_steps=total_steps, warmup_steps=int(.06*total_steps)),
        trainable_parameters=443906, total_parameters=sum(p.numel() for p in model.parameters()),
        historical_runtime_note="Hyperparameters match recorded seed9550. Historical library versions/checkpoint revision were not recorded; no bitwise reproduction claim.")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Resume configuration differs"
    else:
        write(manifest_path, manifest)
    history, best_loss, best_epoch, bad_epochs = [], math.inf, 0, 0
    last_path, best_path = directory / "last.pt", directory / "best_trainable.pt"
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        restore(model, saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best_loss, best_epoch, bad_epochs = (saved[k] for k in ("history", "best_loss", "best_epoch", "bad_epochs"))
        restore_rng(saved["rng"])
        print(f"RESUME after epoch {len(history)}", flush=True)
    initial_rng = rng() if smoke else None
    initial_state = trainable(model) if smoke else None
    for epoch in range(len(history) + 1, args.epochs + 1):
        if bad_epochs >= args.patience:
            break
        model.train()
        accumulated_loss = 0.
        optimizer.zero_grad()
        started = last = time.monotonic()
        for step, batch in enumerate(loaders["train"]):
            output_tensor = model(**{k: v.to("cuda") for k, v in batch.items()})
            loss = output_tensor.loss / args.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.detach().item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if time.monotonic() - last >= 45 or step + 1 == len(loaders["train"]):
                elapsed = time.monotonic() - started
                progress(directory, status="training", tag=tag, seed=seed, epoch=epoch, batch=step + 1,
                    total_batches=len(loaders["train"]), rows_per_second=(step + 1)*16/elapsed,
                    training_loss_so_far=accumulated_loss/(step + 1), gpu_reserved_gib=torch.cuda.memory_reserved()/1024**3)
                last = time.monotonic()
        training_seconds = time.monotonic() - started
        val_loss, probabilities, labels, validation_seconds = evaluate(model, loaders["val"], directory, "validation")
        assert math.isfinite(val_loss)
        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            checkpoint(best_path, trainable(model))
        else:
            bad_epochs += 1
        row = dict(epoch=epoch, train_loss=accumulated_loss/len(loaders["train"]), val_loss=val_loss,
            val_auc=float(roc_auc_score(labels, probabilities)), improved=improved,
            training_seconds=training_seconds, validation_seconds=validation_seconds)
        history.append(row)
        write(directory / "history.json", history)
        checkpoint(last_path, dict(model=trainable(model), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng(), history=history, best_loss=best_loss, best_epoch=best_epoch, bad_epochs=bad_epochs))
        print("EPOCH " + json.dumps(row), flush=True)
    restore(model, torch.load(best_path, map_location="cpu", weights_only=True))
    _, val_probabilities, val_labels, _ = evaluate(model, loaders["val"], directory, "selected_validation")
    threshold, _ = old.find_optimal_threshold(val_labels, val_probabilities)
    _, test_probabilities, test_labels, _ = evaluate(model, loaders["test"], directory, "test")
    np.testing.assert_array_equal(test_labels, loaders["test"].dataset.labels)
    np.savez_compressed(directory / "predictions.npz", labels=test_labels, probabilities=test_probabilities,
        validation_labels=val_labels, validation_probabilities=val_probabilities)
    result = dict(tag=tag, seed=seed, smoke=smoke, test_auc=float(roc_auc_score(test_labels, test_probabilities)),
        best_epoch=best_epoch, best_validation_loss=best_loss, epochs_completed=len(history),
        n_train=len(loaders["train"].dataset), n_val=len(val_labels), n_test=len(test_labels), threshold=float(threshold))
    if smoke:
        # Compare the complete selected model against the original training loop.
        expected = trainable(model)
        restore(model, initial_state)
        restore_rng(initial_rng)
        old.train_and_evaluate(model, loaders["train"], loaders["val"], args, torch.device("cuda"))
        for name, parameter in trainable(model).items():
            torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        result["original_training_loop_comparison"] = "exact trainable-state equality after two epochs and validation selection"
    write(directory / "metrics.json", result)
    progress(directory, status="completed", **result)
    del model, optimizer, scheduler, loaders
    gc.collect()
    torch.cuda.empty_cache()


def summarize(output):
    import re
    import numpy as np
    import pandas as pd
    records = []
    for tag in ("kip_m4", "kip_m6"):
        log = ROOT / f"logs/kip/BERT_{tag}_ordered_9550.log"
        match = re.search(rf"DONE BERT {tag} ordered seed=9550: test_auc=([.0-9]+)", log.read_text())
        assert match
        records.append(dict(tag=tag, seed=9550, test_auc=float(match[1]), source=str(log), precision="four-decimal historical log"))
        for seed in (9551, 9552):
            path = output / "runs" / tag / f"seed{seed}" / "metrics.json"
            if path.exists():
                result = json.loads(path.read_text())
                assert not result["smoke"]
                records.append(dict(tag=tag, seed=seed, test_auc=result["test_auc"], source=str(path), precision="saved-prediction full precision"))
    frame = pd.DataFrame(records)
    frame.to_csv(output / "per_seed.csv", index=False)
    rows = []
    for tag in ("kip_m4", "kip_m6"):
        values = frame[frame.tag == tag].test_auc.to_numpy()
        if len(values) == 3:
            rows.append(dict(tag=tag, n=3, auc_mean=float(np.mean(values)), auc_sample_sd=float(np.std(values, ddof=1))))
    if rows:
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    if len(rows) == 2:
        source_table = ROOT / "results/kip_xgboost_full_20260918/table.tex"
        table = source_table.read_text()
        values = [f"{r['auc_mean']:.3f} ({r['auc_sample_sd']:.3f})".replace("0.", ".") for r in rows]
        lines = table.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("BERT &"):
                fields = line.removesuffix(r"\\").split("&")
                fields[-2:] = [" " + values[0] + " ", " " + values[1]]
                lines[i] = "&".join(fields) + r"\\"
        (output / "table.tex").write_text("\n".join(lines) + "\n")


def supervise(output, smoke):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "matrix.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [(tag, 9551) for tag in ("kip_m6", "kip_m4")] if smoke else JOBS
        protocol = output / "protocol" / sha(Path(__file__))[:12]
        protocol.mkdir(parents=True, exist_ok=True)
        for path in (Path(__file__), SOURCE, ROOT / "scripts/verify_kip_bert_repeats.py"):
            shutil.copyfile(path, protocol / path.name)
        write(output / "matrix.json", dict(jobs=jobs, smoke=smoke, started_unix=time.time(),
            historical_seeds=9550, new_seeds=[9551, 9552], gpu_allocator_limit_gib=10))
        state = dict(status="running", pid=os.getpid(), completed=0, total=len(jobs))
        try:
            for tag, seed in jobs:
                state.update(current=dict(tag=tag, seed=seed), updated_unix=time.time())
                write(output / "matrix_status.json", state)
                command = [sys.executable, "-B", "-u", str(Path(__file__)), "--worker", "--tag", tag,
                    "--seed", str(seed), "--output", str(output)] + (["--smoke"] if smoke else [])
                (output / "logs").mkdir(exist_ok=True)
                logfile = output / "logs" / f"{tag}_seed{seed}.log"
                with logfile.open("a", buffering=1) as handle:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                        env={**os.environ, "PYTHONHASHSEED": str(seed)})
                    state.update(child_pid=process.pid, log=str(logfile))
                    write(output / "matrix_status.json", state)
                    result = process.wait()
                if result:
                    raise RuntimeError(f"{tag}/{seed} exited {result}; see {logfile}")
                assert (output / "runs" / tag / f"seed{seed}" / "metrics.json").exists()
                state["completed"] += 1
                if not smoke:
                    summarize(output)
            if not smoke:
                from verify_kip_bert_repeats import verify
                verify(output)
            state.update(status="completed", updated_unix=time.time())
        except BaseException as error:
            state.update(status="failed", error=repr(error), updated_unix=time.time())
            raise
        finally:
            write(output / "matrix_status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tag", choices=["kip_m4", "kip_m6"])
    parser.add_argument("--seed", type=int, choices=[9551, 9552])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    output = args.output / "smoke" if args.smoke and args.output == OUT else args.output
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
        TOKENIZERS_PARALLELISM="false", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if args.status:
        print((output / "matrix_status.json").read_text())
    elif args.worker:
        assert args.tag and args.seed
        try:
            run_one(args.tag, args.seed, output, args.smoke)
        except BaseException as error:
            progress(output / "runs" / args.tag / f"seed{args.seed}", status="failed", error=repr(error))
            raise
    elif args.background:
        output.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-B", "-u", str(Path(__file__)), "--output", str(output)] + (["--smoke"] if args.smoke else [])
        with (output / "supervisor.log").open("a") as handle:
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=child.pid, output=str(output))))
    else:
        supervise(output, args.smoke)


if __name__ == "__main__":
    main()
