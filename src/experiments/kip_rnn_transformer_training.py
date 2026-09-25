"""Durable, scratch RNNTransformer runs on the fixed ordered KIP splits.

Uses the repository's code-optimal BiLSTM/Transformer architecture. Letters are
encoded A=1..Z=26, reserving 0 for padding: the legacy dataset conflated A with
padding, which would hide one of KIP-M4's keys from Transformer attention.
Selection is strict validation F1 improvement; test is evaluated after fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

# Must precede the first CUDA context for deterministic cuBLAS operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.experiments.DL_TR_baselines_experiment import (
    OPTIMAL_CONFIGS, OPTIMAL_LR, RNNTransformerClassifier,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {"train": 400000, "val": 50000, "test": 50000}
KEYS = {"kip_m4": "COTA", "kip_m6": "GHLYIZ"}
SMOKE_COUNTS = {"train": 256, "val": 512, "test": 512}
METRIC_NAMES = ("auc", "f1", "precision", "recall", "accuracy", "loss")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def save_torch(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def encode_sequence(sequence, tag):
    letters = sequence.split("\x1f")
    if len(letters) != 20 or any(len(c) != 1 or not "A" <= c <= "Z" for c in letters):
        raise ValueError("expected exactly 20 uppercase letters separated by US")
    keys = KEYS[tag]
    if any(letters.count(c) != 1 for c in keys):
        raise ValueError("every KIP key must occur exactly once")
    ranks = [keys.index(c) for c in letters if c in keys]
    inversions = sum(a > b for i, a in enumerate(ranks) for b in ranks[i + 1:])
    return [ord(c) - ord("A") + 1 for c in letters], int(inversions % 2 == 0)


def read_split(directory, tag, split):
    xp, yp = (Path(directory) / f"{kind}_{split}_{tag}.csv" for kind in ("X", "y"))
    expected = EXPECTED_COUNTS[split]
    encoded = np.empty((expected, 20), dtype=np.uint8)
    labels = np.empty(expected, dtype=np.int64)
    # Hash before and after loading to detect edits while input is read.
    hashes = {str(p.resolve()): sha256(p) for p in (xp, yp)}
    recovery_path = Path(directory) / "recovery_manifest.json"
    recovery_hash = None
    if recovery_path.exists():
        recovery_hash = sha256(recovery_path)
        recorded = json.loads(recovery_path.read_text())["output_files_sha256"]
        if any(recorded.get(Path(path).name) != digest for path, digest in hashes.items()):
            raise ValueError("split does not match the canonical ordered-data recovery manifest")
    rows = 0
    with xp.open(newline="") as xh, yp.open(newline="") as yh:
        xr, yr = csv.DictReader(xh), csv.DictReader(yh)
        if xr.fieldnames != ["Sequences"] or yr.fieldnames != ["Outcome"]:
            raise ValueError("unexpected split columns")
        for i, (x, y) in enumerate(zip(xr, yr, strict=True)):
            if i >= expected:
                raise ValueError(f"too many rows in {tag}/{split}")
            tokens, parity_label = encode_sequence(x["Sequences"], tag)
            numeric = float(y["Outcome"])
            if numeric not in (0., 1.) or int(numeric) != parity_label:
                raise ValueError(f"KIP parity/label mismatch in {tag}/{split}, row {i}")
            encoded[i], labels[i] = tokens, int(numeric)
            rows += 1
    if rows != expected:
        raise ValueError(f"expected {expected} rows in {tag}/{split}, got {rows}")
    if len(np.unique(labels)) != 2:
        raise ValueError(f"{tag}/{split} must contain both classes")
    for path, digest in hashes.items():
        if sha256(path) != digest:
            raise ValueError(f"input changed during loading: {path}")
    if recovery_hash is not None and sha256(recovery_path) != recovery_hash:
        raise ValueError("recovery manifest changed during loading")
    return encoded, labels, dict(rows=rows, positives=int(labels.sum()), sha256=hashes,
                                recovery_manifest_sha256=recovery_hash,
                                sequence_length=20, parity_verified_rows=rows)


def model_config():
    return dict(OPTIMAL_CONFIGS["RNNTransformer"], vocab_size=27, max_seq_length=20)


def configure_determinism(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def configure_device(name, gpu_memory_mib, seed):
    # The allocator API requires an explicit index, unlike Tensor.to("cuda").
    device = torch.device("cuda:0" if name == "cuda" else name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        cap = gpu_memory_mib * 1024**2
        total = torch.cuda.get_device_properties(device).total_memory
        if cap >= total:
            raise ValueError("GPU allocator cap must be below GPU total memory")
        torch.cuda.set_per_process_memory_fraction(cap / total, device)
        torch.cuda.manual_seed_all(seed)
    return device


def classification_metrics(labels, logits):
    """CPU metrics from saved logits; argmax ties go to class 0 as historically."""
    labels, logits = np.asarray(labels), np.asarray(logits)
    if labels.ndim != 1 or logits.shape != (len(labels), 2) or not len(labels):
        raise ValueError("invalid prediction shapes")
    if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError("both binary classes are required")
    if not np.isfinite(logits).all():
        raise ValueError("non-finite logits")
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True).astype(np.float64)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probs = np.exp(log_probs[:, 1])
    pred = logits.argmax(axis=1)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics = dict(auc=float(roc_auc_score(labels, probs)),
                   f1=float(f1_score(labels, pred, zero_division=0)),
                   precision=float(precision_score(labels, pred, zero_division=0)),
                   recall=float(recall_score(labels, pred, zero_division=0)),
                   accuracy=float(accuracy_score(labels, pred)),
                   loss=float(-log_probs[np.arange(len(labels)), labels.astype(int)].mean()),
                   confusion_matrix=cm.tolist(), tn=tn, fp=fp, fn=fn, tp=tp,
                   n=int(len(labels)), positives=int(labels.sum()))
    return metrics, probs, pred


class EarlyStopping:
    def __init__(self, patience, best_f1=-1., best_epoch=0, bad_epochs=0):
        if patience < 1:
            raise ValueError("patience must be positive")
        self.patience, self.best_f1 = patience, best_f1
        self.best_epoch, self.bad_epochs = best_epoch, bad_epochs

    def update(self, val_f1, epoch):
        if not math.isfinite(val_f1) or not 0 <= val_f1 <= 1:
            raise ValueError("validation F1 must be finite and in [0,1]")
        improved = val_f1 > self.best_f1
        if improved:
            self.best_f1, self.best_epoch, self.bad_epochs = val_f1, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def stopped(self):
        return self.bad_epochs >= self.patience


def evaluate(model, tokens, labels, batch_size, device):
    model.eval()
    logits = np.empty((len(labels), 2), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch = torch.from_numpy(tokens[start:start + batch_size]).to(device=device, dtype=torch.long)
            logits[start:start + batch_size] = model(batch).float().cpu().numpy()
    return classification_metrics(labels, logits)[0], logits


def rng_state(generator, device):
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                 data_generator=generator.get_state())
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["data_generator"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def manifest_for(args, provenance, datasets):
    sources = [Path(__file__).resolve(), ROOT / "src/experiments/DL_TR_baselines_experiment.py"]
    config = dict(model="RNNTransformer", architecture=model_config(), tag=args.tag, seed=args.seed,
                  mode="ordered", tuning="from_scratch_all_parameters", train_fraction=1.0,
                  n_train=len(datasets["train"][1]), n_val=len(datasets["val"][1]), n_test=len(datasets["test"][1]),
                  is_smoke=args.smoke, device=args.device, gpu_memory_mib=args.gpu_memory_mib,
                  threads=args.threads, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                  eval_batch_size=args.eval_batch_size, optimizer="Adam", learning_rate=OPTIMAL_LR["RNNTransformer"],
                  weight_decay=0., mixed_precision=False, deterministic_algorithms=True, tf32=False,
                  selection="strict_validation_f1_improvement", prediction_rule="argmax_logits_ties_class0",
                  test_policy="one_evaluation_after_training_no_test_stopping",
                  encoding="A=1 through Z=26; PAD=0; no truncation or actual padding",
                  correction="legacy A=0 collided with model padding_idx and attention padding mask",
                  cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                  code_recipe="DL_TR_baselines_experiment.OPTIMAL_CONFIGS/OPTIMAL_LR",
                  scratch_initialization=True, smoke_selection="first_rows" if args.smoke else None)
    manifest = dict(schema_version=1, config=config, data=provenance,
                    source_sha256={str(path): sha256(path) for path in sources},
                    environment=dict(python=platform.python_version(), torch=torch.__version__,
                                     numpy=np.__version__, sklearn=sklearn_version))
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def verify_saved_predictions(output, manifest):
    """Reconstruct all reported metrics from persisted logits on CPU."""
    with np.load(Path(output) / "test_predictions.npz", allow_pickle=False) as saved:
        expected = manifest["config"]["n_test"]
        if not np.array_equal(saved["row_index"], np.arange(expected)):
            raise ValueError("saved test row ordering mismatch")
        metrics, probs, pred = classification_metrics(saved["labels"], saved["logits"])
        if not np.array_equal(saved["predictions"], pred) or not np.array_equal(saved["probabilities"], probs):
            raise ValueError("saved probabilities/predictions do not match saved logits")
        if str(saved["manifest_sha256"].item()) != manifest["manifest_sha256"]:
            raise ValueError("prediction manifest mismatch")
    return metrics


def train(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir
    # Duplicate invocations must not mutate the same run, even before GPU work.
    import fcntl
    lock = (output / ".trainer.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another trainer owns {output}") from exc
    try:
        return _train_locked(args)
    finally:
        lock.close()


def _train_locked(args):
    output = args.output_dir
    datasets, provenance = {}, {}
    for split in EXPECTED_COUNTS:
        tokens, labels, prov = read_split(args.data_dir, args.tag, split)
        if args.smoke:
            count = min(SMOKE_COUNTS[split], len(labels))
            tokens, labels = tokens[:count].copy(), labels[:count].copy()
            if len(np.unique(labels)) != 2:
                raise ValueError("smoke prefix must contain both labels")
        datasets[split], provenance[split] = (tokens, labels), prov
    manifest = manifest_for(args, provenance, datasets)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("immutable manifest mismatch; use a new output directory")
        if not args.resume and not args.dry_run:
            raise FileExistsError("existing run requires --resume")
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires an existing manifest")
        write_json(manifest_path, manifest)
    configure_determinism(args.seed, args.threads)
    model = RNNTransformerClassifier(**model_config())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.dry_run:
        result = dict(status="dry_run_passed", manifest_sha256=manifest["manifest_sha256"],
                      trainable_parameters=trainable, config=manifest["config"], data=provenance,
                      encoding_range=[int(min(x[0].min() for x in datasets.values())),
                                      int(max(x[0].max() for x in datasets.values()))],
                      zero_padding_tokens=sum(int(np.count_nonzero(x[0] == 0)) for x in datasets.values()))
        write_json(output / "dry_run.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        return result
    if (output / "metrics.json").exists():
        metrics = json.loads((output / "metrics.json").read_text())
        recomputed = verify_saved_predictions(output, manifest)
        if metrics["test"] != recomputed or metrics["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("completed metrics fail prediction audit")
        with np.load(output / "test_predictions.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["labels"], datasets["test"][1]):
                raise ValueError("saved test labels differ from canonical data")
        write_json(output / "status.json", dict(status="completed", phase="complete", tag=args.tag,
                   seed=args.seed, is_smoke=args.smoke, epoch=metrics["epochs_completed"],
                   best_epoch=metrics["best_epoch"], test_auc=metrics["test_auc"],
                   test_f1=metrics["test_f1"], manifest_sha256=manifest["manifest_sha256"],
                   updated_at_unix=time.time(), completed_resume_verified=True))
        return metrics
    device = configure_device(args.device, args.gpu_memory_mib, args.seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMAL_LR["RNNTransformer"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    history, best_state = [], None
    stopper = EarlyStopping(args.patience)
    previous_seconds, global_steps = 0., 0
    if (output / "resume.pt").exists():
        if not args.resume:
            raise FileExistsError("checkpoint exists; --resume is required")
        checkpoint = torch.load(output / "resume.pt", map_location="cpu", weights_only=False)
        if checkpoint["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("checkpoint manifest mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_state = checkpoint["history"], checkpoint["best_model"]
        stopper = EarlyStopping(args.patience, **checkpoint["selection"])
        previous_seconds, global_steps = checkpoint["elapsed_seconds"], checkpoint["global_steps"]
        restore_rng(checkpoint["rng"], generator, device)
    # Checkpoint owns history; discard any record written after its commit.
    (output / "history.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in history))
    started = time.monotonic()
    status_base = dict(tag=args.tag, seed=args.seed, model="RNNTransformer", is_smoke=args.smoke,
                       manifest_sha256=manifest["manifest_sha256"], pid=os.getpid(), n_train=len(datasets["train"][1]))

    def status(**changes):
        write_json(output / "status.json", dict(status_base, updated_at_unix=time.time(),
                                                elapsed_seconds=previous_seconds + time.monotonic() - started, **changes))

    tokens, labels = datasets["train"]
    token_tensor, label_tensor = torch.from_numpy(tokens), torch.from_numpy(labels)
    n_batches = math.ceil(len(labels) / args.batch_size)
    for epoch in range(len(history) + 1, args.epochs + 1):
        if stopper.stopped:
            break
        epoch_start = time.monotonic()
        model.train()
        order = torch.randperm(len(labels), generator=generator)
        loss_sum, seen = 0., 0
        status(status="running", phase="training", epoch=epoch, batch=0, batches=n_batches,
               best_epoch=stopper.best_epoch, best_val_f1=stopper.best_f1, bad_epochs=stopper.bad_epochs)
        last_status = time.monotonic()
        for batch, start in enumerate(range(0, len(labels), args.batch_size), 1):
            indices = order[start:start + args.batch_size]
            xb = token_tensor[indices].to(device=device, dtype=torch.long)
            yb = label_tensor[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
            global_steps += 1
            if time.monotonic() - last_status >= 15 or batch == n_batches:
                status(status="running", phase="training", epoch=epoch, batch=batch, batches=n_batches,
                       train_loss=loss_sum / seen, examples_per_second=seen / (time.monotonic() - epoch_start),
                       best_epoch=stopper.best_epoch, best_val_f1=stopper.best_f1, bad_epochs=stopper.bad_epochs)
                last_status = time.monotonic()
        status(status="running", phase="validation", epoch=epoch, batch=n_batches, batches=n_batches,
               best_epoch=stopper.best_epoch, best_val_f1=stopper.best_f1, bad_epochs=stopper.bad_epochs)
        validation, _ = evaluate(model, *datasets["val"], args.eval_batch_size, device)
        improved = stopper.update(validation["f1"], epoch)
        if improved:
            best_state = cpu_state(model)
        row = dict(epoch=epoch, train_loss=loss_sum / seen, validation=validation,
                   improved=improved, best_epoch=stopper.best_epoch, bad_epochs=stopper.bad_epochs,
                   global_steps=global_steps, epoch_seconds=time.monotonic() - epoch_start)
        history.append(row)
        checkpoint = dict(manifest_sha256=manifest["manifest_sha256"], model=cpu_state(model),
                          optimizer=optimizer.state_dict(), best_model=best_state, history=history,
                          selection=dict(best_f1=stopper.best_f1, best_epoch=stopper.best_epoch, bad_epochs=stopper.bad_epochs),
                          rng=rng_state(generator, device), global_steps=global_steps,
                          elapsed_seconds=previous_seconds + time.monotonic() - started)
        save_torch(output / "resume.pt", checkpoint)
        if improved:
            save_torch(output / "best_model.pt", dict(model=best_state, epoch=stopper.best_epoch,
                                                     manifest_sha256=manifest["manifest_sha256"], architecture=model_config()))
        with (output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(dict(tag=args.tag, seed=args.seed, **row), sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("no selected checkpoint")
    model.load_state_dict(best_state)
    save_torch(output / "best_model.pt", dict(model=best_state, epoch=stopper.best_epoch,
                                             manifest_sha256=manifest["manifest_sha256"], architecture=model_config()))
    status(status="running", phase="test", epoch=len(history), best_epoch=stopper.best_epoch,
           best_val_f1=stopper.best_f1, bad_epochs=stopper.bad_epochs)
    predictions_path = output / "test_predictions.npz"
    if not predictions_path.exists():
        test, logits = evaluate(model, *datasets["test"], args.eval_batch_size, device)
        metrics_check, probabilities, predictions = classification_metrics(datasets["test"][1], logits)
        if test != metrics_check:
            raise AssertionError("test metrics mismatch")
        with predictions_path.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, labels=datasets["test"][1].astype(np.uint8), logits=logits,
                                probabilities=probabilities, predictions=predictions.astype(np.uint8),
                                row_index=np.arange(len(logits)), manifest_sha256=manifest["manifest_sha256"])
            handle.flush()
            os.fsync(handle.fileno())
        predictions_path.with_suffix(".npz.tmp").replace(predictions_path)
    # Reopen persisted values and independently reconstruct metrics on CPU.
    test = verify_saved_predictions(output, manifest)
    with np.load(predictions_path, allow_pickle=False) as saved:
        if not np.array_equal(saved["labels"], datasets["test"][1]):
            raise ValueError("saved test labels differ from canonical data")
    result = dict(status="completed", model="RNNTransformer", tag=args.tag, seed=args.seed,
                  is_smoke=args.smoke, n_train=len(labels), n_val=len(datasets["val"][1]), n_test=test["n"],
                  rows=dict(train=len(labels), val=len(datasets["val"][1]), test=test["n"]),
                  train_fraction=1.0, tuning="from_scratch_all_parameters", mode="ordered",
                  test_informed_stopping=False, selected_by="validation_f1", threshold_protocol="argmax",
                  test=test, **{"test_" + name: test[name] for name in METRIC_NAMES},
                  best_epoch=stopper.best_epoch, epochs_completed=len(history), global_steps=global_steps,
                  best_validation=history[stopper.best_epoch - 1]["validation"],
                  stop_reason="validation_patience" if stopper.stopped else "maximum_epochs",
                  elapsed_seconds=previous_seconds + time.monotonic() - started,
                  trainable_parameters=trainable, manifest_sha256=manifest["manifest_sha256"],
                  prediction_sha256=sha256(predictions_path), metrics_verified_from_saved_predictions=True,
                  peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                  peak_gpu_reserved_bytes=int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0)
    write_json(output / "metrics.json", result)
    status(status="completed", phase="complete", epoch=len(history), best_epoch=stopper.best_epoch,
           best_val_f1=stopper.best_f1, test_auc=test["auc"], test_f1=test["f1"])
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, choices=tuple(KEYS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/simulation/kip_ordered_recovered")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-memory-mib", type=int, default=2048)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke else 30)
    for name in ("epochs", "patience", "batch_size", "eval_batch_size", "threads", "gpu_memory_mib"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.output_dir, args.data_dir = args.output_dir.resolve(), args.data_dir.resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        train(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / "failure.json", dict(status="failed", type=type(exc).__name__,
                                                         error=str(exc), updated_at_unix=time.time()))
        raise


if __name__ == "__main__":
    main()
