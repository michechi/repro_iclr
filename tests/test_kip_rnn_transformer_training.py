from __future__ import annotations

import csv
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.experiments import kip_rnn_transformer_training as runner


def test_cuda_allocator_receives_an_explicit_device_index(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(total_memory=80 * 1024**3))
    def allocator(fraction, device):
        # Reproduce the installed torch allocator's stricter device requirement.
        assert device.type == "cuda" and device.index == 0
        calls.append((fraction, device))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", allocator)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    device = runner.configure_device("cuda", 2048, 9550)
    assert device == torch.device("cuda:0")
    assert calls == [(2 / 80, device), 9550]
    with pytest.raises(ValueError, match="below GPU total"):
        runner.configure_device("cuda", 80 * 1024, 9550)


def test_cpu_device_configuration_does_not_initialize_cuda(monkeypatch):
    def forbidden():
        raise AssertionError("CPU path must not access CUDA")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    assert runner.configure_device("cpu", 2048, 9550) == torch.device("cpu")
    assert runner.SMOKE_COUNTS["val"] >= 512
    assert runner.SMOKE_COUNTS["test"] >= 512


def fixture_data(root, monkeypatch, tag="kip_m4"):
    counts = {"train": 32, "val": 16, "test": 12}
    monkeypatch.setattr(runner, "EXPECTED_COUNTS", counts)
    root.mkdir()
    keys = runner.KEYS[tag]
    distractors = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in keys]
    for split, count in counts.items():
        rows = []
        for index in range(count):
            letters = list(keys) + [distractors[index % len(distractors)]] * (20 - len(keys))
            if index % 2:
                letters[0], letters[1] = letters[1], letters[0]
            # Distinct rows without altering key order/parity.
            offset = index % (21 - len(keys))
            letters = letters[len(keys):len(keys) + offset] + letters[:len(keys)] + letters[len(keys) + offset:]
            sequence = "\x1f".join(letters)
            _, label = runner.encode_sequence(sequence, tag)
            rows.append((sequence, label))
        for kind, column, values in (("X", "Sequences", [r[0] for r in rows]),
                                      ("y", "Outcome", [r[1] for r in rows])):
            with (root / f"{kind}_{split}_{tag}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([column])
                writer.writerows([[v] for v in values])
    return root


def arguments(data, output, *extra):
    return runner.parse_args(["--tag", "kip_m4", "--seed", "9550", "--data-dir", str(data),
                              "--output-dir", str(output), "--device", "cpu", "--threads", "1",
                              "--smoke", "--epochs", "3", "--batch-size", "8", "--eval-batch-size", "8", *extra])


def test_letter_a_is_visible_and_trainable_padding_is_not():
    sequence = "\x1f".join(list("COTA") + ["B"] * 16)
    tokens, label = runner.encode_sequence(sequence, "kip_m4")
    assert label == 1 and tokens[:4] == [3, 15, 20, 1]
    assert all(token > 0 for token in tokens)
    legacy_tokens = [ord(c) - ord("A") for c in sequence.split("\x1f")]
    assert legacy_tokens[3] == 0  # Original path masked A as padding.
    runner.configure_determinism(5, 1)
    model = runner.RNNTransformerClassifier(**runner.model_config())
    assert sum(p.numel() for p in model.parameters()) == 82498
    captured = []
    hook = model.transformer.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs["src_key_padding_mask"].clone()), with_kwargs=True)
    x = torch.tensor([tokens], dtype=torch.long)
    torch.nn.functional.cross_entropy(model(x), torch.tensor([label])).backward()
    hook.remove()
    assert not captured[0].any()
    assert model.embedding.weight.grad[1].abs().sum() > 0
    assert model.embedding.weight.grad[0].abs().sum() == 0


@pytest.mark.parametrize("tag", ["kip_m4", "kip_m6"])
def test_split_validation_checks_labels_and_key_multiplicity(tmp_path, monkeypatch, tag):
    data = fixture_data(tmp_path / "data", monkeypatch, tag)
    tokens, labels, provenance = runner.read_split(data, tag, "train")
    assert tokens.shape == (32, 20) and tokens.dtype == np.uint8
    assert provenance["parity_verified_rows"] == 32
    assert np.array_equal(labels, np.tile([1, 0], 16))
    path = data / f"y_train_{tag}.csv"
    path.write_text(path.read_text().replace("1\n", "0\n", 1))
    with pytest.raises(ValueError, match="parity/label mismatch"):
        runner.read_split(data, tag, "train")
    with pytest.raises(ValueError, match="exactly once"):
        runner.encode_sequence("\x1f".join(["B"] * 20), tag)


def test_validation_f1_selection_counts_ties_as_no_improvement():
    stop = runner.EarlyStopping(2)
    assert stop.update(.6, 1)
    assert not stop.update(.6, 2)
    assert not stop.stopped
    assert stop.update(.7, 3)
    assert not stop.update(.69, 4)
    assert not stop.update(.7, 5)
    assert stop.stopped and stop.best_epoch == 3
    with pytest.raises(ValueError, match="finite"):
        stop.update(float("nan"), 6)


def test_metrics_preserve_argmax_ties_and_auc_ranking():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([[0., 0.], [1., 0.], [0., 2.], [2., 0.]], dtype=np.float32)
    metrics, probabilities, predictions = runner.classification_metrics(labels, logits)
    assert predictions.tolist() == [0, 0, 1, 0]
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert metrics["precision"] == 1 and metrics["recall"] == .5
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["auc"] == .5 and probabilities[0] == .5
    expected_loss = torch.nn.functional.cross_entropy(torch.tensor(logits, dtype=torch.float64), torch.tensor(labels))
    assert metrics["loss"] == pytest.approx(float(expected_loss), abs=1e-15)


def test_epoch_resume_matches_uninterrupted_and_test_runs_only_after_fit(tmp_path, monkeypatch):
    data = fixture_data(tmp_path / "data", monkeypatch)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    full_metrics = runner.train(arguments(data, full))
    original_save, original_eval = runner.save_torch, runner.evaluate
    evaluations = []

    def capture_eval(model, tokens, labels, batch_size, device):
        evaluations.append(len(labels))
        return original_eval(model, tokens, labels, batch_size, device)

    def interrupt_after_commit(path, checkpoint):
        original_save(path, checkpoint)
        if Path(path).name == "resume.pt" and len(checkpoint["history"]) == 1:
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(runner, "evaluate", capture_eval)
    monkeypatch.setattr(runner, "save_torch", interrupt_after_commit)
    with pytest.raises(RuntimeError, match="simulated"):
        runner.train(arguments(data, resumed))
    assert evaluations == [16]  # No test call at a training checkpoint.
    monkeypatch.setattr(runner, "save_torch", original_save)
    resumed_metrics = runner.train(arguments(data, resumed, "--resume"))
    assert evaluations == [16, 16, 16, 12]
    assert full_metrics["test"] == resumed_metrics["test"]
    assert full_metrics["best_epoch"] == resumed_metrics["best_epoch"]
    with np.load(full / "test_predictions.npz") as a, np.load(resumed / "test_predictions.npz") as b:
        assert np.array_equal(a["logits"], b["logits"])
    af = torch.load(full / "resume.pt", weights_only=False)
    ar = torch.load(resumed / "resume.pt", weights_only=False)
    assert all(torch.equal(af["model"][k], ar["model"][k]) for k in af["model"])
    assert len((resumed / "history.jsonl").read_text().splitlines()) == 3
    # Completed restart verifies saved predictions without any evaluation.
    again = runner.train(arguments(data, resumed, "--resume"))
    assert again["test"] == full_metrics["test"] and len(evaluations) == 4
    with pytest.raises(ValueError, match="immutable manifest"):
        runner.train(arguments(data, resumed, "--resume", "--patience", "7"))


def test_dry_run_is_cpu_only_and_refuses_changed_inputs(tmp_path, monkeypatch):
    data = fixture_data(tmp_path / "data", monkeypatch)
    output = tmp_path / "dry"
    def no_cuda():
        raise AssertionError("dry run should never inspect CUDA availability")
    monkeypatch.setattr(torch.cuda, "is_available", no_cuda)
    dry = runner.train(arguments(data, output, "--dry-run"))
    assert dry["status"] == "dry_run_passed" and dry["zero_padding_tokens"] == 0
    assert dry["trainable_parameters"] == 82498
    assert not (output / "test_predictions.npz").exists()
    path = data / "X_train_kip_m4.csv"
    rows = path.read_text().splitlines()
    rows[1] = rows[1].replace("B", "D")  # Valid parity, changed input content.
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="immutable manifest"):
        runner.train(arguments(data, output, "--dry-run"))


def test_saved_prediction_verifier_rejects_inconsistent_probabilities(tmp_path):
    labels = np.array([0, 1])
    logits = np.array([[1., 0.], [0., 1.]])
    _, probs, pred = runner.classification_metrics(labels, logits)
    np.savez_compressed(tmp_path / "test_predictions.npz", labels=labels, logits=logits,
                        probabilities=probs + .01, predictions=pred, row_index=np.arange(2), manifest_sha256="abc")
    with pytest.raises(ValueError, match="do not match"):
        runner.verify_saved_predictions(tmp_path, {"config": {"n_test": 2}, "manifest_sha256": "abc"})


def test_production_defaults_are_fixed_budget_and_smoke_is_explicit():
    args = runner.parse_args(["--tag", "kip_m6", "--seed", "9550", "--output-dir", "/tmp/unused"])
    assert (args.epochs, args.patience, args.batch_size, args.eval_batch_size) == (30, 5, 64, 512)
    assert args.gpu_memory_mib == 2048 and not args.smoke
