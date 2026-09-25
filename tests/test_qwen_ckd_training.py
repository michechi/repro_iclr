"""CPU checks for CKD split pairing, held-out isolation and QLoRA fit bookkeeping."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.mimic import qwen_ckd_training as runner
from src.experiments import kip_efficiency_models as factory


class TinyTokenizer:
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token_id = 0

    def __init__(self):
        self.options = []

    def __call__(self, texts, **kwargs):
        self.options.append(kwargs)
        return {"input_ids": [list(range(1, len(text.split()) + 1)) for text in texts]}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 4)
        self.embedding.requires_grad_(False)
        self.score = nn.Linear(4, 2, bias=False)

    def forward(self, input_ids, attention_mask):
        last = attention_mask.sum(-1) - 1
        hidden = self.embedding(input_ids)[torch.arange(len(last)), last]
        return SimpleNamespace(logits=self.score(hidden))


def write_data(directory, splits=("train", "val", "test")):
    directory.mkdir(parents=True, exist_ok=True)
    sizes = {"train": 10, "val": 8, "test": 6}
    for split in splits:
        count = sizes[split]
        labels = np.arange(count) % 2
        sequences = [runner.SEP.join(["CCS_39", "OTHER"] + ["CCS_100"] * i) for i in range(count)]
        for condition in ("ordered", "shuffled"):
            strings = sequences if condition == "ordered" else [runner.SEP.join(reversed(s.split(runner.SEP))) for s in sequences]
            pd.DataFrame({"Sequences": strings}).to_csv(directory / f"X_{split}_{condition}.csv", index=False)
            pd.DataFrame({"Outcome": labels}).to_csv(directory / f"y_{split}_{condition}.csv", index=False)


def fake_snapshot(directory):
    directory.mkdir()
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-1.safetensors"}}))
    (directory / "model-1.safetensors").write_bytes(b"small test weights")
    (directory / "config.json").write_text('{"model_type":"test"}')
    return directory


class ClinicalDataTests(unittest.TestCase):
    def test_pairing_rejects_labels_even_if_prevalence_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_data(directory)
            path = directory / "y_train_shuffled.csv"
            labels = pd.read_csv(path)
            labels["Outcome"] = 1 - labels["Outcome"]
            labels.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "labels or row counts differ"):
                runner.read_paired_data(directory)

    def test_pairing_rejects_event_multiplicity_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_data(directory)
            path = directory / "X_val_shuffled.csv"
            data = pd.read_csv(path)
            data.loc[0, "Sequences"] += runner.SEP + "CCS_39"
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "event multisets differ"):
                runner.read_paired_data(directory)

    def test_smoke_does_not_read_or_require_test_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_data(directory, splits=("train", "val"))
            data, provenance = runner.read_paired_data(directory, smoke=True)
            self.assertEqual(set(data), {"train", "val"})
            self.assertEqual(set(provenance), {"train", "val"})

    def test_tokenization_forbids_truncation_and_preserves_prompt(self):
        tokenizer = TinyTokenizer()
        sequences = ["CCS_39" + runner.SEP + "OTHER", "CCS_100"]
        dataset = runner.ClinicalDataset(sequences, np.array([0, 1]), tokenizer, 64)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(runner.sequence_prompt(sequences[0]), "Sequential events: CCS_39 OTHER\nOutcome (0 or 1):")
        self.assertTrue(all(options["truncation"] is False for options in tokenizer.options))
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            runner.ClinicalDataset(sequences, np.array([0, 1]), tokenizer, 2)
        for sequence in ("", "CCS_39 other", "CCS_39" + runner.SEP, "Outcome=1"):
            with self.assertRaises(ValueError):
                runner.sequence_prompt(sequence)

    def test_smoke_includes_longest_without_changing_stratified_counts(self):
        sequences = [runner.SEP.join(["CCS_100"] * (i + 1)) for i in range(100)]
        dataset = runner.ClinicalDataset(sequences, np.arange(100) % 2, TinyTokenizer(), 128)
        expected = runner.subset_indices(dataset.labels, 16, 42)
        actual = runner.smoke_indices(dataset, 16, 42, longest_count=2)
        self.assertTrue({98, 99}.issubset(set(actual)))
        self.assertEqual(len(set(actual)), 16)
        self.assertEqual(dataset.labels[actual].sum(), dataset.labels[expected].sum())
        np.testing.assert_array_equal(actual, np.sort(actual))

    def test_dry_run_audits_every_condition_without_cuda_or_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_data(root / "data")
            args = runner.parse_args(["--model-path", str(root / "no-weights"), "--data-dir", str(root / "data"),
                                      "--output-dir", str(root / "audit"), "--dry-run"])
            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                    mock.patch.object(factory, "load_classifier", side_effect=AssertionError("weights loaded")), \
                    mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
                    redirect_stdout(io.StringIO()):
                audit = runner.run(args)
            self.assertFalse(audit["gpu_used"])
            self.assertFalse(audit["weights_loaded"])
            self.assertEqual(len(audit["token_lengths"]), 6)
            self.assertEqual(json.loads((root / "audit/status.json").read_text())["status"], "completed")


class OptimizationTests(unittest.TestCase):
    def test_final_partial_accumulation_matches_true_group_mean_gradient(self):
        inputs = torch.arange(1, 11, dtype=torch.float32)
        weight = torch.tensor(0.5, requires_grad=True)
        sizes, observed = [], []
        for index, values in enumerate(inputs.split(3)):
            stop, samples = runner.accumulation_samples(index, 4, 3, 10, 2)
            ((weight * values).square().mean() * len(values) / samples).backward()
            if index + 1 == stop:
                observed.append(weight.grad.item())
                sizes.append(samples)
                weight.grad = None
        self.assertEqual(sizes, [6, 4])
        expected = [(2 * 0.5 * group.square()).mean().item() for group in (inputs[:6], inputs[6:])]
        np.testing.assert_allclose(observed, expected, rtol=1e-6)

    def test_checkpoint_contains_trainable_head_only_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model = TinyModel()
            for epoch in (1, 3):
                runner.save_trainable_checkpoint(model, output, epoch)
            payload = torch.load(output / "best_trainable.pt", weights_only=True)
            self.assertEqual(set(payload["state_dict"]), {"score.weight"})
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual([path.name for path in output.iterdir()], ["best_trainable.pt"])
            with mock.patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
                    self.assertRaises(OSError):
                runner.save_trainable_checkpoint(model, output, 4)
            self.assertEqual(torch.load(output / "best_trainable.pt", weights_only=True)["epoch"], 3)

    def test_end_to_end_full_and_smoke_test_isolation(self):
        real_zeros, real_loader = torch.zeros, runner.DataLoader
        real_loss, real_evaluate = runner.classifier_loss, runner.evaluate
        for smoke in (False, True):
            with self.subTest(smoke=smoke), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_data(root / "data", splits=("train", "val") if smoke else ("train", "val", "test"))
                source = fake_snapshot(root / "snapshot")
                arguments = ["--model-path", str(source), "--data-dir", str(root / "data"), "--output-dir", str(root / "fit"),
                             "--batch-size", "3", "--gradient-accumulation-steps", "2", "--eval-batch-size", "3",
                             "--epochs", "2", "--min-epochs", "1"]
                args = runner.parse_args(arguments + (["--smoke"] if smoke else []))
                evaluated = []

                def cpu_loss(model, batch, device, precision="bf16"):
                    return real_loss(model, batch, torch.device("cpu"), "fp32")

                def cpu_evaluate(model, loader, device):
                    evaluated.append(len(loader.dataset))
                    return real_evaluate(model, loader, torch.device("cpu"), "fp32")

                def cpu_zeros(*arguments, **kwargs):
                    kwargs["device"] = "cpu"
                    return real_zeros(*arguments, **kwargs)

                def cpu_loader(*arguments, **kwargs):
                    kwargs["pin_memory"] = False
                    return real_loader(*arguments, **kwargs)

                with ExitStack() as stack:
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    stack.enter_context(mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()))
                    stack.enter_context(mock.patch.object(factory, "load_classifier", return_value=(TinyModel(), TinyTokenizer(), {})))
                    stack.enter_context(mock.patch.object(runner, "seed_all"))
                    stack.enter_context(mock.patch.object(runner, "classifier_loss", side_effect=cpu_loss))
                    stack.enter_context(mock.patch.object(runner, "evaluate", side_effect=cpu_evaluate))
                    stack.enter_context(mock.patch.object(runner, "DataLoader", side_effect=cpu_loader))
                    stack.enter_context(mock.patch.object(torch, "zeros", side_effect=cpu_zeros))
                    stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
                    stack.enter_context(mock.patch.object(torch.optim.Optimizer, "_accelerator_graph_capture_health_check"))
                    for name in ("reset_peak_memory_stats", "empty_cache"):
                        stack.enter_context(mock.patch.object(torch.cuda, name))
                    stack.enter_context(mock.patch.object(torch.cuda, "max_memory_allocated", return_value=0))
                    metrics = runner.run(args)
                self.assertEqual(metrics["status"], "completed")
                self.assertFalse(metrics["test_informed_stopping"])
                self.assertEqual(metrics["optimizer_steps"], (1 if smoke else 2) * 2)
                self.assertEqual(evaluated[-1], 8 if smoke else 6)
                self.assertEqual(evaluated.count(6), 0 if smoke else 1)
                self.assertEqual("test" in metrics, not smoke)
                self.assertEqual((root / "fit/test_predictions.csv").exists(), not smoke)
                self.assertEqual((root / "fit/test_indices.npy").exists(), not smoke)
                self.assertEqual(metrics["validation"]["auprc"], metrics["validation"]["average_precision"])
                manifest = json.loads((root / "fit/manifest.json").read_text())
                self.assertEqual(manifest["config"]["effective_batch_size"], 6)
                self.assertEqual(manifest["token_lengths"]["train"]["max"], manifest["source_token_lengths"]["train_ordered"]["max"])


if __name__ == "__main__":
    unittest.main()
