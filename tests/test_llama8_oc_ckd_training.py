"""CPU checks for fixed data identities, full-context LoRA selection and audits."""
from contextlib import redirect_stdout
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

from src.experiments import llama8_oc_ckd as runner
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
        self.embedding = nn.Embedding(512, 4)
        self.embedding.requires_grad_(False)
        self.score = nn.Linear(4, 2, bias=False)

    def forward(self, input_ids, attention_mask):
        last = attention_mask.sum(-1) - 1
        hidden = self.embedding(input_ids)[torch.arange(len(last)), last]
        return SimpleNamespace(logits=self.score(hidden))


def make_plan(root, task="ckd", smoke=False, condition="ordered", sizes=None):
    sizes = sizes or {"train": 10, "val": 8, "test": 6}
    source = root / "snapshot"
    source.mkdir(exist_ok=True)
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-1.safetensors"}}))
    (source / "model-1.safetensors").write_bytes(b"tiny pretrained fixture")
    (source / "config.json").write_text('{"model_type":"llama"}')
    datasets = {}
    for data_condition in ("ordered", "shuffled"):
        datasets[data_condition] = {}
        for split, count in sizes.items():
            labels = np.arange(count) % 2
            if task == "ckd":
                sequences = [runner.SEP.join(["CCS_39", "OTHER"] + ["CCS_100"] * i) for i in range(count)]
            else:
                sequences = [runner.SEP.join("ABCDEFGHIJKLMNOPQRST") for _ in range(count)]
            if data_condition == "shuffled":
                sequences = [runner.SEP.join(reversed(sequence.split(runner.SEP))) for sequence in sequences]
            x, y = root / f"X_{split}_{data_condition}.csv", root / f"y_{split}_{data_condition}.csv"
            pd.DataFrame({"Sequences": sequences}).to_csv(x, index=False)
            pd.DataFrame({"Outcome": labels}).to_csv(y, index=False)
            datasets[data_condition][split] = dict(features=str(x), labels=str(y), rows=count,
                features_sha256=runner.sha256(x), labels_sha256=runner.sha256(y))
    recipe = dict(seed=9950 if task == "oc" else 42, epochs=2, min_epochs=1, patience=3,
                  batch_size=3, eval_batch_size=3, effective_batch_size=6, lr=2e-5,
                  max_length=64 if task == "oc" else 16384)
    job = dict(key="fit", kind="fit", task=task, condition=condition, model=runner.ARM,
               smoke=smoke, output_dir=str(root / "fit"))
    plan = dict(plan_sha256="test-plan", source_sha256={}, jobs=[job],
                models={runner.ARM: dict(path=str(source), name=runner.MODEL_NAME,
                    files_sha256={"config.json": runner.sha256(source / "config.json")})},
                datasets={task: datasets}, recipes={task: recipe})
    (root / "launch_plan.json").write_text(json.dumps(plan))
    return plan


def run_tiny(root, *extra, evaluation=None):
    args = runner.parse_args(["--run-root", str(root), "--job", "fit", "--device", "cpu", *extra])
    model = TinyModel()
    metadata = {"trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad]}
    with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
            mock.patch.object(factory, "load_classifier", return_value=(model, TinyTokenizer(), metadata)), \
            mock.patch.object(runner, "evaluate", side_effect=evaluation or runner.evaluate), \
            redirect_stdout(io.StringIO()):
        result = runner.train(args)
    return result, model


class DataTests(unittest.TestCase):
    def test_prompts_are_exact_and_refuse_malformed_events(self):
        self.assertEqual(runner.prompt("CCS_39" + runner.SEP + "OTHER", "ckd"),
                         "Sequential events: CCS_39 OTHER\nOutcome (0 or 1):")
        self.assertEqual(runner.prompt(runner.SEP.join("ABCDEFGHIJKLMNOPQRST"), "oc"),
                         "Sequential events: A B C D E F G H I J K L M N O P Q R S T")
        for sequence, task in [("", "ckd"), ("Outcome=1", "ckd"), ("CCS_39" + runner.SEP, "ckd"),
                               ("A", "oc"), ("OTHER", "unknown")]:
            with self.subTest(sequence=sequence, task=task), self.assertRaises(ValueError):
                runner.prompt(sequence, task)

    def test_full_context_never_silently_truncates(self):
        tokenizer = TinyTokenizer()
        strings = [runner.SEP.join(["CCS_39"] * count) for count in (5, 80)]
        dataset = runner.TextDataset(strings, [0, 1], tokenizer, "ckd", 100)
        self.assertGreater(dataset.audit["max_full_tokens"], 64)
        self.assertEqual(dataset.audit["truncated_rows"], 0)
        self.assertTrue(all(options["truncation"] is False for options in tokenizer.options))
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            runner.TextDataset(strings, [0, 1], tokenizer, "ckd", 64)

    def test_reader_detects_changed_input_even_with_same_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            path = root / "y_train_ordered.csv"
            data = pd.read_csv(path)
            data["Outcome"] = 1 - data["Outcome"]
            data.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "frozen input changed"):
                runner.read_data(plan, "ckd", "ordered", "train")

    def test_smoke_includes_each_conditions_longest_and_preserves_counts(self):
        labels, lengths = np.arange(100) % 2, np.arange(100) + 1
        selected = runner.smoke_indices(labels, lengths, 16, 42, protected_indices=(92, 95))
        baseline = runner.subset_indices(labels, 16, 42)
        self.assertTrue({92, 95, 99}.issubset(selected))
        self.assertEqual(len(selected), 16)
        self.assertEqual(labels[selected].sum(), labels[baseline].sum())
        np.testing.assert_array_equal(selected, np.unique(selected))

    def test_production_oc_subset_exactly_matches_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.npy"
            labels = np.arange(100) % 2
            recipe = dict(seed=9950, train_rows=40)
            expected = runner.subset_indices(labels, 40, 9950)
            np.save(path, expected, allow_pickle=False)
            recipe["reference_train_indices"] = dict(path=str(path), sha256=runner.sha256(path))
            actual = runner.split_indices(labels, "train", recipe)
            runner.verify_reference_indices(actual, recipe)
            with self.assertRaisesRegex(ValueError, "differs from the fixed encoder OC sample"):
                runner.verify_reference_indices(actual[::-1], recipe)
            np.save(path, actual[::-1], allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "reference training indices changed"):
                runner.verify_reference_indices(actual, recipe)
            np.testing.assert_array_equal(runner.split_indices(labels, "test", recipe), np.arange(100))

    def test_batch_override_preserves_effective_batch_and_rejects_bad_recipe(self):
        job = dict(task="ckd", smoke=False)
        plan = dict(recipes={"ckd": dict(seed=42, epochs=30, min_epochs=5, patience=5, batch_size=4,
            eval_batch_size=4, effective_batch_size=32, lr=2e-5)})
        args = SimpleNamespace(batch_size=2, eval_batch_size=None)
        actual = runner.actual_recipe(plan, job, args)
        self.assertEqual(actual["gradient_accumulation_steps"], 16)
        self.assertEqual(actual["eval_batch_size"], 2)
        args.batch_size = 3
        with self.assertRaisesRegex(ValueError, "divide"):
            runner.actual_recipe(plan, job, args)

    def test_dry_run_never_loads_weights_or_queries_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_plan(root)
            args = runner.parse_args(["--run-root", str(root), "--job", "fit", "--dry-run"])
            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=TinyTokenizer()), \
                    mock.patch.object(factory, "load_classifier", side_effect=AssertionError("weights loaded")), \
                    mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")), \
                    redirect_stdout(io.StringIO()):
                audit = runner.train(args)
            self.assertFalse(audit["weights_loaded"])
            self.assertFalse(audit["gpu_used"])
            self.assertEqual(set(audit["tokenization"]), {"train", "val", "test"})
            self.assertFalse((root / "fit/best_trainable.pt").exists())

    def test_ckd_smokes_use_paired_rows_and_include_longest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sizes = dict(train=100, val=80, test=70)
            for condition in ("ordered", "shuffled"):
                make_plan(root, smoke=True, condition=condition, sizes=sizes)
                run_tiny(root, "--dry-run", "--output-dir", str(root / condition))
            for split, count in sizes.items():
                ordered = np.load(root / "ordered" / f"{split}_indices.npy")
                shuffled = np.load(root / "shuffled" / f"{split}_indices.npy")
                np.testing.assert_array_equal(ordered, shuffled)
                self.assertIn(count - 1, ordered)
                self.assertEqual(len(ordered), 64 if split == "train" else 32)

    def test_pairing_check_rejects_swapped_labels_in_smoke(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root, smoke=True)
            path = root / "y_train_shuffled.csv"
            data = pd.read_csv(path)
            data["Outcome"] = 1 - data["Outcome"]
            data.to_csv(path, index=False)
            plan["datasets"]["ckd"]["shuffled"]["train"]["labels_sha256"] = runner.sha256(path)
            (root / "launch_plan.json").write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "not paired"):
                run_tiny(root, "--dry-run")


class TrainingTests(unittest.TestCase):
    def test_partial_accumulation_equals_sample_weighted_batch_gradient(self):
        values = torch.arange(1, 11, dtype=torch.float32)
        weight = torch.tensor(.5, requires_grad=True)
        observed, counts = [], []
        for index, batch in enumerate(values.split(3)):
            stop, count = runner.accumulation_samples(index, 4, 3, 10, 2)
            ((weight * batch).square().mean() * len(batch) / count).backward()
            if index + 1 == stop:
                observed.append(weight.grad.item())
                counts.append(count)
                weight.grad = None
        self.assertEqual(counts, [6, 4])
        expected = [batch.square().mean().item() for batch in (values[:6], values[6:])]
        np.testing.assert_allclose(observed, expected, rtol=1e-6)

    def test_checkpoint_retains_only_trainable_weights_and_survives_disk_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = TinyModel()
            runner.save_trainable_checkpoint(model, root, 3)
            payload = torch.load(root / "best_trainable.pt", weights_only=True)
            self.assertEqual(payload["arm"], runner.ARM)
            self.assertEqual(set(payload["state_dict"]), {"score.weight"})
            with mock.patch.object(runner.shutil, "disk_usage", return_value=SimpleNamespace(free=1)), \
                    self.assertRaises(OSError):
                runner.save_trainable_checkpoint(model, root, 4)
            self.assertEqual(torch.load(root / "best_trainable.pt", weights_only=True)["epoch"], 3)

    def test_full_and_smoke_end_to_end_restore_selected_weights_before_test(self):
        real_evaluate = runner.evaluate
        for task in ("oc", "ckd"):
            for smoke in (False, True):
                with self.subTest(task=task, smoke=smoke), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    make_plan(root, task=task, smoke=smoke)
                    evaluations = []

                    def recorded_evaluation(model, loader, device, precision="bf16"):
                        evaluations.append(len(loader.dataset))
                        if len(loader.dataset) == 6:
                            saved = torch.load(root / "fit/best_trainable.pt", weights_only=True)
                            self.assertTrue(torch.equal(model.score.weight, saved["state_dict"]["score.weight"]))
                        return real_evaluate(model, loader, device, precision)

                    result, model = run_tiny(root, evaluation=recorded_evaluation)
                    self.assertEqual(result["status"], "completed")
                    self.assertFalse(result["test_informed_stopping"])
                    self.assertEqual(result["optimizer_steps"], 2 if smoke else 4)
                    self.assertEqual(evaluations.count(6), 1)
                    self.assertEqual(evaluations[-1], 6)
                    self.assertEqual(result["selected_checkpoint_sha256"], runner.sha256(root / "fit/best_trainable.pt"))
                    manifest = json.loads((root / "fit/manifest.json").read_text())
                    saved = torch.load(root / "fit/best_trainable.pt", weights_only=True)
                    self.assertEqual(set(saved["state_dict"]), set(manifest["model"]["trainable_parameter_names"]))
                    self.assertEqual(manifest["config"]["arm"], runner.ARM)
                    self.assertEqual(manifest["recipe"]["gradient_accumulation_steps"], 2)
                    self.assertEqual(manifest["index_sha256"]["test"], runner.sha256(root / "fit/test_indices.npy"))
                    self.assertEqual(len(pd.read_csv(root / "fit/test_predictions.csv")), 6)
                    self.assertFalse((root / "fit/tokenizer").exists())

    def test_minimum_epochs_and_validation_patience_never_consult_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = make_plan(root)
            plan["recipes"]["ckd"].update(epochs=8, min_epochs=5, patience=2)
            (root / "launch_plan.json").write_text(json.dumps(plan))
            seen = []

            def evaluation(model, loader, device):
                size = len(loader.dataset)
                seen.append(size)
                labels = loader.dataset.labels
                probability = np.where(labels, .8, .2)
                return (0.3 if len(seen) == 1 else .4), probability, labels

            result, _ = run_tiny(root, evaluation=evaluation)
            self.assertEqual(result["epochs_completed"], 5)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["stop_reason"], "validation_patience")
            self.assertEqual(seen, [8, 8, 8, 8, 8, 8, 6])

    def test_refuses_overwriting_existing_artifact_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_plan(root)
            run_tiny(root, "--dry-run")
            before = (root / "fit/manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                run_tiny(root, "--dry-run")
            self.assertEqual((root / "fit/manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
