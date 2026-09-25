"""Real tiny-Llama checks for native OC and full-context CKD attention."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.analysis.kip_checkpoint_attention import capture_native_attention, digest, infer, write_json
from src.analysis.llama8_oc_ckd_attention import (
    audit, capture_readout_attention, encode_events, paired_summary, sampled_rows, validate_fit,
)
from src.analysis.oc_attention_core import label, oc_statistics
from src.experiments.kip_efficiency_models import LastTokenClassifier


def tiny_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["[PAD]", "[UNK]", "Sequential", "events", ":", "Outcome", "(", "0", "or", "1", "):"]
    words += list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["CCS_1", "CCS_2", "CCS_17", "CCS_123"]
    native = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    native.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=native, unk_token="[UNK]", pad_token="[PAD]",
                                  model_max_length=2048, padding_side="right")


def tiny_classifier():
    from transformers import LlamaConfig, LlamaModel

    torch.manual_seed(452)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                         pad_token_id=0, attn_implementation="sdpa", attention_dropout=0.)
    return LastTokenClassifier(LlamaModel(config)).eval()


class ReadoutCaptureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_last_unmasked_query_matches_full_attention_with_gqa_and_right_padding(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        model = tiny_classifier()
        inputs = dict(input_ids=torch.tensor([[2, 4, 5, 3, 0, 0, 0, 0], [2, 6, 7, 8, 3, 4, 0, 0]]),
                      attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]]))
        native = ALL_ATTENTION_FUNCTIONS["sdpa"]
        before, _ = infer(model, inputs, True)
        full, readout = {}, {}
        with capture_native_attention(full, {0: 4, 1: 6}):
            infer(model, inputs, True)
        with capture_readout_attention(readout, {0: 4, 1: 6}):
            after, _ = infer(model, inputs, True)
        np.testing.assert_array_equal(before, after)
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], native)
        self.assertEqual(sorted(readout), [0, 1])
        for layer in (0, 1):
            for row, length in [(0, 4), (1, 6)]:
                self.assertEqual(readout[layer][row].shape, (4, length))
                np.testing.assert_allclose(readout[layer][row], full[layer][row][:, length - 1], atol=.003)
                np.testing.assert_allclose(readout[layer][row].sum(-1), 1, atol=.01)

    def test_unpadded_implicit_causal_mask_and_linear_storage(self):
        model = tiny_classifier()
        inputs = dict(input_ids=torch.tensor([[2, 4, 5, 3, 6, 7, 8, 9]]), attention_mask=torch.ones((1, 8), dtype=torch.long))
        full, readout = {}, {}
        with capture_native_attention(full, {0: 8}):
            before, _ = infer(model, inputs, True)
        with capture_readout_attention(readout, {0: 8}):
            after, _ = infer(model, inputs, True)
        np.testing.assert_array_equal(before, after)
        for layer in full:
            self.assertEqual(readout[layer][0].size * 8, full[layer][0].size)
            np.testing.assert_allclose(readout[layer][0], full[layer][0][:, -1], atol=.003)

    def test_registry_restored_on_invalid_capture_and_user_exception(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        native = ALL_ATTENTION_FUNCTIONS["sdpa"]
        model = tiny_classifier()
        inputs = dict(input_ids=torch.tensor([[2, 4, 5, 3]]), attention_mask=torch.ones((1, 4), dtype=torch.long))
        for lengths in ({0: 0}, {0: 5}, {1: 4}, {0: 3.5}):
            with self.assertRaisesRegex(ValueError, "invalid selected"):
                with capture_readout_attention({}, lengths):
                    infer(model, inputs, True)
            self.assertIs(native, ALL_ATTENTION_FUNCTIONS["sdpa"])
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with capture_readout_attention({}, {0: 4}):
                raise RuntimeError("intentional")
        self.assertIs(native, ALL_ATTENTION_FUNCTIONS["sdpa"])
        with self.assertRaises(ValueError):
            with capture_readout_attention({}, {}):
                pass
        self.assertIs(native, ALL_ATTENTION_FUNCTIONS["sdpa"])

    def test_training_capture_rejected(self):
        model = tiny_classifier().train()
        inputs = dict(input_ids=torch.tensor([[2, 4, 5, 3]]), attention_mask=torch.ones((1, 4), dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "uncached causal"):
            with capture_readout_attention({}, {0: 4}):
                infer(model, inputs, True)

    def test_full_context_mapping_matches_training_and_rejects_truncation(self):
        from src.experiments.llama8_oc_ckd import TextDataset

        tokenizer = tiny_tokenizer()
        sequence = "\x1f".join(["CCS_123", "CCS_17"] * 300)
        dataset = TextDataset([sequence], np.array([1]), tokenizer, "ckd", 2048)
        encoded, events, groups = encode_events(tokenizer, sequence, "ckd", 2048)
        self.assertEqual(encoded["input_ids"], dataset.input_ids[0])
        self.assertEqual(len(events), 600)
        self.assertGreater(len(encoded["input_ids"]), 512)
        self.assertEqual(len(groups), 600)
        self.assertEqual(len({i for group in groups for i in group}), sum(map(len, groups)))
        with self.assertRaisesRegex(ValueError, "truncate"):
            encode_events(tokenizer, sequence, "ckd", 512)

    def test_sample_is_restricted_to_saved_smoke_rows_and_production_matches_common_seed(self):
        from src.analysis.kip_attention_core import fixed_sample

        labels = np.arange(120) % 2
        order = np.arange(40, 72)
        sampled = sampled_rows(labels, order, 16, 9950)
        self.assertEqual(len(sampled), 16)
        self.assertTrue(set(sampled) <= set(order))
        np.testing.assert_array_equal(sampled_rows(labels, np.arange(120), 60, 9950), fixed_sample(labels, 60, 9950))
        lengths = np.ones(120, dtype=int)
        lengths[71] = 1000
        protected = sampled_rows(labels, order, 16, 9950, lengths=lengths, protected_indices=[70])
        self.assertIn(71, protected)
        self.assertIn(70, protected)
        self.assertEqual(labels[protected].sum(), 8)
        for invalid in ([0, 0], [-1, 3], [2, 120], [2.5, 3.5]):
            with self.assertRaises(ValueError):
                sampled_rows(labels, np.asarray(invalid), 2, 9950)

    def test_oc_uniform_baseline_is_causal(self):
        events = ["B"] * 20
        events[0], events[7], events[14] = "W", "D", "Q"
        causal = np.tri(24, dtype=float)
        causal /= causal.sum(-1, keepdims=True)
        masses, baseline = oc_statistics(np.stack([causal] * 4), [[i + 3] for i in range(20)], events)
        valid = np.isfinite(baseline) & (baseline > 0)
        np.testing.assert_allclose(masses[valid] / baseline[valid, None], 1)


class EndToEndAttention(unittest.TestCase):
    def test_real_tiny_llama_oc_and_paired_full_history_ckd(self):
        from src.experiments.kip_encoder_training import DynamicPadding
        from src.experiments.llama8_oc_ckd import TextDataset

        torch.set_num_threads(2)
        tokenizer, model = tiny_tokenizer(), tiny_classifier()
        with tempfile.TemporaryDirectory(prefix="llama8_attention_test_") as temporary:
            root = Path(temporary)
            labels = np.arange(32) % 2
            datasets, jobs = {}, []
            sequences_by_condition = {}
            for task, condition in [("oc", "ordered"), ("ckd", "ordered"), ("ckd", "shuffled")]:
                datasets.setdefault(task, {})[condition] = {}
                sequences = []
                for i, target in enumerate(labels):
                    if task == "oc":
                        events = ["B"] * 20
                        events[0], events[7], events[14] = "W", "D", "Q" if target else "W"
                        events[18] = "A"
                        self.assertEqual(label(events), target)
                    else:
                        events = (["CCS_1", "CCS_123"] if target else ["CCS_2", "CCS_17"]) * (3 + i % 4)
                        if condition == "shuffled":
                            events = np.random.default_rng(42 + i).permutation(events).tolist()
                    sequences.append("\x1f".join(events))
                sequences_by_condition[task, condition] = sequences
                xp, yp = root / f"{task}_{condition}_x.csv", root / f"{task}_{condition}_y.csv"
                pd.DataFrame(dict(Sequences=sequences)).to_csv(xp, index=False)
                pd.DataFrame(dict(Outcome=labels)).to_csv(yp, index=False)
                source = dict(features=str(xp), labels=str(yp), features_sha256=digest(xp), labels_sha256=digest(yp), rows=len(labels))
                for split in ("train", "val", "test"):
                    datasets[task][condition][split] = source
                key = f"llama8_{task}_{condition}"
                run = root / "fits" / key
                run.mkdir(parents=True)
                indices = np.arange(len(labels))
                for split in ("train", "val", "test"):
                    np.save(run / f"{split}_indices.npy", indices)
                torch.save(dict(arm="llama8b_lora", epoch=1, state_dict=model.state_dict()), run / "best_trainable.pt")
                recipe = dict(eval_batch_size=4, max_length=2048 if task == "ckd" else 64)
                write_json(run / "manifest.json", dict(config=dict(arm="llama8b_lora", task=task,
                    condition=condition, is_smoke=False, plan_sha256="tiny-fixture"), recipe=recipe,
                    plan_sha256="tiny-fixture",
                    index_sha256={split: digest(run / f"{split}_indices.npy") for split in ("train", "val", "test")}))
                dataset = TextDataset(sequences, labels, tokenizer, task, recipe["max_length"])
                collate = DynamicPadding(tokenizer.pad_token_id, recipe["max_length"])
                probabilities = []
                for start in range(0, len(labels), 4):
                    inputs = {k: v for k, v in collate([dataset[i] for i in range(start, start + 4)]).items() if k != "labels"}
                    p, _ = infer(model, inputs, True)
                    probabilities.extend(p)
                pd.DataFrame(dict(row_index=indices, label=labels, probability=probabilities)).to_csv(run / "test_predictions.csv", index=False)
                write_json(run / "metrics.json", dict(status="completed", is_smoke=False, test={"tiny_fixture": True},
                    task=task, condition=condition, model="llama8b_lora", plan_sha256="tiny-fixture",
                    selected_checkpoint_sha256=digest(run / "best_trainable.pt")))
                common = dict(model="llama8b_lora", task=task, condition=condition, smoke=False)
                jobs.append(dict(key=key + "_fit", kind="fit", output_dir=str(run), **common))
                jobs.append(dict(key=key + "_attention", kind="attention", fit_job=key + "_fit",
                                 output_dir=str(root / "attention" / key), **common))
            plan = dict(plan_sha256="tiny-fixture", models={"llama8b_lora": {"path": "unused-tiny-loader"}},
                        datasets=datasets, jobs=jobs, samples=16, sample_seed=9950, perturbations=4, source_sha256={})
            write_json(root / "launch_plan.json", plan)
            with patch("src.analysis.llama8_oc_ckd_attention.load_checkpoint", return_value=(model, tokenizer, {"tiny_fixture": True})):
                for job in [j for j in jobs if j["kind"] == "attention"]:
                    audit(argparse.Namespace(run_root=root, job=job["key"], device="cpu"))
                    output = Path(job["output_dir"])
                    metrics = json.loads((output / "metrics.json").read_text())
                    self.assertEqual(metrics["status"], "completed")
                    self.assertEqual(metrics["samples"], 16)
                    self.assertEqual(metrics["capture_probability_delta"], 0)
                    self.assertLessEqual(metrics["max_reference_probability_error"], 1e-6)
                    self.assertTrue(metrics["causal"])
                    self.assertEqual(metrics["attention_readout"], "last unmasked token")
                    if job["task"] == "ckd":
                        self.assertEqual(metrics["truncated_patients"], 0)
                        self.assertTrue((output / "per_patient_head_summary.csv.gz").exists())
                        self.assertTrue((output / "code_attention_summary.csv.gz").exists())
                    else:
                        arrays = np.load(output / "per_example_attention.npz")
                        self.assertEqual(arrays["masses"].shape, (16, 2, 8, 4))
                        self.assertEqual(metrics["counterfactuals"]["random_shuffle"]["samples"], 4)
            paired_summary(root)
            paired = json.loads((root / "ckd_paired_comparison" / "metrics.json").read_text())
            self.assertEqual(paired["paired_patients"], 16)
            self.assertGreater(paired["mean_event_conditioned_code_overlap"], 0)
            self.assertLessEqual(paired["mean_event_conditioned_code_overlap"], 1)
            # Saved source identity guards must reject a changed selected checkpoint.
            first = jobs[0]
            run = Path(first["output_dir"])
            manifest = json.loads((run / "manifest.json").read_text())
            metrics = json.loads((run / "metrics.json").read_text())
            for field, changed in [("condition", "shuffled"), ("task", "ckd"), ("model", "qwen27b_qlora")]:
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    validate_fit(plan, jobs[1], run, manifest, {**metrics, field: changed})
            with self.assertRaisesRegex(ValueError, "different frozen plan"):
                validate_fit(plan, jobs[1], run, manifest, {**metrics, "plan_sha256": "different"})
            (run / "best_trainable.pt").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "selected weights changed"):
                validate_fit(plan, jobs[1], run, manifest, metrics)


if __name__ == "__main__":
    unittest.main()
