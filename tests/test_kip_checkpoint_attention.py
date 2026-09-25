import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from src.analysis.kip_attention_core import (
    KEYS, METRICS, PREFIX, attention_statistics, counterfactuals,
    fixed_sample, letter_token_groups, validate_sequence,
)
from src.analysis.kip_checkpoint_attention import attention_layers, capture_native_attention, infer, write_json
from scripts import run_kip_checkpoint_attention as queue


def sequence(keys):
    return "\x1f".join(list(keys) + ["B", "E"] * ((20 - len(keys)) // 2))


class ScientificControls(unittest.TestCase):
    def test_every_key_permutation_has_valid_transposition_control(self):
        for tag, keys in KEYS.items():
            for permutation in itertools.permutations(keys):
                original = sequence(permutation)
                label = validate_sequence(original, tag)[1]
                variants = counterfactuals(original, tag)
                self.assertEqual(variants["key_swap"][1], 1 - label)
                self.assertEqual(variants["distractor_swap"][1], label)
                for changed, _ in variants.values():
                    self.assertEqual(sorted(original.split("\x1f")), sorted(changed.split("\x1f")))

    def test_invalid_multiplicity_rejected(self):
        with self.assertRaises(ValueError):
            validate_sequence(sequence("CCOA"), "kip_m4")

    def test_offset_alignment_excludes_letters_in_prompt(self):
        letters = list("COTA")
        offsets = [(0, 10), (11, 17)] + [(len(PREFIX) + 2 * i, len(PREFIX) + 2 * i + 1) for i in range(4)]
        self.assertEqual(letter_token_groups(offsets, letters), [[2], [3], [4], [5]])
        with self.assertRaises(ValueError):
            letter_token_groups([(0, 0), (len(PREFIX), len(PREFIX) + 3)], letters[:2])

    def test_split_token_alignment(self):
        self.assertEqual(letter_token_groups([(len(PREFIX), len(PREFIX)+1)] * 2, ["C"]), [[0, 1]])

    def test_uniform_attention_has_unit_enrichment_with_both_masks(self):
        letters = list("COTA") + ["B"] * 16
        groups = [[i + 1] for i in range(20)]
        for causal in (True, False):
            allowed = np.tri(22) if causal else np.ones((22, 22))
            uniform = allowed / allowed.sum(-1, keepdims=True)
            values, base = attention_statistics(uniform[None], groups, letters, "kip_m4", causal, 21 if causal else 0)
            valid = np.isfinite(base)
            np.testing.assert_allclose(values[valid, 0] / base[valid], 1)
            # Ascending reference order has no inversion pairs; do not report zero.
            self.assertTrue(np.isnan(values[METRICS.index("key_to_inversion"), 0]))

    def test_masked_future_mass_is_not_accepted(self):
        with self.assertRaises(ValueError):
            attention_statistics(np.ones((1, 20, 20)) / 20, [[i] for i in range(20)],
                                 list("COTA") + ["B"] * 16, "kip_m4", True, 19)

    def test_inversion_direction_is_based_on_key_ranks_not_alphabet(self):
        # COTA reference order is inversion-free although not alphabetically sorted.
        letters = list("OCAT") + ["B"] * 16
        a = np.tri(20); a /= a.sum(-1, keepdims=True)
        mass, base = attention_statistics(a[None], [[i] for i in range(20)], letters, "kip_m4", True, 19)
        self.assertTrue(np.isfinite(base[METRICS.index("key_to_inversion")]))
        self.assertTrue(np.isfinite(base[METRICS.index("key_to_noninversion")]))

    def test_sample_is_fixed_stratified_and_without_replacement(self):
        labels = np.array([0] * 700 + [1] * 300)
        a, b = fixed_sample(labels, 100, 42), fixed_sample(labels, 100, 42)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(len(set(a)), 100)
        self.assertEqual(int(labels[a].sum()), 30)

    def test_hybrid_layer_ids_are_not_compacted(self):
        config = type("Config", (), dict(num_hidden_layers=4, layer_types=["linear_attention", "full_attention"] * 2))()
        self.assertEqual([i for i, _ in attention_layers(config, ["a", "b"])], [1, 3])
        self.assertEqual([i for i, _ in attention_layers(config, [None, "a", None, "b"])], [1, 3])
        with self.assertRaises(ValueError):
            attention_layers(config, [None, None])


class ModelInterfaces(unittest.TestCase):
    def test_native_capture_preserves_batched_padded_outputs_and_masks(self):
        import torch
        from transformers import (
            BertConfig, BertForSequenceClassification, LlamaConfig, LlamaForSequenceClassification,
            RobertaConfig, RobertaForSequenceClassification, Qwen2Config, Qwen2ForSequenceClassification,
        )
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        models = [
            BertForSequenceClassification(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=32, num_labels=2, attn_implementation="sdpa")),
            RobertaForSequenceClassification(RobertaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, intermediate_size=32, num_labels=2, pad_token_id=0, attn_implementation="sdpa")),
            LlamaForSequenceClassification(LlamaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, intermediate_size=32, num_labels=2,
                pad_token_id=0, attn_implementation="sdpa")),
            Qwen2ForSequenceClassification(Qwen2Config(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, intermediate_size=32, num_labels=2,
                pad_token_id=0, attn_implementation="sdpa")),
        ]
        ids = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 0, 0]])
        inputs = dict(input_ids=ids, attention_mask=(ids != 0).long())
        original_dispatch = ALL_ATTENTION_FUNCTIONS["sdpa"]
        for model in models:
            with self.subTest(model=model.config.model_type):
                model.eval()
                before, _ = infer(model, inputs, False)
                captured = {}
                with capture_native_attention(captured, {0: 4, 1: 6}):
                    after, _ = infer(model, inputs, False)
                np.testing.assert_array_equal(before, after)
                self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], original_dispatch)
                self.assertEqual(sorted(captured), [0, 1])
                model.set_attn_implementation("eager")
                _, eager = infer(model, inputs, False, True)
                for row, length in [(0, 4), (1, 6)]:
                    self.assertEqual(captured[0][row].shape, (2, length, length))
                    # First-layer Q/K are identical before any SDPA/eager reductions diverge.
                    np.testing.assert_allclose(captured[0][row], eager[0][row, :, :length, :length].float().numpy(), atol=.004)
                    np.testing.assert_allclose(captured[0][row].sum(-1), 1, atol=.004)
                    if model.config.model_type in ("llama", "qwen2"):
                        np.testing.assert_array_equal(np.triu(captured[0][row], 1), 0)

    def test_native_capture_restores_dispatch_on_error(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        original = ALL_ATTENTION_FUNCTIONS["sdpa"]
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with capture_native_attention({}, {0: 1}):
                raise RuntimeError("intentional")
        self.assertIs(ALL_ATTENTION_FUNCTIONS["sdpa"], original)

    def test_hf_loading_info_sets_are_saved_without_discarding_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "manifest.json"
            write_json(destination, {"loading_info": {"missing_keys": {"z", "a"}, "unexpected_keys": set()}})
            self.assertEqual(json.loads(destination.read_text()),
                             {"loading_info": {"missing_keys": ["a", "z"], "unexpected_keys": []}})

    def test_real_tiny_encoder_and_decoder_extract_without_changing_scores(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, LlamaConfig, LlamaForSequenceClassification
        models = [BertForSequenceClassification(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=32, num_labels=2, attn_implementation="eager")),
                  LlamaForSequenceClassification(LlamaConfig(vocab_size=32, hidden_size=16, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, intermediate_size=32, num_labels=2,
                    pad_token_id=0, attn_implementation="eager"))]
        inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones((1, 4), dtype=torch.long))
        for model in models:
            model.eval()
            p1, _ = infer(model, inputs, False)
            p2, arrays = infer(model, inputs, False, True)
            np.testing.assert_allclose(p1, p2, atol=1e-6)
            self.assertEqual(len(attention_layers(model.config, arrays)), 2)

    def test_real_tiny_qwen_hybrid_with_classifier_head(self):
        import torch
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from src.experiments.kip_efficiency_models import LastTokenClassifier
        cfg = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                               linear_num_key_heads=2, linear_num_value_heads=2,
                               linear_key_head_dim=8, linear_value_head_dim=8,
                               layer_types=["linear_attention", "full_attention"], attn_implementation="eager")
        model = LastTokenClassifier(Qwen3_5TextModel(cfg)).eval()
        inputs = dict(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones((1, 4), dtype=torch.long))
        p1, _ = infer(model, inputs, True)
        p2, arrays = infer(model, inputs, True, True)
        np.testing.assert_allclose(p1, p2, atol=1e-6)
        self.assertEqual([i for i, _ in attention_layers(model.config, arrays)], [1])
        model.backbone.set_attn_implementation("sdpa")
        padded = dict(input_ids=torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0]]),
                      attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0]]))
        before, _ = infer(model, padded, True)
        captured = {}
        with capture_native_attention(captured, {0: 4, 1: 5}):
            after, _ = infer(model, padded, True)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(sorted(captured), [1])
        self.assertEqual(captured[1][1].shape, (4, 5, 5))
        np.testing.assert_allclose(captured[1][1].sum(-1), 1, atol=.004)
        np.testing.assert_array_equal(np.triu(captured[1][1], 1), 0)


class PriorityQueue(unittest.TestCase):
    def test_incomplete_ckd_never_releases_dependency(self):
        self.assertFalse(queue.ckd_complete(dict(status="running", jobs=[])))
        self.assertFalse(queue.ckd_complete(dict(status="completed", child_pid=42, jobs=[dict(status="completed")])))
        self.assertFalse(queue.ckd_complete(dict(status="completed", jobs=[])))
        self.assertFalse(queue.ckd_complete(dict(status="completed", jobs=[dict(status="failed")])))
        self.assertTrue(queue.ckd_complete(dict(status="completed", child_pid=None, jobs=[dict(status="completed")])))

    def test_active_llama_cannot_be_paused(self):
        identity = dict(pid=123, cwd=str(queue.ROOT), command=[b"python", b"scripts/run_kip_llama70.py"])
        good = dict(status="waiting_for_ckd", supervisor_pid=123, child_pid=None, jobs=[dict(status="pending")])
        queue.validate_waiting_llama(good, identity)
        for change in [dict(child_pid=9), dict(status="running"), dict(supervisor_pid=777),
                       dict(current_job="smoke"), dict(jobs=[dict(status="completed")])]:
            with self.assertRaises(RuntimeError):
                queue.validate_waiting_llama({**good, **change}, identity)

    def test_real_pause_and_resume_of_disposable_waiter_only(self):
        # This subprocess only sleeps; the production supervisor is never addressed.
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "scripts/run_kip_llama70.py"], cwd=queue.ROOT)
        try:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                status = root / "queue_status.json"
                status.write_text(json.dumps(dict(status="waiting_for_ckd", supervisor_pid=child.pid,
                                                  child_pid=None, jobs=[dict(status="pending")])))
                (root / "launch_plan.json").write_text("{}")
                with mock.patch.object(queue, "LLAMA", status):
                    hold = queue.pause_waiter(root)
                    self.assertIn(queue.process_identity(child.pid)["state"], ("T", "t"))
                    queue.pause_waiter(root, hold)  # Resumption of our own priority supervisor is idempotent.
                    queue.resume_waiter(hold, root)
                    self.assertTrue((root / "llama70_resumed.json").exists())
                    with self.assertRaises(RuntimeError):
                        queue.resume_waiter({**hold, "start_ticks": "wrong"}, root)
        finally:
            if child.poll() is None:
                os.kill(child.pid, signal.SIGCONT)
                child.terminate()
            child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
