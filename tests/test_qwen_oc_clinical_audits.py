import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.analysis.oc_attention_core import KEYS, METRICS, label, oc_statistics, perturbations
from src.analysis.ckd_attention_audit import capture_readout_attention, clinical_groups, summarize_readout
from src.analysis.qwen_oc_audit import reference_batch_groups, replace_batch_row
from scripts.run_qwen_oc_clinical_audits import ROOT, command_for, validate_idle_attention, validate_resume_states


class OCControls(unittest.TestCase):
    def test_original_batches_preserve_row_order_companions_and_partial_tail(self):
        groups = reference_batch_groups([3, 8, 6], [8, 2, 6, 1, 3], 2)
        self.assertEqual([(rows.tolist(), selected) for rows, selected in groups],
                         [([8, 2], [(1, 0)]), ([6, 1], [(2, 0)]), ([3], [(0, 0)])])
        for samples, order, size in [([3], [3, 3], 2), ([9], [0, 1], 2),
                                     ([0, 0], [0, 1], 2), ([0], [0, 1], 0)]:
            with self.assertRaises(ValueError): reference_batch_groups(samples, order, size)

    def test_control_replaces_only_one_row_without_changing_padding(self):
        import torch
        inputs = dict(input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
                      attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]))
        result = replace_batch_row(inputs, 1, torch.tensor([7, 8]))
        self.assertTrue(torch.equal(result["input_ids"], torch.tensor([[1, 2, 3, 0], [7, 8, 0, 0]])))
        self.assertTrue(torch.equal(result["attention_mask"], inputs["attention_mask"]))
        self.assertEqual(inputs["input_ids"][1, 0].item(), 4)
        with self.assertRaises(ValueError): replace_batch_row(inputs, 1, torch.tensor([1, 2, 3]))

    def test_maximal_chain_is_not_any_increasing_pair(self):
        s = ["B"] * 20
        s[0], s[7] = "W", "D"
        self.assertEqual(label(s), 1)
        s[14] = "W"
        self.assertEqual(label(s), 0)
        s[14] = "U"
        self.assertEqual(label(s), 1)

    def test_paper_U_key_is_used_instead_of_legacy_N(self):
        s = ["B"] * 20
        s[0], s[7] = "X", "U"
        self.assertEqual(label(s), 1)
        s[7] = "N"
        self.assertEqual(label(s), 0)

    def test_count_preserving_controls_use_actual_oracle(self):
        rng = np.random.default_rng(42)
        alphabet = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for seed in range(100):
            original = "\x1f".join(rng.choice(alphabet, 20))
            result = perturbations(original, seed)
            for variant, (changed, expected) in result.items():
                self.assertEqual(sorted(original.split("\x1f")), sorted(changed.split("\x1f")))
                self.assertEqual(label(changed), expected)
                if variant == "count_matched_label_flip": self.assertNotEqual(expected, label(original))
                if variant == "distractor_swap": self.assertEqual(expected, label(original))

    def test_no_keys_has_missing_key_metrics_and_prompt_readout(self):
        a = np.tri(22); a /= a.sum(-1, keepdims=True)
        mass, base = oc_statistics(a[None], [[i+1] for i in range(20)], ["B"]*20)
        self.assertTrue(np.isnan(mass[:5]).all())
        self.assertTrue(np.isnan(base[METRICS.index("readout_to_key")]))
        self.assertAlmostEqual(mass[METRICS.index("readout_to_distractor"),0],20/22)

    def test_lag_uniform_baseline_respects_causal_visibility(self):
        s = ["B"]*20; s[0],s[7],s[14] = "W","D","Q"
        a = np.tri(22); a /= a.sum(-1, keepdims=True)
        mass, base = oc_statistics(a[None], [[i+1] for i in range(20)], s)
        valid = np.isfinite(base)
        np.testing.assert_allclose(mass[valid,0]/base[valid],1)
        with self.assertRaises(ValueError): oc_statistics(np.ones((1,22,22))/22, [[i+1] for i in range(20)], s)

    def test_fit_command_preserves_budget_and_has_no_test_stopping(self):
        plan = dict(model_source=dict(snapshot="/tmp/model"),data_dir="/tmp/data",seed=9950)
        command, output = command_for(dict(kind="fit",tag="ocdet"),Path("/tmp/run"),plan,batch=32,attempt=2)
        self.assertEqual(command[command.index("--gradient-accumulation-steps")+1],"8")
        self.assertNotIn("--stop-on-perfect-test",command)
        self.assertEqual(output.name,"attempt_002")


class ClinicalReadout(unittest.TestCase):
    def test_variable_length_event_spans_exclude_prompt_and_suffix(self):
        events=["CCS_1","CCS_123","OTHER"]
        text="Sequential events: "+" ".join(events)+"\nOutcome (0 or 1):"
        offsets=[(i,i+1) for i in range(len(text))]
        groups=clinical_groups(offsets,events)
        self.assertEqual([len(g) for g in groups],[5,7,5])
        self.assertEqual(''.join(text[i] for i in groups[1]),'CCS_123')
        with self.assertRaises(ValueError): clinical_groups(offsets[:18],events)

    def test_leading_space_tokens_and_merged_events(self):
        groups=clinical_groups([(0,18),(18,24),(24,32)],['CCS_1','CCS_123'])
        self.assertEqual(groups,[[1],[2]])
        with self.assertRaises(ValueError): clinical_groups([(18,31)],['CCS_1','CCS_123'])

    def test_uniform_token_baseline_accounts_for_event_token_count(self):
        mass,base,entropy=summarize_readout(np.ones((2,10))/10,[[1,2],[3,4,5]])
        np.testing.assert_allclose(mass,np.array([[.2,.3],[.2,.3]]))
        np.testing.assert_allclose(mass/base,1)
        np.testing.assert_allclose(entropy,1)

    def test_real_hybrid_capture_matches_eager_and_preserves_native_outputs(self):
        import torch
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from src.experiments.kip_efficiency_models import LastTokenClassifier
        from src.analysis.kip_checkpoint_attention import infer,attention_layers
        config=Qwen3_5TextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_num_key_heads=2,
            linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,
            layer_types=['linear_attention','full_attention'],attn_implementation='sdpa')
        model=LastTokenClassifier(Qwen3_5TextModel(config)).eval()
        inputs=dict(input_ids=torch.tensor([[1,2,3,4,5]]),attention_mask=torch.ones((1,5),dtype=torch.long))
        original_interface=ALL_ATTENTION_FUNCTIONS['sdpa']
        expected,_=infer(model,inputs,True)
        captured={}
        with capture_readout_attention(captured):
            actual,_=infer(model,inputs,True)
        np.testing.assert_array_equal(actual,expected)
        self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],original_interface)
        self.assertEqual(captured[1].shape,(4,5))
        model.backbone.set_attn_implementation('eager')
        _,arrays=infer(model,inputs,True,True)
        eager=attention_layers(model.config,arrays)[0][1][0,:,-1,:].float().detach().numpy()
        np.testing.assert_allclose(captured[1],eager,atol=.002)
        with self.assertRaises(RuntimeError):
            with capture_readout_attention({}): raise RuntimeError('test cleanup')
        self.assertIs(ALL_ATTENTION_FUNCTIONS['sdpa'],original_interface)

    def test_padded_clinical_batch_capture_uses_selected_last_unmasked_query(self):
        import torch
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        from src.experiments.kip_efficiency_models import LastTokenClassifier
        from src.analysis.kip_checkpoint_attention import capture_native_attention, infer
        from src.experiments.kip_encoder_training import DynamicPadding
        config = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_key_heads=2,
            linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
            layer_types=['linear_attention', 'full_attention'], attn_implementation='sdpa')
        model = LastTokenClassifier(Qwen3_5TextModel(config)).eval()
        batch = DynamicPadding(0, 32)([([1, 2, 3], 0), ([4, 5, 6, 7, 8], 1)])
        inputs = {key: value for key, value in batch.items() if key != 'labels'}
        self.assertEqual(tuple(inputs['input_ids'].shape), (2, 8))
        original, _ = infer(model, inputs, True)
        full = {}
        with capture_native_attention(full, {0: 3, 1: 5}):
            unchanged, _ = infer(model, inputs, True)
        np.testing.assert_array_equal(unchanged, original)
        for row, length in ((0, 3), (1, 5)):
            readout = {}
            with capture_readout_attention(readout, row=row, length=length):
                actual, _ = infer(model, inputs, True)
            np.testing.assert_array_equal(actual, original)
            self.assertEqual(readout[1].shape, (4, length))
            np.testing.assert_allclose(readout[1], full[1][row][:, -1], atol=.002)
            np.testing.assert_allclose(readout[1].sum(-1), 1, atol=.01)


class ReplacementGuards(unittest.TestCase):
    def test_resume_only_failed_queues_without_reported_workers(self):
        extension = dict(status='failed', child_pid=None)
        attention = dict(status='blocked_analysis_failures', child_pid=None)
        validate_resume_states(extension, attention)
        for changed in [dict(status='running'), dict(status='completed'), dict(child_pid=42)]:
            with self.assertRaises(RuntimeError):
                validate_resume_states({**extension, **changed}, attention)
        for changed in [dict(status='running'), dict(status='completed'), dict(child_pid=42)]:
            with self.assertRaises(RuntimeError):
                validate_resume_states(extension, {**attention, **changed})

    def test_replace_only_original_idle_waiter(self):
        state=dict(status='waiting_for_ckd',supervisor_pid=12,child_pid=None,current_job=None)
        identity=dict(pid=12,cwd=str(ROOT),command=[b'python',str(ROOT/'scripts/run_kip_checkpoint_attention.py').encode()])
        validate_idle_attention(state,identity)
        for change in [dict(status='running'),dict(child_pid=13),dict(current_job='fit')]:
            with self.assertRaises(RuntimeError):validate_idle_attention({**state,**change},identity)
        with self.assertRaises(RuntimeError):validate_idle_attention(state,{**identity,'pid':99})
        with self.assertRaises(RuntimeError):validate_idle_attention(state,{**identity,'command':[b'python',b'other.py']})


if __name__ == '__main__':
    unittest.main()
