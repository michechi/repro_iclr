"""CPU checks for held-out isolation, model selection and checkpoint integrity."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch

from scripts import audit_kip_efficiency_weights as weight_audit
from src.experiments import kip_data_efficiency as runner
from src.experiments import kip_efficiency_models as factory


def fake_snapshot(root: Path, payload: bytes = b'checkpoint payload') -> Path:
    source, blobs = root / 'snapshots' / ('a' * 40), root / 'blobs'
    source.mkdir(parents=True)
    blobs.mkdir()
    digest = hashlib.sha256(payload).hexdigest()
    (blobs / digest).write_bytes(payload)
    (source / 'model-00001-of-00001.safetensors').symlink_to(blobs / digest)
    (source / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'model.weight': 'model-00001-of-00001.safetensors'}}))
    (source / 'config.json').write_text('{"model_type": "test"}')
    return source


class SamplingTests(unittest.TestCase):
    def test_all_fractions_match_historical_stratified_membership(self):
        labels = np.tile(np.array([0, 0, 0, 1, 1], dtype=np.int64), 80000)
        for fraction, size in zip(runner.FRACTIONS, (4000, 40000, 120000, 200000, 300000, 400000)):
            with self.subTest(fraction=fraction):
                actual = runner.selected_indices(labels, fraction, 9950)
                expected = np.arange(len(labels)) if fraction == 1 else train_test_split(
                    np.arange(len(labels)), train_size=fraction, stratify=labels, random_state=9950)[0]
                np.testing.assert_array_equal(actual, np.sort(expected))
                self.assertEqual(len(actual), size)
                self.assertEqual(len(np.unique(actual)), size)
                self.assertAlmostEqual(float(labels[actual].mean()), 0.4, places=12)

    def test_held_out_rows_are_identical_at_every_fraction_and_seed(self):
        labels = np.arange(50000) % 2
        for split in ('val', 'test'):
            for fraction in runner.FRACTIONS:
                for seed in (9950, 9951):
                    np.testing.assert_array_equal(
                        runner.selected_indices(labels, fraction, seed, split=split), np.arange(50000))

    def test_smoke_is_bounded_and_invalid_fractions_rejected(self):
        labels = np.arange(400000) % 2
        self.assertEqual(len(runner.selected_indices(labels, 1.0, 9950, smoke=True)), 256)
        self.assertEqual(len(runner.selected_indices(labels[:50000], 1.0, 9950,
                                                    smoke=True, split='test')), 128)
        for fraction in (0, -0.1, 1.1, float('nan'), float('inf')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                runner.selected_indices(labels, fraction, 9950)

    def test_invalid_training_configuration_is_rejected(self):
        base = ['--arm', 'qwen27b_qlora', '--model-path', '/missing', '--output-dir', '/missing-output',
                '--fraction', '0.01']
        for extra in (['--batch-size', '0'], ['--gradient-accumulation-steps', '0'],
                      ['--lr', 'nan'], ['--fraction', 'nan'], ['--optimizer-offload', 'cpu']):
            with self.subTest(extra=extra), mock.patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(base + extra)


class WeightIntegrityTests(unittest.TestCase):
    def test_hash_audit_and_per_fit_provenance_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_snapshot(root)
            report = weight_audit.audit_weights(source, root / 'evidence' / 'weights.json')
            provenance = runner.model_provenance(source)
            self.assertTrue(report['verified'])
            self.assertEqual(report['shards'], 1)
            for name, record in report['weights'].items():
                self.assertEqual(record['sha256'], provenance['weights'][name]['expected_hf_blob_sha256'])
                self.assertEqual(record['bytes'], provenance['weights'][name]['bytes'])
            self.assertEqual(json.loads((root / 'evidence' / 'weights.json').read_text()), report)
            self.assertFalse(list((root / 'evidence').glob('*.tmp')))

    def test_corrupted_payload_fails_with_persistent_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_snapshot(root)
            shard = source / 'model-00001-of-00001.safetensors'
            shard.write_bytes(b'corrupted payload!')
            with self.assertRaisesRegex(ValueError, 'payload SHA256'):
                weight_audit.audit_weights(source, root / 'weights.json')
            evidence = json.loads((root / 'weights.json').read_text())
            self.assertEqual(evidence['status'], 'failed')
            self.assertFalse(evidence['verified'])
            self.assertFalse(next(iter(evidence['weights'].values()))['verified'])

    def test_missing_shard_and_non_blob_target_are_rejected(self):
        for missing in (True, False):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = fake_snapshot(root)
                shard = source / 'model-00001-of-00001.safetensors'
                shard.unlink()
                if not missing:
                    shard.write_bytes(b'unidentified payload')
                with self.assertRaises((ValueError, FileNotFoundError)):
                    weight_audit.audit_weights(source, root / 'weights.json')
                self.assertFalse(json.loads((root / 'weights.json').read_text())['verified'])
                if missing:
                    with self.assertRaises(FileNotFoundError):
                        runner.model_provenance(source)

    def test_empty_and_traversing_index_rejected(self):
        for weight_map in ({}, {'x': '../outside.safetensors'}, {'x': 123}):
            with self.subTest(weight_map=weight_map), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = fake_snapshot(root)
                (source / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
                with self.assertRaises(ValueError):
                    weight_audit.audit_weights(source, root / 'weights.json')

    def test_mutation_during_hash_fails_even_when_digest_was_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_snapshot(root)
            original = weight_audit.sha256

            def hash_then_mutate(path):
                digest = original(path)
                path.write_bytes(path.read_bytes() + b'changed')
                return digest

            with mock.patch.object(weight_audit, 'sha256', side_effect=hash_then_mutate), \
                    self.assertRaisesRegex(ValueError, 'changed during hashing'):
                weight_audit.audit_weights(source, root / 'weights.json')


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        if kwargs.get('truncation') is not False:
            raise AssertionError('input truncation must be disabled')
        return {'input_ids': [[1, 2 if text.endswith(' A') else 3] for text in texts]}

    def save_pretrained(self, path):
        path.mkdir(exist_ok=True)
        (path / 'tokenizer_config.json').write_text('{"test_only": true}')


class TinyClassifier(torch.nn.Module):
    def __init__(self, full=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 4)
        self.embedding.weight.requires_grad_(full)
        self.score = torch.nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask):
        last = attention_mask.sum(1) - 1
        hidden = self.embedding(input_ids)[torch.arange(len(input_ids)), last]
        return SimpleNamespace(logits=self.score(hidden))


class CPUTorch:
    """Route the production driver's GPU allocation to CPU for its tiny model."""
    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: 'CPU test double',
                           empty_cache=lambda: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(value):
        return torch.device('cpu')


class TrainingLifecycleTests(unittest.TestCase):
    def test_best_validation_checkpoint_restored_and_test_is_not_used_for_selection(self):
        torch.set_num_threads(1)
        for arm in ('qwen14b_qlora', 'llama8b_full'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = fake_snapshot(root)
                args = runner.parse_args([
                    '--arm', arm, '--model-path', str(source), '--output-dir', str(root / 'run'),
                    '--fraction', '0.00002', '--smoke', '--epochs', '6', '--patience', '2',
                    '--batch-size', '3', '--gradient-accumulation-steps', '2',
                    '--eval-batch-size', '128', '--threads', '1', '--progress-steps', '100'])
                torch.manual_seed(73)
                model = TinyClassifier(full=arm == 'llama8b_full')
                initial = {name: value.detach().clone() for name, value in model.named_parameters()}
                calls, states, split_loaders = [], {}, {}
                real_evaluate = runner.evaluate

                def data_read(data_dir, tag, split):
                    rows = 400000 if split == 'train' else 50000
                    labels = np.arange(rows) % 2
                    sequences = ['\x1f'.join([letter] * 20) for letter in ('A', 'B')]
                    return [sequences[label] for label in labels], labels, {'test_fixture_split': split}

                def controlled_evaluate(actual_model, loader, device, precision='bf16'):
                    calls.append(loader)
                    loss, probabilities, labels = real_evaluate(actual_model, loader, device, 'fp32')
                    if len(calls) <= 4:
                        split_loaders['val'] = loader
                        states[len(calls)] = {name: value.detach().clone() for name, value in model.named_parameters()}
                        loss = (0.4, 0.2, 0.7, 0.8)[len(calls) - 1]
                    else:
                        for name, value in model.named_parameters():
                            torch.testing.assert_close(value, states[2][name], rtol=0, atol=0)
                    return loss, probabilities, labels

                with ExitStack() as stack:
                    stack.enter_context(mock.patch.object(runner, 'torch', CPUTorch()))
                    stack.enter_context(mock.patch.object(runner, 'read_split', side_effect=data_read))
                    loader_mock = stack.enter_context(mock.patch.object(factory, 'load_classifier',
                        return_value=(model, TinyTokenizer(), {'test_only': True})))
                    stack.enter_context(mock.patch.object(runner, 'evaluate', side_effect=controlled_evaluate))
                    stack.enter_context(mock.patch.object(runner.importlib.metadata, 'version', return_value='test'))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    result = runner.train(args)
                    with self.assertRaises(FileExistsError):
                        runner.train(args)
                loader_mock.assert_called_once()
                self.assertEqual(result['best_epoch'], 2)
                self.assertEqual(result['epochs_completed'], 4)
                self.assertTrue(result['early_stopped'])
                self.assertEqual(result['rows'], {'train': 8, 'val': 128, 'test': 128})
                self.assertEqual(len(calls), 6)
                self.assertTrue(all(loader is split_loaders['val'] for loader in calls[:5]))
                self.assertIsNot(calls[-1], split_loaders['val'])
                self.assertEqual(result['validation']['threshold'], result['test']['threshold'])
                output = args.output_dir
                checkpoint = output / 'best_trainable.pt'
                self.assertEqual(checkpoint.exists(), arm != 'llama8b_full')
                if checkpoint.exists():
                    saved = torch.load(checkpoint, weights_only=True)
                    self.assertEqual(saved['epoch'], 2)
                    self.assertEqual(set(saved['state_dict']), {'score.weight', 'score.bias'})
                    for name, value in saved['state_dict'].items():
                        torch.testing.assert_close(value, states[2][name], rtol=0, atol=0)
                    torch.testing.assert_close(model.embedding.weight, initial['embedding.weight'], rtol=0, atol=0)
                self.assertFalse(torch.equal(initial['score.weight'], states[2]['score.weight']))
                predictions = pd.read_csv(output / 'test_predictions.csv')
                np.testing.assert_array_equal(predictions.row_index, np.load(output / 'test_indices.npy'))
                self.assertAlmostEqual(roc_auc_score(predictions.label, predictions.probability), result['test']['auc'])
                self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'completed')
                for relative, digest in result['artifact_sha256'].items():
                    self.assertEqual(runner.sha256(output / relative), digest)


class TestInformedStoppingTests(unittest.TestCase):
    def test_perfect_test_stopping_is_opt_in(self):
        base = ['--arm', 'qwen27b_qlora', '--model-path', '/missing',
                '--output-dir', '/missing-output', '--fraction', '0.01']
        self.assertFalse(runner.parse_args(base).stop_on_perfect_test)
        self.assertTrue(runner.parse_args(base + ['--stop-on-perfect-test']).stop_on_perfect_test)

    def test_diagnostic_restores_all_rngs_loader_generator_and_model_mode_even_on_error(self):
        for training in (False, True):
            for fail in (False, True):
                with self.subTest(training=training, fail=fail):
                    model = TinyClassifier()
                    model.train(training)
                    generator = torch.Generator().manual_seed(551)
                    random.seed(123)
                    np.random.seed(456)
                    torch.manual_seed(789)
                    random_state, numpy_state = random.getstate(), np.random.get_state()
                    torch_state, loader_state = torch.get_rng_state(), generator.get_state()
                    sentinel = (0.25, np.array([0.2, 0.8]), np.array([0, 1]))

                    def consume_rng(*args, **kwargs):
                        model.eval()
                        random.random()
                        np.random.random(7)
                        torch.rand(9)
                        torch.rand(11, generator=generator)
                        if fail:
                            raise RuntimeError('diagnostic failure')
                        return sentinel

                    with mock.patch.object(runner, 'evaluate', side_effect=consume_rng):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'):
                                runner.diagnostic_test_evaluation(model, object(), torch.device('cpu'), generator)
                        else:
                            actual = runner.diagnostic_test_evaluation(model, object(), torch.device('cpu'), generator)
                            self.assertIs(actual, sentinel)
                    self.assertEqual(model.training, training)
                    self.assertEqual(random.getstate(), random_state)
                    actual_numpy_state = np.random.get_state()
                    self.assertEqual(actual_numpy_state[0], numpy_state[0])
                    np.testing.assert_array_equal(actual_numpy_state[1], numpy_state[1])
                    self.assertEqual(actual_numpy_state[2:], numpy_state[2:])
                    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
                    torch.testing.assert_close(generator.get_state(), loader_state, rtol=0, atol=0)

    @staticmethod
    def near_perfect_probabilities(labels):
        """One misranked pair: prints 1.0000 but is strictly below one."""
        negative, positive = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
        probabilities = np.empty(len(labels), dtype=np.float64)
        probabilities[negative] = np.linspace(0.2, 0.4, len(negative))
        probabilities[positive] = np.linspace(0.6, 0.8, len(positive))
        probabilities[negative[-1]] = 0.6000001
        return probabilities

    def run_controlled_training(self, root, *, flag=True, smoke=False, losses=(0.4, 0.3, 0.2),
                                validation_auc=(1, 1, 1), diagnostic_aucs=('near', 1)):
        """Train eight real examples; control full validation/test score arrays."""
        torch.set_num_threads(1)
        source = fake_snapshot(root)
        args_list = [
            '--arm', 'qwen14b_qlora', '--model-path', str(source), '--output-dir', str(root / 'run'),
            '--fraction', '0.00002', '--epochs', str(len(losses)), '--patience', '3',
            '--batch-size', '3', '--gradient-accumulation-steps', '2',
            '--eval-batch-size', '128', '--threads', '1', '--progress-steps', '100']
        if flag:
            args_list.append('--stop-on-perfect-test')
        if smoke:
            args_list.append('--smoke')
        args = runner.parse_args(args_list)
        torch.manual_seed(73)
        model = TinyClassifier()
        loaders, calls, states = {}, [], {}
        original_loader = runner.DataLoader

        def data_read(data_dir, tag, split):
            rows = 400000 if split == 'train' else 50000
            labels = np.arange(rows) % 2
            sequences = ['\x1f'.join([letter] * 20) for letter in ('A', 'B')]
            return [sequences[label] for label in labels], labels, {'test_fixture_split': split}

        def capture_loader(*args, **kwargs):
            loader = original_loader(*args, **kwargs)
            loaders[('train', 'val', 'test')[len(loaders)]] = loader
            return loader

        val_calls = 0
        test_calls = 0

        def controlled_evaluate(actual_model, loader, device, precision='bf16'):
            nonlocal val_calls, test_calls
            self.assertIs(actual_model, model)
            model.eval()
            labels = loader.dataset.labels.copy()
            if loader is loaders['val']:
                val_calls += 1
                calls.append(('val', val_calls))
                epoch = min(val_calls, len(losses))
                states[epoch] = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                probabilities = labels * 0.2 + 0.3 if validation_auc[epoch - 1] == 1 else np.full(len(labels), 0.5)
                return losses[epoch - 1], probabilities, labels
            self.assertIs(loader, loaders['test'])
            test_calls += 1
            calls.append(('test', test_calls))
            # Simulate DataLoader/other diagnostic RNG consumption; it must not
            # affect subsequent training or dropout after an unsuccessful check.
            random.random()
            np.random.random()
            torch.rand(1)
            auc = diagnostic_aucs[min(test_calls - 1, len(diagnostic_aucs) - 1)]
            probabilities = self.near_perfect_probabilities(labels) if auc == 'near' else (
                0.6 + labels * 0.1 if auc == 1 else np.full(len(labels), 0.5))
            return 0.8, probabilities, labels

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, 'torch', CPUTorch()))
            stack.enter_context(mock.patch.object(runner, 'read_split', side_effect=data_read))
            stack.enter_context(mock.patch.object(runner, 'DataLoader', side_effect=capture_loader))
            stack.enter_context(mock.patch.object(factory, 'load_classifier',
                return_value=(model, TinyTokenizer(), {'test_only': True})))
            stack.enter_context(mock.patch.object(runner, 'evaluate', side_effect=controlled_evaluate))
            stack.enter_context(mock.patch.object(runner.importlib.metadata, 'version', return_value='test'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            metrics = runner.train(args)
        return metrics, calls, states, args.output_dir

    def test_near_perfect_test_continues_then_exact_auc_stops_and_reuses_proven_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, calls, states, output = self.run_controlled_training(Path(directory))
            self.assertEqual(calls, [('val', 1), ('test', 1), ('val', 2), ('test', 2)])
            self.assertEqual(metrics['epochs_completed'], 2)
            self.assertEqual(metrics['best_epoch'], 2)
            self.assertEqual(metrics['stop_reason'], 'test_auc_1')
            self.assertEqual(metrics['stopping_rule'], 'validation_gated_test_auc_1')
            self.assertTrue(metrics['test_informed_stopping'])
            self.assertEqual(metrics['rows']['test'], 50000)
            self.assertEqual(metrics['test']['auc'], 1.0)
            self.assertEqual(metrics['test_at_0_5']['accuracy'], 0.5)
            checks = [json.loads(line) for line in (output / 'test_checks.jsonl').read_text().splitlines()]
            self.assertEqual([check['epoch'] for check in checks], [1, 2])
            first = pd.read_csv(output / 'test_check_epoch_001.csv')
            first_auc = roc_auc_score(first.label, first.probability)
            self.assertLess(first_auc, 1)
            self.assertEqual(f'{first_auc:.4f}', '1.0000')
            trigger = pd.read_csv(output / 'test_check_epoch_002.csv')
            final = pd.read_csv(output / 'test_predictions.csv')
            for column in ('row_index', 'label', 'probability'):
                np.testing.assert_array_equal(trigger[column], final[column])
            saved = torch.load(output / 'best_trainable.pt', weights_only=True)
            self.assertEqual(saved['epoch'], 2)
            for name, value in saved['state_dict'].items():
                torch.testing.assert_close(value, states[2][name], rtol=0, atol=0)
            for relative, digest in metrics['artifact_sha256'].items():
                self.assertEqual(runner.sha256(output / relative), digest)

    def test_validation_gate_requires_improvement_and_exact_validation_auc(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, calls, _, output = self.run_controlled_training(
                Path(directory), losses=(0.4, 0.5, 0.3), validation_auc=(0.5, 1, 1), diagnostic_aucs=(1,))
            self.assertEqual(calls, [('val', 1), ('val', 2), ('val', 3), ('test', 1)])
            self.assertEqual(metrics['stop_reason'], 'test_auc_1')
            checks = [json.loads(line) for line in (output / 'test_checks.jsonl').read_text().splitlines()]
            self.assertEqual([check['epoch'] for check in checks], [3])

    def test_smoke_cannot_use_test_checks_or_claim_test_informed_stopping(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, calls, _, output = self.run_controlled_training(
                Path(directory), smoke=True, losses=(0.4, 0.3), validation_auc=(1, 1), diagnostic_aucs=(1,))
            self.assertEqual(calls, [('val', 1), ('val', 2), ('val', 3), ('test', 1)])
            self.assertEqual(metrics['epochs_completed'], 2)
            self.assertFalse(metrics['test_informed_stopping'])
            self.assertNotEqual(metrics['stop_reason'], 'test_auc_1')
            self.assertFalse((output / 'test_checks.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
