import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('rnn_launcher', Path(__file__).resolve().parents[1] / 'scripts/run_kip_rnn_transformer.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def metrics(tag='kip_m4', seed=9550, auc=.5):
    return dict(status='completed', tag=tag, seed=seed, is_smoke=False,
                rows=dict(train=400000, val=50000, test=50000), test_informed_stopping=False,
                test=dict(auc=auc, f1=.6, precision=.5, recall=.8, accuracy=.5))


class LauncherTests(unittest.TestCase):
    def test_smoke_or_wrong_counts_cannot_be_reported_as_production(self):
        for changed in ({'is_smoke': True}, {'rows': dict(train=256, val=64, test=64)},
                        {'test_informed_stopping': True}, {'seed': 9551}, {'tag': 'kip_m6'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                launcher.validate_metrics(dict(metrics(), **changed), 'kip_m4', 9550)

    def test_nonfinite_metrics_rejected(self):
        for value in (float('nan'), float('inf'), -.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                launcher.validate_metrics(metrics(auc=value), 'kip_m4', 9550)

    def test_summary_requires_all_seeds_and_uses_sample_sd(self):
        with tempfile.TemporaryDirectory() as name:
            root, jobs = Path(name), []
            for seed, value in zip(launcher.SEEDS, (.4, .5, .6)):
                output = root / str(seed)
                launcher.write(output / 'metrics.json', metrics(seed=seed, auc=value))
                jobs.append(dict(tag='kip_m4', seed=seed, smoke=False, status='completed',
                                 output_dir=str(output), metrics_sha256=launcher.digest(output / 'metrics.json')))
            launcher.summarize(root, {'jobs': jobs[:2]})
            self.assertEqual(len(list(launcher.csv.DictReader((root / 'summary.csv').read_text().splitlines()))), 0)
            launcher.summarize(root, {'jobs': jobs})
            rows = list(launcher.csv.DictReader((root / 'summary.csv').read_text().splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]['auc_mean']), .5)
            self.assertAlmostEqual(float(rows[0]['auc_sample_sd']), .1)
            self.assertIn('0.500 (0.100)', (root / 'table.tex').read_text())

    def test_altered_completed_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / 'run/metrics.json'
            launcher.write(path, metrics())
            job = dict(tag='kip_m4', seed=9550, smoke=False, status='completed',
                       output_dir=str(path.parent), metrics_sha256=launcher.digest(path))
            launcher.write(path, metrics(auc=1.))
            with self.assertRaises(ValueError):
                launcher.summarize(root, {'jobs': [job]})

    def test_queue_commands_keep_fixed_protocol_and_distinct_smoke(self):
        production = launcher.command(Path('/tmp/rnn_test'), 'kip_m6', 9552)
        smoke = launcher.command(Path('/tmp/rnn_smoke'), 'kip_m4', 9550, smoke=True)
        self.assertNotIn('--smoke', production)
        self.assertNotIn('--dry-run', production)
        self.assertEqual(production[production.index('--epochs') + 1], '30')
        self.assertEqual(production[production.index('--patience') + 1], '5')
        self.assertEqual(smoke[smoke.index('--epochs') + 1], '1')
        self.assertIn('--smoke', smoke)

    def test_plan_hash_detects_changes(self):
        plan = dict(rows=400000, seeds=[9550, 9551, 9552])
        plan['plan_sha256'] = launcher.plan_digest(plan)
        self.assertEqual(plan['plan_sha256'], launcher.plan_digest(plan))
        plan['rows'] = 200000
        self.assertNotEqual(plan['plan_sha256'], launcher.plan_digest(plan))


if __name__ == '__main__':
    unittest.main()
