"""CPU checks for the CKD supervisor's resource exclusion and recovery."""
import fcntl
import json
import os
from pathlib import Path
import subprocess

import pytest

from scripts import run_qwen_ckd as launcher


@pytest.fixture
def args(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'SOURCE_FILES', ())
    model, data = tmp_path / 'model', tmp_path / 'data'
    model.mkdir()
    data.mkdir()
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json'):
        (model / name).write_text('{}')
    for name in launcher.DATA_FILES:
        (data / name).write_text('label\n0\n1\n')
    prerequisite = tmp_path / 'kip_status.json'
    prerequisite.write_text(json.dumps({'status': 'completed', 'child_pid': None}))
    parsed = launcher.parse_args(['--run-root', str(tmp_path / 'run'), '--model-path', str(model),
                                  '--data-dir', str(data), '--prerequisite-status', str(prerequisite)])
    monkeypatch.setattr(launcher, 'gpu_ready', lambda: (True, 'test GPU available'))
    monkeypatch.setattr(launcher, 'ensure_model_audit', lambda *unused: {'status': 'verified'})
    monkeypatch.setattr(launcher, 'verify_model_audit', lambda *unused: None)
    return parsed


def metrics(job):
    result = {'status': 'completed', 'seed': 42, 'data': job['data'], 'is_smoke': job['is_smoke']}
    if job['is_smoke']:
        result['validation'] = {'auc': .65}
    else:
        result.update(rows={'train': 2, 'val': 2, 'test': 2},
                      test={'auc': .65, 'auprc': .2, 'f1': .25}, best_epoch=6, epochs_completed=11)
    return result


def fake_children(monkeypatch, behavior=None):
    calls = []

    class Child:
        pid = 54321

        def __init__(self, command, **options):
            assert options['stdin'] == subprocess.DEVNULL
            assert len(options['pass_fds']) == 2
            for descriptor in options['pass_fds']:
                assert os.fstat(descriptor)
            output = Path(command[command.index('--output-dir') + 1])
            assert output.is_dir() and not list(output.iterdir())
            job = {'data': command[command.index('--data') + 1], 'is_smoke': '--smoke' in command}
            calls.append({'command': command, 'output': output, 'job': job})
            self.code, write_metrics, text = behavior(calls) if behavior else (0, True, 'done\n')
            options['stdout'].write(text)
            if write_metrics:
                launcher.atomic_json(metrics(job), output / 'metrics.json')

        def wait(self):
            return self.code

    monkeypatch.setattr(launcher.subprocess, 'Popen', Child)
    return calls


def test_jobs_keep_effective_batch_and_no_test_based_stopping(args):
    plan = launcher.build_plan(args)
    assert [(job['data'], job['is_smoke']) for job in plan['jobs']] == [
        ('ordered', True), ('ordered', False), ('shuffled', False)]
    assert plan['production_rows'] == {'train': 2, 'val': 2, 'test': 2}
    assert plan['required_prerequisite_status'] == 'completed'
    for job in plan['jobs']:
        for batch in (1, 2):
            command = launcher.training_command(args, job, args.run_root, batch, batch)
            assert int(command[command.index('--batch-size') + 1]) * int(
                command[command.index('--gradient-accumulation-steps') + 1]) == 32
            assert '--stop-on-perfect-test' not in command


@pytest.mark.parametrize('status', ['running', 'failed', 'aborted', 'completed_with_failures', None])
def test_prerequisite_never_treats_failure_as_completion(args, status):
    launcher.atomic_json({'status': status}, args.prerequisite_status)
    assert launcher.prerequisite_ready(args.prerequisite_status)[0] is False


def test_prerequisite_requires_no_reported_child(args):
    launcher.atomic_json({'status': 'completed', 'child_pid': 456}, args.prerequisite_status)
    assert not launcher.prerequisite_ready(args.prerequisite_status)[0]
    args.prerequisite_status.write_text('incomplete')
    assert not launcher.prerequisite_ready(args.prerequisite_status)[0]


def test_waiting_for_kip_does_not_hold_gpu_lock(args, monkeypatch):
    launcher.atomic_json({'status': 'running'}, args.prerequisite_status)
    calls = []
    gpu_path = launcher.ROOT / 'results/.kip_data_efficiency_gpu.lock'

    def finish_kip(seconds):
        state = json.loads((args.run_root / 'queue_status.json').read_text())
        assert state['status'] == 'waiting_for_kip'
        with gpu_path.open('a') as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher.atomic_json({'status': 'completed'}, args.prerequisite_status)
        calls.append(seconds)

    monkeypatch.setattr(launcher.time, 'sleep', finish_kip)
    children = fake_children(monkeypatch)
    state = launcher.supervise(args, launcher.build_plan(args))
    assert calls == [30]
    assert len(children) == 3
    assert state['status'] == 'completed'


def test_held_kip_lock_prevents_ckd_start_even_when_status_completed(args, monkeypatch):
    gpu_path = launcher.ROOT / 'results/.kip_data_efficiency_gpu.lock'
    gpu_path.parent.mkdir()
    blocker = gpu_path.open('a')
    fcntl.flock(blocker, fcntl.LOCK_EX | fcntl.LOCK_NB)
    calls = []

    def release_kip(seconds):
        state = json.loads((args.run_root / 'queue_status.json').read_text())
        assert state['status'] == 'waiting_for_gpu'
        assert 'lock' in state['wait_reason']
        calls.append(seconds)
        blocker.close()

    monkeypatch.setattr(launcher.time, 'sleep', release_kip)
    children = fake_children(monkeypatch)
    launcher.supervise(args, launcher.build_plan(args))
    assert len(calls) == 1 and len(children) == 3


def test_gpu_guard_rejects_processes_low_memory_and_query_errors(monkeypatch):
    def fake_run(command, **unused):
        stdout = '130000\n' if '--query-gpu=memory.free' in command else '123, training\n'
        return subprocess.CompletedProcess(command, 0, stdout, '')

    # Undo the fixture's GPU mock by not requesting that fixture.
    monkeypatch.setattr(launcher.subprocess, 'run', fake_run)
    assert not launcher.gpu_ready()[0]
    monkeypatch.setattr(launcher.subprocess, 'run', lambda command, **unused:
                        subprocess.CompletedProcess(command, 0, '10000\n' if '--query-gpu=memory.free' in command else '', ''))
    assert not launcher.gpu_ready()[0]
    monkeypatch.setattr(launcher.subprocess, 'run', lambda command, **unused:
                        subprocess.CompletedProcess(command, 0, '130000\n' if '--query-gpu=memory.free' in command else '', ''))
    assert launcher.gpu_ready()[0]

    def unavailable(*unused, **kwargs):
        raise FileNotFoundError('nvidia-smi')

    monkeypatch.setattr(launcher.subprocess, 'run', unavailable)
    assert not launcher.gpu_ready()[0]


def test_smoke_oom_retries_clean_and_persists_smaller_batches(args, monkeypatch):
    calls = fake_children(monkeypatch, lambda calls: (1, False, 'torch.OutOfMemoryError: CUDA out of memory\n')
                          if len(calls) == 1 else (0, True, 'done\n'))
    state = launcher.supervise(args, launcher.build_plan(args))
    assert state['status'] == 'completed' and len(calls) == 4
    assert calls[0]['output'] != calls[1]['output']
    assert (calls[0]['output'] / 'queue_attempt_error.json').exists()
    assert not (calls[0]['output'] / 'metrics.json').exists()
    for call in calls[1:]:
        command = call['command']
        assert command[command.index('--batch-size') + 1] == '1'
        assert command[command.index('--gradient-accumulation-steps') + 1] == '32'
        assert command[command.index('--eval-batch-size') + 1] == '1'
    assert state['jobs'][0]['attempts'][0]['out_of_memory'] is True


def test_production_oom_preserves_failed_artifacts_and_restarts(args, monkeypatch):
    calls = fake_children(monkeypatch, lambda calls: (1, False, 'CUDA out of memory\n')
                          if len(calls) == 2 else (0, True, 'done\n'))
    state = launcher.supervise(args, launcher.build_plan(args))
    assert len(calls) == 4 and state['status'] == 'completed'
    assert calls[1]['output'].name == 'attempt_001'
    assert calls[2]['output'].name == 'attempt_002'
    assert state['jobs'][1]['retry_note'].startswith('fresh model')


def test_non_oom_error_fails_loudly_before_later_jobs(args, monkeypatch):
    calls = fake_children(monkeypatch, lambda calls: (1, False, 'ValueError: bad input\n'))
    with pytest.raises(RuntimeError, match='exited with code'):
        launcher.supervise(args, launcher.build_plan(args))
    state = json.loads((args.run_root / 'queue_status.json').read_text())
    assert len(calls) == 1
    assert state['status'] == state['jobs'][0]['status'] == 'failed'
    assert state['jobs'][1]['status'] == 'pending'


def test_resume_skips_verified_results_but_not_tampered_metrics(args, monkeypatch):
    calls = fake_children(monkeypatch)
    plan = launcher.build_plan(args)
    state = launcher.supervise(args, plan)
    launcher.supervise(args, plan)
    assert len(calls) == 3
    metrics_path = Path(state['jobs'][-1]['metrics_path'])
    changed = json.loads(metrics_path.read_text())
    changed['test']['auc'] = .7
    launcher.atomic_json(changed, metrics_path)
    launcher.supervise(args, plan)
    assert len(calls) == 4 and calls[-1]['job']['data'] == 'shuffled'


def test_frozen_data_changes_are_rejected(args, monkeypatch):
    plan = launcher.build_plan(args)
    (args.data_dir / 'X_train_ordered.csv').write_text('changed\n')
    calls = fake_children(monkeypatch)
    with pytest.raises(RuntimeError, match='frozen input changed'):
        launcher.supervise(args, plan)
    assert not calls


def test_local_queue_lock_rejects_duplicate_supervisor(args):
    args.run_root.mkdir()
    with (args.run_root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='already running'):
            launcher.supervise(args, launcher.build_plan(args))


def test_background_launcher_detaches_and_confirms_durable_waiting(args, monkeypatch, capsys):
    seen = {}

    class Supervisor:
        pid = 98765

        def __init__(self, command, **options):
            seen.update(options)
            assert '--background' not in command
            launcher.atomic_json({'status': 'waiting_for_kip', 'supervisor_pid': self.pid},
                                 args.run_root / 'queue_status.json')

        def poll(self):
            return None

    monkeypatch.setattr(launcher.subprocess, 'Popen', Supervisor)
    launcher.main(launcher.background_command(args)[4:] + ['--background'])
    assert seen['start_new_session'] is True and seen['stdin'] == subprocess.DEVNULL
    assert json.loads(capsys.readouterr().out)['status'] == 'waiting_for_kip'


def test_smoke_test_evaluation_or_nan_results_are_rejected(args):
    plan = launcher.build_plan(args)
    job = plan['jobs'][0]
    path = args.run_root / 'metrics.json'
    result = metrics(job)
    result['test'] = {'auc': .7}
    launcher.atomic_json(result, path)
    with pytest.raises(ValueError, match='must not evaluate test'):
        launcher.validated_metrics(path, job, plan)
    job = plan['jobs'][1]
    result = metrics(job)
    result['test']['auc'] = float('nan')
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='nonfinite'):
        launcher.validated_metrics(path, job, plan)
