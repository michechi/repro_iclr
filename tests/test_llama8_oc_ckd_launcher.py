"""CPU checks for detached Llama OC/CKD resource exclusion and recovery."""
import copy
import fcntl
import os
from pathlib import Path
import subprocess

import pytest

from scripts import run_llama8_oc_ckd_attention as launcher


@pytest.fixture
def queue(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'GPU_LOCK', tmp_path / 'gpu.lock')
    monkeypatch.setattr(launcher.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: '')
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 2 * 1024**3})())
    root = tmp_path / 'run'
    root.mkdir()
    source, inputs, weights = tmp_path / 'source.py', tmp_path / 'input.csv', tmp_path / 'weights'
    source.write_text('source\n')
    inputs.write_text('input\n')
    weights.mkdir()
    (weights / 'model.bin').write_bytes(b'pretrained weights')
    jobs = []
    for name, task, smoke, kind in [
        ('smoke_oc_fit', 'oc', True, 'fit'),
        ('smoke_oc_attention', 'oc', True, 'attention'),
        ('oc_fit', 'oc', False, 'fit'),
        ('ckd_fit', 'ckd', False, 'fit'),
        ('ckd_shuffled_fit', 'ckd', False, 'fit'),
    ]:
        job = dict(key=name, task=task, kind=kind, model=launcher.ARM, smoke=smoke,
                   condition='shuffled' if 'shuffled' in name else 'ordered', status='pending',
                   output_dir=str(root / ('fits' if kind == 'fit' else 'attention') / name /
                                  'attempt_001') if kind == 'fit' else str(root / 'attention' / name))
        if kind == 'attention':
            job['fit_job'] = 'smoke_oc_fit'
        jobs.append(job)
    plan = dict(source_sha256={source.name: launcher.digest(source)},
                input_sha256={str(inputs): launcher.digest(inputs)},
                models={launcher.ARM: dict(path=str(weights), files_sha256={'model.bin': launcher.digest(weights / 'model.bin')})},
                recipes=dict(oc=dict(batch_size=8, effective_batch_size=256, eval_batch_size=8),
                             ckd=dict(batch_size=4, effective_batch_size=32, eval_batch_size=4)), jobs=jobs)
    plan['plan_sha256'] = launcher.plan_digest(plan)
    launcher.write(root / 'launch_plan.json', plan)
    launcher.write(root / 'queue_status.json', dict(status='prepared', jobs=copy.deepcopy(jobs),
        plan_sha256=plan['plan_sha256'], supervisor_pid=None, child_pid=None, current_job=None, validated_batches={}))
    return root, plan


def children(monkeypatch, root, behavior=None):
    calls = []

    class Child:
        pid = 76543

        def __init__(self, command, **options):
            assert options['stdin'] == subprocess.DEVNULL
            assert len(options['pass_fds']) == 2
            for descriptor in options['pass_fds']:
                os.fstat(descriptor)
            # Separate file descriptions cannot acquire either inherited lock.
            for path in (root / 'queue.lock', launcher.GPU_LOCK):
                with path.open('a') as another:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(another, fcntl.LOCK_EX | fcntl.LOCK_NB)
            job_key = command[command.index('--job') + 1]
            state = launcher.read(root / 'queue_status.json')
            job = next(j for j in state['jobs'] if j['key'] == job_key)
            output = Path(command[command.index('--output-dir') + 1]) if '--output-dir' in command else Path(job['output_dir'])
            assert not output.exists() or not any(output.iterdir())
            output.mkdir(parents=True, exist_ok=True)
            batch = int(command[command.index('--batch-size') + 1]) if '--batch-size' in command else None
            calls.append(dict(command=command, job=job_key, output=output, batch=batch))
            self.returncode, text = behavior(calls) if behavior else (0, 'completed\n')
            options['stdout'].write(text)
            options['stdout'].flush()
            (output / 'attempt_evidence.txt').write_text(text)
            if self.returncode == 0:
                result = dict(status='completed', job=job_key, task=job['task'], condition=job['condition'],
                              arm=launcher.ARM, model=launcher.ARM, is_smoke=job['smoke'],
                              plan_sha256=state['plan_sha256'])
                if job['kind'] == 'fit':
                    checkpoint = output / 'best_trainable.pt'
                    checkpoint.write_bytes(b'selected adapter/head')
                    result['selected_checkpoint_sha256'] = launcher.digest(checkpoint)
                launcher.write(output / 'metrics.json', result)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(launcher.subprocess, 'Popen', Child)
    return calls


def test_commands_use_fresh_fit_attempt_and_keep_plan_path(queue):
    root, plan = queue
    original = copy.deepcopy(plan['jobs'][0])
    command, output = launcher.command_for(root, original, batch=2, attempt=3)
    assert output.name == 'attempt_003'
    assert command[command.index('--batch-size') + 1] == '2'
    assert command[command.index('--output-dir') + 1] == str(output)
    assert original == plan['jobs'][0]
    attention, _ = launcher.command_for(root, plan['jobs'][1], batch=1)
    assert '--batch-size' not in attention
    assert 'src.analysis.llama8_oc_ckd_attention' in attention
    summary, _ = launcher.command_for(root, dict(key='summary', kind='summary', output_dir=str(root / 'summary')))
    assert '--paired-summary' in summary and '--job' not in summary


def test_plan_hash_excludes_only_its_own_field(queue):
    _, plan = queue
    assert launcher.plan_digest(plan) == launcher.plan_digest({**plan, 'plan_sha256': 'ignored'})
    changed = copy.deepcopy(plan)
    changed['recipes']['oc']['effective_batch_size'] = 128
    assert launcher.plan_digest(changed) != plan['plan_sha256']
    with pytest.raises(ValueError, match='content hash'):
        launcher.verify(changed)


@pytest.mark.parametrize('kind', ['source', 'input', 'model'])
def test_frozen_source_input_and_model_changes_rejected(queue, kind):
    root, plan = queue
    path = {'source': root.parent / 'source.py', 'input': root.parent / 'input.csv',
            'model': root.parent / 'weights/model.bin'}[kind]
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        launcher.verify(plan, models=True)


def test_environment_is_offline_with_local_caches(queue):
    root, _ = queue
    environment = launcher.environment(root)
    assert environment['HF_HUB_OFFLINE'] == environment['TRANSFORMERS_OFFLINE'] == '1'
    assert environment['HF_HUB_DISABLE_IMPLICIT_TOKEN'] == '1'
    assert environment['TOKENIZERS_PARALLELISM'] == 'false'
    assert Path(environment['TRITON_CACHE_DIR']).is_relative_to(root)


def test_duplicate_supervisor_cannot_enter(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            launcher.supervise(root)
    assert not calls


def test_shared_gpu_lock_defers_all_children(queue, monkeypatch):
    root, _ = queue
    blocker = launcher.GPU_LOCK.open('a')
    fcntl.flock(blocker.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    calls = children(monkeypatch, root)
    waits = []

    def release(seconds):
        assert not calls
        assert launcher.read(root / 'queue_status.json')['status'] == 'waiting_for_gpu'
        waits.append(seconds)
        blocker.close()

    monkeypatch.setattr(launcher.time, 'sleep', release)
    launcher.supervise(root)
    assert waits == [10]
    assert len(calls) == 5


def test_external_gpu_process_rejected(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root)
    monkeypatch.setattr(launcher.subprocess, 'check_output', lambda *args, **kwargs: '12345\n')
    with pytest.raises(RuntimeError, match='unexpected GPU process'):
        launcher.supervise(root)
    assert not calls
    assert launcher.read(root / 'queue_status.json')['status'] == 'failed'


def test_smoke_oom_retries_fresh_and_carries_batch_to_production(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root, lambda c: (1, 'CUDA out of memory\n') if len(c) == 1 else (0, 'done\n'))
    launcher.supervise(root)
    assert calls[0]['batch'] == 8 and calls[1]['batch'] == 4
    assert calls[0]['output'].name == 'attempt_001' and calls[1]['output'].name == 'attempt_002'
    assert calls[0]['output'] != calls[1]['output']
    assert (calls[0]['output'] / 'attempt_evidence.txt').exists()
    assert not (calls[0]['output'] / 'metrics.json').exists()
    assert next(c for c in calls if c['job'] == 'oc_fit')['batch'] == 4
    assert next(c for c in calls if c['job'] == 'ckd_fit')['batch'] == 4
    state = launcher.read(root / 'queue_status.json')
    assert state['validated_batches'] == {'oc': 4, 'ckd': 4}
    assert state['status'] == 'completed'
    for call in calls[1:]:
        if '--output-dir' in call['command']:
            assert (call['output'] / 'best_trainable.pt').exists()


def test_production_oom_updates_later_condition_only(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root, lambda c: (1, 'CUDA out of memory\n')
                     if c[-1]['job'] == 'ckd_fit' and c[-1]['batch'] == 4 else (0, 'done\n'))
    launcher.supervise(root)
    assert [c['batch'] for c in calls if c['job'] == 'ckd_fit'] == [4, 2]
    assert next(c for c in calls if c['job'] == 'ckd_shuffled_fit')['batch'] == 2
    assert next(c for c in calls if c['job'] == 'oc_fit')['batch'] == 8


def test_non_oom_failure_stops_later_jobs(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root, lambda c: (1, 'ValueError: invalid data\n'))
    with pytest.raises(RuntimeError, match='job failed'):
        launcher.supervise(root)
    assert len(calls) == 1
    state = launcher.read(root / 'queue_status.json')
    assert state['status'] == state['jobs'][0]['status'] == 'failed'
    assert state['jobs'][1]['status'] == 'pending'


def test_oom_at_batch_one_stops_without_zero_batch(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root, lambda c: (1, 'CUDA out of memory\n'))
    with pytest.raises(RuntimeError, match='job failed'):
        launcher.supervise(root)
    assert [c['batch'] for c in calls] == [8, 4, 2, 1]


def test_attention_failure_preserves_fit_then_resume_archives_partial_analysis(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root, lambda c: (1, 'analysis failed\n')
                     if c[-1]['job'] == 'smoke_oc_attention' else (0, 'done\n'))
    with pytest.raises(RuntimeError, match='job failed'):
        launcher.supervise(root)
    checkpoint = calls[0]['output'] / 'best_trainable.pt'
    original_hash = launcher.digest(checkpoint)
    retry = children(monkeypatch, root)
    launcher.supervise(root)
    assert retry[0]['job'] == 'smoke_oc_attention'
    archive = root / 'failed_attention_attempts/smoke_oc_attention_001/attempt_evidence.txt'
    assert archive.read_text() == 'analysis failed\n'
    assert launcher.digest(checkpoint) == original_hash


def test_resume_refuses_modified_completed_metrics(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root, lambda c: (1, 'analysis failed\n')
                     if c[-1]['job'] == 'smoke_oc_attention' else (0, 'done\n'))
    with pytest.raises(RuntimeError):
        launcher.supervise(root)
    metrics = calls[0]['output'] / 'metrics.json'
    data = launcher.read(metrics)
    data['modified'] = True
    launcher.write(metrics, data)
    retry = children(monkeypatch, root)
    with pytest.raises(ValueError, match='metrics changed'):
        launcher.supervise(root)
    assert not retry


def test_low_disk_preserves_artifacts_and_starts_no_child(queue, monkeypatch):
    root, _ = queue
    calls = children(monkeypatch, root)
    monkeypatch.setattr(launcher.shutil, 'disk_usage', lambda path: type('Usage', (), {'free': 299 * 1024**2})())
    with pytest.raises(OSError, match='headroom'):
        launcher.supervise(root)
    assert not calls


@pytest.mark.parametrize('mutation', ['wrong_plan_hash', 'missing_jobs', 'changed_identity'])
def test_resume_rejects_state_disagreement_with_frozen_plan(queue, monkeypatch, mutation):
    root, _ = queue
    state = launcher.read(root / 'queue_status.json')
    if mutation == 'wrong_plan_hash':
        state['plan_sha256'] = '0' * 64
    elif mutation == 'missing_jobs':
        state['jobs'] = []
    else:
        state['jobs'][0]['task'] = 'ckd'
    launcher.write(root / 'queue_status.json', state)
    calls = children(monkeypatch, root)
    with pytest.raises((ValueError, RuntimeError)):
        launcher.supervise(root)
    assert not calls


def test_prepare_rejects_existing_plan(queue):
    root, _ = queue
    before = launcher.digest(root / 'launch_plan.json')
    with pytest.raises(FileExistsError):
        launcher.prepare(root)
    assert launcher.digest(root / 'launch_plan.json') == before
