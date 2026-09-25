#!/usr/bin/env python3
"""Detached Llama 3.1 8B LoRA fits and attention audits on OC and paired CKD."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/llama8_oc_ckd_attention_20260924'
GPU_LOCK = ROOT / 'results/.kip_data_efficiency_gpu.lock'
ARM = 'llama8b_lora'
SOURCES = (
    'scripts/run_llama8_oc_ckd_attention.py', 'src/experiments/llama8_oc_ckd.py',
    'src/analysis/llama8_oc_ckd_attention.py', 'docs/llama8_oc_ckd_attention_20260924.md',
    'src/experiments/kip_efficiency_models.py', 'src/experiments/kip_encoder_training.py',
    'src/experiments/kip_llama_training.py', 'src/experiments/kip_data_efficiency.py',
    'src/experiments/qwen38_fewshot_eval.py', 'src/mimic/qwen_ckd_training.py',
    'src/experiments/encoder_oc_ckd.py', 'src/analysis/kip_checkpoint_attention.py',
    'src/analysis/kip_attention_core.py', 'src/analysis/oc_attention_core.py',
    'src/analysis/ckd_attention_audit.py', 'src/analysis/qwen_oc_audit.py',
    'repro/src/analysis/mechanism_id/scripts/common.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def plan_digest(plan):
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != 'plan_sha256'},
                                    sort_keys=True).encode()).hexdigest()


def environment(root):
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               HF_HUB_DISABLE_PROGRESS_BARS='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONPATH=os.pathsep.join((str(ROOT / '.deps/kip_eff'), str(ROOT))))
    for variable, folder in [('MPLCONFIGDIR', 'matplotlib'), ('TRITON_CACHE_DIR', 'triton'),
                             ('CUDA_CACHE_PATH', 'cuda'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor')]:
        env[variable] = str(root / 'kernel_cache' / folder)
    return env


def prepare(root):
    if (root / 'launch_plan.json').exists():
        raise FileExistsError('an immutable launch plan already exists')
    audit_path = root / 'preflight_cpu.json'
    audit = read(audit_path)
    if audit.get('status') != 'passed':
        raise ValueError('CPU preflight has not passed')
    if audit['model']['name'] != 'meta-llama/Llama-3.1-8B':
        raise ValueError('unexpected 8B model identity')
    if not audit['paired_ckd'] or not audit['oc_indices']['computed_matches_reference']:
        raise ValueError('paired CKD / matched OC training subsets were not verified')
    models = {ARM: audit['model']}
    jobs = []

    def pair(task, condition, smoke=False):
        key = ('smoke_' if smoke else '') + f'llama8_{task}_{condition}'
        jobs.append(dict(key=key + '_fit', kind='fit', model=ARM, task=task, condition=condition,
                         smoke=smoke, status='pending', output_dir=str(root / 'fits' / key / 'attempt_001')))
        jobs.append(dict(key=key + '_attention', kind='attention', model=ARM, task=task, condition=condition,
                         smoke=smoke, status='pending', fit_job=key + '_fit',
                         output_dir=str(root / 'attention' / key)))

    pair('oc', 'ordered', True)
    pair('ckd', 'ordered', True)
    pair('oc', 'ordered')
    pair('ckd', 'ordered')
    pair('ckd', 'shuffled')
    jobs.append(dict(key='ckd_paired_comparison', kind='summary', smoke=False, status='pending',
                     output_dir=str(root / 'ckd_paired_comparison')))
    inputs = {str(audit_path): digest(audit_path)}
    for task in audit['datasets'].values():
        for condition in task.values():
            for split in condition.values():
                for key in ('features', 'labels'):
                    inputs[split[key]] = split[key + '_sha256']
    reference = audit['oc_indices']
    inputs[reference['reference_path']] = reference['reference_sha256']
    plan = dict(schema_version=1, task='llama8_lora_oc_ckd_attention', arm=ARM, models=models,
        model_name_resolution=dict(requested='Llama 3.2-8B LoRA', resolved='meta-llama/Llama-3.1-8B',
            basis='existing 8B KIP checkpoint; clarification requested and default stated'),
        datasets=audit['datasets'], jobs=jobs,
        recipes=dict(
            oc=dict(seed=9950, train_rows=200000, epochs=20, min_epochs=1, patience=3, lr=2e-5,
                    batch_size=128, effective_batch_size=256, eval_batch_size=128, max_length=64,
                    reference_train_indices=dict(path=reference['reference_path'], sha256=reference['reference_sha256'])),
            ckd=dict(seed=42, epochs=30, min_epochs=5, patience=5, lr=2e-5, batch_size=4,
                     effective_batch_size=32, eval_batch_size=4, max_length=16384)),
        tuning='lora', quantization=None, lora=dict(r=8, alpha=16, dropout=.1,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']),
        samples=1000, sample_seed=9950, perturbations=200, smoke_samples=16, smoke_perturbations=4,
        source_sha256={name: digest(ROOT / name) for name in SOURCES}, input_sha256=inputs,
        preflight_sha256=digest(audit_path), checkpoint_policy='retain selected adapter/head for every fit',
        ckd_context_policy='full history without truncation; last unmasked token readout',
        test_informed_stopping=False, fresh_pretrained_initialization=True,
        schedule='OC and CKD GPU smoke fits/audits -> OC fit/audit -> CKD ordered fit/audit -> CKD shuffled fit/audit -> paired comparison',
        created_at_unix=time.time())
    plan['plan_sha256'] = plan_digest(plan)
    verify(plan, models=True)
    write(root / 'launch_plan.json', plan)
    write(root / 'queue_status.json', dict(status='prepared', jobs=jobs, plan_sha256=plan['plan_sha256'],
          supervisor_pid=None, child_pid=None, current_job=None, validated_batches={}, updated_at_unix=time.time()))
    print(json.dumps(dict(status='prepared', jobs=len(jobs), plan_sha256=plan['plan_sha256'])), flush=True)


def verify(plan, models=False):
    if plan_digest(plan) != plan['plan_sha256']:
        raise ValueError('launch plan content hash mismatch')
    for relative, expected in plan['source_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('frozen source changed: ' + relative)
    for filename, expected in plan['input_sha256'].items():
        if digest(filename) != expected:
            raise ValueError('frozen input changed: ' + filename)
    if models:
        for model in plan['models'].values():
            for filename, expected in model['files_sha256'].items():
                if digest(Path(model['path']) / filename) != expected:
                    raise ValueError('pinned base model changed: ' + filename)


def command_for(root, job, batch=None, attempt=1):
    if job['kind'] == 'fit':
        output = Path(job['output_dir']).parent / f'attempt_{attempt:03d}'
        command = [sys.executable, '-B', '-u', '-m', 'src.experiments.llama8_oc_ckd',
                   '--run-root', str(root), '--job', job['key'], '--output-dir', str(output)]
        if batch is not None:
            command += ['--batch-size', str(batch)]
    else:
        output = Path(job['output_dir'])
        command = [sys.executable, '-B', '-u', '-m', 'src.analysis.llama8_oc_ckd_attention', '--run-root', str(root)]
        command += ['--paired-summary'] if job['kind'] == 'summary' else ['--job', job['key']]
    return command, output


def validate_state(plan, state):
    if state.get('plan_sha256') != plan['plan_sha256']:
        raise ValueError('queue state references a different launch plan')
    jobs = state.get('jobs', [])
    if not jobs or [j.get('key') for j in jobs] != [j['key'] for j in plan['jobs']]:
        raise ValueError('queue jobs differ from the immutable launch plan')
    for job, original in zip(jobs, plan['jobs']):
        for field in ('kind', 'model', 'task', 'condition', 'smoke', 'fit_job'):
            if job.get(field) != original.get(field):
                raise ValueError('queue job identity changed: ' + job['key'])
        if job.get('status') not in {'pending', 'running', 'failed', 'completed'}:
            raise ValueError('invalid queue job status')
        output, expected = Path(job['output_dir']), Path(original['output_dir'])
        if job['kind'] == 'fit':
            if output.parent != expected.parent or not re.fullmatch(r'attempt_[0-9]{3,}', output.name):
                raise ValueError('fit output path differs from its planned attempt directory')
        elif output != expected:
            raise ValueError('analysis output path differs from the plan')
    for task, batch in state.get('validated_batches', {}).items():
        if task not in plan['recipes']:
            raise ValueError('validated batch has an unknown task')
        recipe = plan['recipes'][task]
        if (type(batch) is not int or batch < 1 or batch > recipe['batch_size']
                or recipe['effective_batch_size'] % batch):
            raise ValueError('validated batch changes the planned effective batch')


def supervise(root):
    with (root / 'queue.lock').open('a+') as local:
        fcntl.flock(local.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root / 'launch_plan.json')
        verify(plan)
        state = read(root / 'queue_status.json')
        validate_state(plan, state)
        if state.get('status') == 'completed':
            raise RuntimeError('queue already completed')
        state.update(supervisor_pid=os.getpid(), child_pid=None, current_job=None, error=None)
        state.setdefault('validated_batches', {})

        def progress(status=None, **details):
            if status:
                state['status'] = status
            state.update(details, updated_at_unix=time.time())
            write(root / 'queue_status.json', state)

        try:
            with GPU_LOCK.open('a+') as gpu:
                progress('waiting_for_gpu')
                while True:
                    try:
                        fcntl.flock(gpu.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(10)
                        progress()
                if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
                    raise RuntimeError('unexpected GPU process; refusing overlap')
                verify(plan, models=True)
                for job in state['jobs']:
                    if job.get('status') == 'completed':
                        metrics = Path(job['output_dir']) / 'metrics.json'
                        if read(metrics).get('status') != 'completed' or digest(metrics) != job['metrics_sha256']:
                            raise ValueError('completed job metrics changed')
                        continue
                    verify(plan)
                    original = next(j for j in plan['jobs'] if j['key'] == job['key'])
                    batch = state['validated_batches'].get(job.get('task'), plan['recipes'].get(job.get('task'), {}).get('batch_size'))
                    attempt = len(job.get('attempts', [])) + 1
                    if job['kind'] != 'fit' and Path(job['output_dir']).exists():
                        previous = Path(job['output_dir'])
                        if (previous / 'metrics.json').exists() and read(previous / 'metrics.json').get('status') == 'completed':
                            job.update(status='completed', metrics_sha256=digest(previous / 'metrics.json'))
                            progress()
                            continue
                        archive = root / 'failed_attention_attempts' / f"{job['key']}_{attempt-1:03d}"
                        archive.parent.mkdir(exist_ok=True)
                        if archive.exists():
                            raise FileExistsError('partial analysis archive already exists')
                        previous.rename(archive)
                    while True:
                        command, output = command_for(root, original, batch, attempt)
                        if job['kind'] == 'fit' and output.exists() and any(output.iterdir()):
                            raise FileExistsError('fresh fit output already exists')
                        if shutil.disk_usage(root).free < 300 * 1024**2:
                            raise OSError('less than 300 MiB output headroom; preserving all existing artifacts')
                        log = root / 'logs' / f"{job['key']}_attempt_{attempt:03d}.log"
                        log.parent.mkdir(exist_ok=True)
                        job.update(status='running', output_dir=str(output))
                        job.setdefault('attempts', []).append(dict(command=command, output_dir=str(output), log=str(log),
                                                                  batch_size=batch, started_at_unix=time.time()))
                        with log.open('a') as stream:
                            child = subprocess.Popen(command, cwd=ROOT, env=environment(root), stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, pass_fds=(local.fileno(), gpu.fileno()))
                            progress('running', current_job=job['key'], child_pid=child.pid)
                            while child.poll() is None:
                                time.sleep(10)
                                progress()
                        job['attempts'][-1].update(exit_code=child.returncode, finished_at_unix=time.time())
                        progress(child_pid=None)
                        metrics = output / 'metrics.json'
                        if child.returncode == 0 and metrics.exists() and read(metrics).get('status') == 'completed':
                            job.update(status='completed', metrics_sha256=digest(metrics))
                            if job['kind'] == 'fit':
                                state['validated_batches'][job['task']] = batch
                            progress()
                            break
                        if job['kind'] == 'fit' and batch > 1 and 'out of memory' in log.read_text(errors='replace')[-16000:].lower():
                            batch //= 2
                            attempt += 1
                            progress('retrying_oom')
                            continue
                        job['status'] = 'failed'
                        raise RuntimeError('job failed: ' + job['key'] + '; see ' + str(log))
            progress('completed', current_job=None, child_pid=None, finished_at_unix=time.time())
        except BaseException as error:
            progress('failed', error=f'{type(error).__name__}: {error}')
            raise


def background(root):
    plan = read(root / 'launch_plan.json')
    verify(plan)
    with (root / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / 'queue_status.json')
        validate_state(plan, state)
        if state['status'] == 'completed':
            raise RuntimeError('queue already completed')
    with (root / 'supervisor.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--run-root', str(root)],
            cwd=ROOT, env=environment(root), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    (root / 'supervisor.pid').write_text(str(child.pid) + '\n')
    for _ in range(150):
        time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError('detached supervisor exited; inspect supervisor.log')
        state = read(root / 'queue_status.json')
        if state.get('supervisor_pid') == child.pid and state.get('status') in {'waiting_for_gpu', 'running'}:
            print(json.dumps(dict(status=state['status'], supervisor_pid=child.pid, run_root=str(root))), flush=True)
            return
    raise RuntimeError('startup not acknowledged; inspect status before retrying')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=RUN)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--background', action='store_true')
    args = parser.parse_args()
    args.run_root = args.run_root.resolve()
    prepare(args.run_root) if args.prepare else background(args.run_root) if args.background else supervise(args.run_root)
