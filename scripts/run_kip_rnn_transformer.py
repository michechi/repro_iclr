#!/usr/bin/env python3
"""Detached, resumable six-fit RNNTransformer queue on the canonical KIP splits."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/kip_rnn_transformer_20260924'
DATA = ROOT / 'data/simulation/kip_ordered_recovered'
GPU_LOCK = ROOT / 'results/.kip_data_efficiency_gpu.lock'
SOURCES = ('scripts/run_kip_rnn_transformer.py',
           'src/experiments/kip_rnn_transformer_training.py',
           'src/experiments/DL_TR_baselines_experiment.py')
SEEDS = (9550, 9551, 9552)
TAGS = ('kip_m4', 'kip_m6')
METRICS = ('auc', 'f1', 'precision', 'recall', 'accuracy')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def environment(root):
    env = dict(os.environ)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env['CUDA_CACHE_PATH'] = str(root / 'kernel_cache/cuda')
    return env


def command(output, tag, seed, *, smoke=False, dry=False, device='cuda', resume=False):
    cmd = [sys.executable, '-B', '-u', '-m', 'src.experiments.kip_rnn_transformer_training',
           '--tag', tag, '--seed', str(seed), '--output-dir', str(output),
           '--data-dir', str(DATA), '--device', device, '--threads', '4',
           '--batch-size', '64', '--patience', '5', '--epochs', '1' if smoke else '30',
           '--gpu-memory-mib', '2048']
    if smoke:
        cmd.append('--smoke')
    if dry:
        cmd.append('--dry-run')
    if resume:
        cmd.append('--resume')
    return cmd


def plan_digest(plan):
    content = {k: v for k, v in plan.items() if k != 'plan_sha256'}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'launch_plan.json').exists():
        raise FileExistsError('immutable launch plan already exists')
    if shutil.disk_usage(root).free < 400 * 1024**2:
        raise OSError('less than 400 MiB disk headroom')
    for tag in TAGS:
        for kind in ('dry_run', 'cpu_smoke'):
            output = root / 'preflight' / tag / kind
            if output.exists():
                raise FileExistsError('preflight already exists: ' + str(output))
            log = root / 'preflight' / f'{tag}_{kind}.log'
            log.parent.mkdir(exist_ok=True)
            with log.open('w') as stream:
                subprocess.run(command(output, tag, SEEDS[0], smoke=kind == 'cpu_smoke',
                                       dry=kind == 'dry_run', device='cpu'), cwd=ROOT,
                               env=environment(root), stdout=stream, stderr=subprocess.STDOUT,
                               check=True)
            if kind == 'cpu_smoke':
                validate_metrics(read(output / 'metrics.json'), tag, SEEDS[0], smoke=True)
    jobs = []
    for tag in TAGS:
        jobs.append(dict(key=f'{tag}_gpu_smoke', tag=tag, seed=SEEDS[0], smoke=True,
                         output_dir=str(root / 'gpu_smoke' / tag)))
    for tag in TAGS:
        for seed in SEEDS:
            jobs.append(dict(key=f'{tag}_seed{seed}', tag=tag, seed=seed, smoke=False,
                             output_dir=str(root / 'runs' / tag / f'seed{seed}')))
    paths = [DATA / f'{prefix}_{split}_{tag}.csv'
             for tag in TAGS for split in ('train', 'val', 'test') for prefix in ('X', 'y')]
    paths += [DATA / f'{tag}_rule.json' for tag in TAGS]
    plan = dict(schema_version=1, created_at_unix=time.time(), run_root=str(root),
                model='RNNTransformer', architecture='code-optimal BiLSTM + Transformer',
                trainable_parameters=82498, seeds=list(SEEDS), tags=list(TAGS),
                rows=dict(train=400000, val=50000, test=50000),
                recipe=dict(optimizer='Adam', lr=.001, batch_size=64, max_epochs=30,
                            patience=5, selection='strict validation F1 improvement',
                            threshold='argmax', precision='FP32', token_ids='A=1,...,Z=26; PAD=0'),
                test_informed_stopping=False, source_sha256={s: digest(ROOT / s) for s in SOURCES},
                data_sha256={str(p): digest(p) for p in paths}, jobs=jobs,
                preflight_sha256={str(p): digest(p) for p in (root / 'preflight').rglob('*.json')})
    plan['plan_sha256'] = plan_digest(plan)
    write(root / 'launch_plan.json', plan)
    write(root / 'queue_status.json', dict(status='prepared', plan_sha256=plan['plan_sha256'],
          jobs=[dict(j, status='pending') for j in jobs], supervisor_pid=None, child_pid=None,
          current_job=None, updated_at_unix=time.time()))
    print(json.dumps(dict(status='prepared', root=str(root), fits=6, gpu_smokes=2)), flush=True)


def verify_plan(root, data=False):
    plan = read(root / 'launch_plan.json')
    if plan_digest(plan) != plan['plan_sha256'] or plan['run_root'] != str(root):
        raise ValueError('launch plan changed or moved')
    for rel, expected in plan['source_sha256'].items():
        if digest(ROOT / rel) != expected:
            raise ValueError('source changed: ' + rel)
    if data:
        for path, expected in {**plan['data_sha256'], **plan['preflight_sha256']}.items():
            if digest(path) != expected:
                raise ValueError('data or preflight changed: ' + path)
    return plan


def validate_metrics(metrics, tag, seed, smoke=False):
    if metrics.get('status') != 'completed' or metrics.get('tag') != tag or metrics.get('seed') != seed:
        raise ValueError('missing completed metrics or wrong task/seed')
    if metrics.get('is_smoke') is not smoke:
        raise ValueError('smoke/production mismatch')
    if not smoke and metrics.get('rows') != dict(train=400000, val=50000, test=50000):
        raise ValueError('production split counts differ from planned 400K/50K/50K')
    for name in METRICS:
        value = metrics['test'][name]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError('invalid test metric: ' + name)
    if metrics.get('test_informed_stopping') is not False:
        raise ValueError('unexpected test-informed stopping')


def write_csv(path, rows, fields):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    temp = path.with_name('.' + path.name + '.tmp')
    temp.write_text(buffer.getvalue())
    temp.replace(path)


def summarize(root, state):
    rows = []
    for job in state['jobs']:
        if job['smoke'] or job['status'] != 'completed':
            continue
        path = Path(job['output_dir']) / 'metrics.json'
        metrics = read(path)
        validate_metrics(metrics, job['tag'], job['seed'])
        if digest(path) != job['metrics_sha256']:
            raise ValueError('completed metrics changed')
        rows.append(dict(tag=job['tag'], seed=job['seed'], train_rows=400000,
                         **{k: metrics['test'][k] for k in METRICS}, metrics_path=str(path)))
    write_csv(root / 'per_seed.csv', rows, ['tag', 'seed', 'train_rows', *METRICS, 'metrics_path'])
    summary = []
    for tag in TAGS:
        group = [r for r in rows if r['tag'] == tag]
        if sorted(r['seed'] for r in group) != list(SEEDS):
            continue
        row = dict(tag=tag, n=3, train_rows=400000)
        for key in METRICS:
            values = [r[key] for r in group]
            row[key + '_mean'] = statistics.mean(values)
            row[key + '_sample_sd'] = statistics.stdev(values)
        summary.append(row)
    write_csv(root / 'summary.csv', summary, ['tag', 'n', 'train_rows',
              *[k + suffix for k in METRICS for suffix in ('_mean', '_sample_sd')]])
    lines = ['% RNNTransformer, 400K; three training seeds, sample SD.',
             '% Columns: task, AUC, F1, precision, recall.']
    for row in summary:
        cells = [f"{row[k + '_mean']:.3f} ({row[k + '_sample_sd']:.3f})" for k in METRICS[:4]]
        lines.append('RNNTransformer (' + row['tag'].replace('_', '-') + ') & ' +
                     ' & '.join(cells) + r' \\')
    (root / 'table.tex').write_text('\n'.join(lines) + '\n')


def supervise(root):
    with (root / 'queue.lock').open('a+') as local:
        fcntl.flock(local.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify_plan(root, data=True)
        state = read(root / 'queue_status.json')
        if state['plan_sha256'] != plan['plan_sha256']:
            raise ValueError('queue does not match plan')
        expected = [{k: j[k] for k in ('key', 'tag', 'seed', 'smoke', 'output_dir')} for j in state['jobs']]
        if expected != plan['jobs']:
            raise ValueError('queue jobs changed')
        if state['status'] == 'completed':
            raise RuntimeError('queue already completed')
        state.update(supervisor_pid=os.getpid(), child_pid=None, current_job=None, error=None)

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
                if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                            '--format=csv,noheader'], text=True).strip():
                    raise RuntimeError('GPU has an unexpected process outside the shared scheduler')
                verify_plan(root, data=True)
                for job in state['jobs']:
                    if job['status'] == 'completed':
                        path = Path(job['output_dir']) / 'metrics.json'
                        validate_metrics(read(path), job['tag'], job['seed'], job['smoke'])
                        if digest(path) != job['metrics_sha256']:
                            raise ValueError('completed job metrics changed')
                        continue
                    verify_plan(root)
                    if shutil.disk_usage(root).free < 400 * 1024**2:
                        raise OSError('less than 400 MiB disk headroom; preserving existing artifacts')
                    output = Path(job['output_dir'])
                    path = output / 'metrics.json'
                    if path.exists():
                        validate_metrics(read(path), job['tag'], job['seed'], job['smoke'])
                        job.update(status='completed', metrics_sha256=digest(path))
                        summarize(root, state)
                        progress()
                        continue
                    cmd = command(output, job['tag'], job['seed'], smoke=job['smoke'],
                                  resume=(output / 'manifest.json').exists())
                    log = root / 'logs' / (job['key'] + '.log')
                    log.parent.mkdir(exist_ok=True)
                    job.update(status='running', command=cmd, log=str(log), started_at_unix=time.time())
                    with log.open('a') as stream:
                        child = subprocess.Popen(cmd, cwd=ROOT, env=environment(root), stdin=subprocess.DEVNULL,
                            stdout=stream, stderr=subprocess.STDOUT, pass_fds=(local.fileno(), gpu.fileno()))
                        progress('running', current_job=job['key'], child_pid=child.pid)
                        while child.poll() is None:
                            time.sleep(10)
                            progress()
                    job.update(exit_code=child.returncode, finished_at_unix=time.time())
                    progress(child_pid=None)
                    if child.returncode != 0 or not path.exists():
                        job['status'] = 'failed'
                        raise RuntimeError('job failed: ' + job['key'] + '; see ' + str(log))
                    validate_metrics(read(path), job['tag'], job['seed'], job['smoke'])
                    job.update(status='completed', metrics_sha256=digest(path))
                    summarize(root, state)
                    progress()
                summarize(root, state)
            progress('completed', current_job=None, child_pid=None, finished_at_unix=time.time())
        except BaseException as exc:
            progress('failed', error=f'{type(exc).__name__}: {exc}')
            raise


def background(root):
    verify_plan(root)
    with (root / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(root / 'queue_status.json')['status'] == 'completed':
            raise RuntimeError('queue already completed')
    with (root / 'supervisor.log').open('a') as stream:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                  '--run-root', str(root)], cwd=ROOT, env=environment(root),
                                 stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    (root / 'supervisor.pid').write_text(str(child.pid) + '\n')
    for _ in range(150):
        time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError('detached supervisor exited; inspect supervisor.log')
        state = read(root / 'queue_status.json')
        if state.get('supervisor_pid') == child.pid and state['status'] in {'waiting_for_gpu', 'running'}:
            print(json.dumps(dict(status=state['status'], supervisor_pid=child.pid, root=str(root))), flush=True)
            return
    raise RuntimeError('supervisor launch acknowledgement timed out; inspect status before retrying')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--background', action='store_true')
    mode.add_argument('--status', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.prepare:
        prepare(root)
    elif args.background:
        background(root)
    elif args.status:
        print(json.dumps(read(root / 'queue_status.json'), indent=2))
    else:
        supervise(root)


if __name__ == '__main__':
    main()
