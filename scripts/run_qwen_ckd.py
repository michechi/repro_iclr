#!/usr/bin/env python3
"""Detached Qwen CKD queue that waits for successful completion of KIP."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_PATH = ROOT / Path('data/model_cache/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
SOURCE_FILES = (
    'scripts/run_qwen_ckd.py', 'scripts/run_kip_data_efficiency.py',
    'docs/qwen38_ckd_20260921.md',
    'scripts/audit_kip_efficiency_weights.py', 'src/mimic/qwen_ckd_training.py',
    'src/experiments/kip_data_efficiency.py', 'src/experiments/kip_efficiency_models.py',
    'src/experiments/kip_llama_training.py', 'src/experiments/kip_encoder_training.py',
    'src/experiments/kip_cpu_offload.py', 'src/experiments/qwen38_fewshot_eval.py',
    'src/experiments/qwen_fewshot_eval.py',
)
DATA_FILES = tuple(f'{kind}_{split}_{mode}.csv' for mode in ('ordered', 'shuffled')
                   for split in ('train', 'val', 'test') for kind in ('X', 'y'))


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'results/qwen38_27b_ckd_20260921_seed42')
    parser.add_argument('--model-path', type=Path, default=MODEL_PATH)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/processed/mimic_training')
    parser.add_argument('--prerequisite-status', type=Path,
                        default=ROOT / 'results/kip_m4_data_efficiency_20260920_seed9950/queue_status.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, choices=(1, 2), default=2)
    parser.add_argument('--eval-batch-size', type=int, choices=(1, 2), default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=16384)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for name in ('epochs', 'patience', 'min_epochs', 'max_length', 'threads', 'poll_seconds', 'lr'):
        if getattr(args, name) <= 0 or not math.isfinite(getattr(args, name)):
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if args.min_epochs > args.epochs:
        parser.error('--min-epochs cannot exceed --epochs')
    for name in ('run_root', 'model_path', 'data_dir', 'prerequisite_status'):
        setattr(args, name, getattr(args, name).resolve())
    return args


def environment(args):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED=str(args.seed),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_PROGRESS_BARS='1',
               TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT / '.deps/kip_eff'), str(ROOT)))
    for variable, name in (('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                           ('CUDA_CACHE_PATH', 'cuda')):
        env[variable] = str(args.run_root / 'kernel_cache' / name)
    return env


def training_command(args, job, output, batch, eval_batch):
    command = [sys.executable, '-B', '-u', '-m', 'src.mimic.qwen_ckd_training',
               '--model-path', str(args.model_path), '--data-dir', str(args.data_dir),
               '--data', job['data'], '--output-dir', str(output), '--seed', str(args.seed),
               '--batch-size', str(batch), '--gradient-accumulation-steps', str(32 // batch),
               '--eval-batch-size', str(eval_batch), '--epochs', str(args.epochs),
               '--patience', str(args.patience), '--min-epochs', str(args.min_epochs),
               '--lr', str(args.lr), '--max-length', str(args.max_length), '--threads', str(args.threads)]
    if job['is_smoke']:
        command.append('--smoke')
    return command


def build_plan(args):
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES if (ROOT / name).is_file()}
    data = {name: sha256(args.data_dir / name) for name in DATA_FILES if (args.data_dir / name).is_file()}
    metadata_names = ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json')
    metadata = {name: sha256(args.model_path / name) for name in metadata_names
                if (args.model_path / name).is_file()}
    jobs = [{'key': 'smoke_ordered', 'data': 'ordered', 'is_smoke': True},
            {'key': 'ordered', 'data': 'ordered', 'is_smoke': False},
            {'key': 'shuffled', 'data': 'shuffled', 'is_smoke': False}]
    for job in jobs:
        job['command'] = training_command(args, job, args.run_root / job['key'] / 'attempt_001',
                                          args.batch_size, args.eval_batch_size)
    rows = {}
    for split in ('train', 'val', 'test'):
        path = args.data_dir / f'y_{split}_ordered.csv'
        if path.is_file():
            with path.open(newline='') as stream:
                rows[split] = sum(1 for _ in csv.reader(stream)) - 1
    plan = {'schema_version': 1, 'task': 'CKD_to_ESRD', 'model': 'Qwen/Qwen3.8-27B',
            'method': 'QLoRA', 'seed': args.seed, 'python': sys.executable,
            'run_root': str(args.run_root), 'model_path': str(args.model_path),
            'data_dir': str(args.data_dir), 'prerequisite_status': str(args.prerequisite_status),
            'required_prerequisite_status': 'completed', 'minimum_free_gpu_gib': 120,
            'effective_batch_size': 32, 'epochs': args.epochs, 'patience': args.patience,
            'min_epochs': args.min_epochs, 'learning_rate': args.lr, 'max_length': args.max_length,
            'selection': 'validation_loss', 'production_rows': rows,
            'source_sha256': sources, 'data_sha256': data, 'model_metadata_sha256': metadata,
            'missing_files': ([str(ROOT / name) for name in SOURCE_FILES if name not in sources]
                              + [str(args.data_dir / name) for name in DATA_FILES if name not in data]
                              + [str(args.model_path / name) for name in metadata_names if name not in metadata]),
            'oom_policy': 'retry in a fresh attempt with batch 1, accumulation 32, evaluation batch 1',
            'smoke_policy': 'ordered longest training sequences; no test evaluation', 'jobs': jobs}
    plan['plan_sha256'] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    return plan


def freeze_plan(args, plan):
    if plan['missing_files']:
        raise RuntimeError(f'missing required inputs: {plan["missing_files"]}')
    path = args.run_root / 'launch_plan.json'
    if path.exists():
        if json.loads(path.read_text()).get('plan_sha256') != plan['plan_sha256']:
            raise RuntimeError('existing run-root has a different protocol; use a new run-root')
        return
    for name in plan['source_sha256']:
        target = args.run_root / 'protocol' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    atomic_json(plan, path)


def verify_inputs(args, plan):
    for base, expected in ((ROOT, plan['source_sha256']), (args.data_dir, plan['data_sha256']),
                            (args.model_path, plan['model_metadata_sha256'])):
        for name, digest in expected.items():
            if not (base / name).is_file() or sha256(base / name) != digest:
                raise RuntimeError(f'frozen input changed: {base / name}')


def prerequisite_ready(path):
    try:
        state = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        return False, f'prerequisite status unavailable: {error}'
    status = state.get('status')
    if status != 'completed':
        return False, f'KIP status is {status!r}; waiting for explicit successful completion'
    if state.get('child_pid'):
        return False, 'KIP still reports an active child process'
    return True, 'KIP completed successfully'


def gpu_ready():
    try:
        memory = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free',
                                 '--format=csv,noheader,nounits'], text=True, capture_output=True,
                                timeout=15, check=True)
        processes = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], text=True, capture_output=True,
                                   timeout=15, check=True)
        free_mib = int(memory.stdout.strip())
        if processes.stdout.strip():
            return False, 'GPU compute processes remain: ' + processes.stdout.strip()
        if free_mib < 120 * 1024:
            return False, f'GPU has only {free_mib} MiB free; require at least 122880 MiB'
        return True, f'GPU has {free_mib} MiB free and no competing compute processes'
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return False, f'GPU readiness unavailable: {error}'


def publish(args, state):
    state['updated_at_unix'] = time.time()
    atomic_json(state, args.run_root / 'queue_status.json')


def wait_for_gpu(args, state, gpu_lock):
    """Keep only our queue lock while KIP owns the shared GPU lock."""
    last_message = None
    while True:
        ready, message = prerequisite_ready(args.prerequisite_status)
        phase = 'waiting_for_kip'
        if ready:
            phase = 'waiting_for_gpu'
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                ready, message = False, 'shared GPU lock is held by another queue or surviving trainer'
            else:
                ready, message = gpu_ready()
                if ready:
                    # Close the gap between reading status and acquiring the GPU lock.
                    ready, message = prerequisite_ready(args.prerequisite_status)
                if ready:
                    state.update(status='running', wait_reason=None, gpu_acquired_at_unix=time.time())
                    publish(args, state)
                    return
                fcntl.flock(gpu_lock, fcntl.LOCK_UN)
        state.update(status=phase, wait_reason=message, child_pid=None)
        publish(args, state)
        if message != last_message:
            print(message, flush=True)
            last_message = message
        time.sleep(args.poll_seconds)


def ensure_model_audit(args, state, lock_fds):
    from scripts.run_kip_data_efficiency import valid_model_audit
    evidence = args.run_root / 'protocol/model_sha256.json'
    if valid_model_audit(args.model_path, evidence):
        return {'status': 'verified', 'path': str(evidence), 'reused': True}
    log_path = args.run_root / 'model_audit.log'
    command = [sys.executable, '-B', '-u', str(ROOT / 'scripts/audit_kip_efficiency_weights.py'),
               '--model-path', str(args.model_path), '--output', str(evidence)]
    state['current_job'] = 'model_payload_audit'
    publish(args, state)
    with log_path.open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, env=environment(args), pass_fds=lock_fds)
        state['child_pid'] = child.pid
        publish(args, state)
        code = child.wait()
    state['child_pid'] = None
    if code or not valid_model_audit(args.model_path, evidence):
        raise RuntimeError(f'checkpoint audit failed (exit {code}); inspect {log_path}')
    return {'status': 'verified', 'path': str(evidence), 'reused': False}


def finite_metric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def verify_model_audit(args):
    from scripts.run_kip_data_efficiency import valid_model_audit
    if not valid_model_audit(args.model_path, args.run_root / 'protocol/model_sha256.json'):
        raise RuntimeError('checkpoint identity changed after its payload audit; refusing to start training')


def validated_metrics(path, job, plan):
    metrics = json.loads(Path(path).read_text())
    for field, expected in {'status': 'completed', 'data': job['data'], 'seed': plan['seed'],
                             'is_smoke': job['is_smoke']}.items():
        if metrics.get(field) != expected:
            raise ValueError(f'metrics {field} mismatch: {metrics.get(field)!r} != {expected!r}')
    if job['is_smoke']:
        if not finite_metric(metrics.get('validation', {}).get('auc')):
            raise ValueError('smoke validation AUC is missing or nonfinite')
        if metrics.get('test'):
            raise ValueError('smoke must not evaluate test data')
    else:
        if metrics.get('rows') != plan['production_rows']:
            raise ValueError('production metrics have incorrect split counts')
        for name in ('auc', 'auprc', 'f1'):
            if not finite_metric(metrics.get('test', {}).get(name)):
                raise ValueError(f'production test metric {name} is missing or nonfinite')
    return metrics


def verified_success(job, plan):
    if job.get('status') != 'completed' or not job.get('metrics_path'):
        return False
    try:
        validated_metrics(job['metrics_path'], job, plan)
        return sha256(job['metrics_path']) == job.get('metrics_sha256')
    except (OSError, ValueError, TypeError):
        return False


def is_oom(log_path):
    with Path(log_path).open('rb') as stream:
        stream.seek(max(0, stream.seek(0, os.SEEK_END) - 256 * 1024))
        tail = stream.read().decode(errors='replace').lower()
    return any(marker in tail for marker in ('cuda out of memory', 'cuda error: out of memory',
                                              'torch.outofmemoryerror', 'torch.cuda.outofmemoryerror'))


def run_job(args, plan, state, job, batch, eval_batch, lock_fds):
    while True:
        verify_inputs(args, plan)
        verify_model_audit(args)
        if shutil.disk_usage(args.run_root).free < 512 * 1024**2:
            raise RuntimeError('less than 512 MiB disk space remains; refusing to start a new attempt')
        job_dir = args.run_root / job['key']
        job_dir.mkdir(parents=True, exist_ok=True)
        index = 1
        while (job_dir / f'attempt_{index:03d}').exists():
            index += 1
        output = job_dir / f'attempt_{index:03d}'
        output.mkdir()
        log_path = job_dir / f'attempt_{index:03d}.log'
        command = training_command(args, job, output, batch, eval_batch)
        attempt = {'output_dir': str(output), 'log': str(log_path), 'command': command,
                   'batch_size': batch, 'gradient_accumulation_steps': 32 // batch,
                   'eval_batch_size': eval_batch, 'started_at_unix': time.time(), 'status': 'starting'}
        job.setdefault('attempts', []).append(attempt)
        job.update(status='running', error=None)
        state.update(status='running', current_job=job['key'])
        publish(args, state)
        try:
            with log_path.open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, env=environment(args), pass_fds=lock_fds)
                state['child_pid'] = child.pid
                attempt.update(status='running', child_pid=child.pid)
                publish(args, state)
                code = child.wait()
            state['child_pid'] = None
            attempt.update(exit_code=code, finished_at_unix=time.time())
            if code:
                raise RuntimeError(f'{job["key"]} exited with code {code}; inspect {log_path}')
            metrics_path = output / 'metrics.json'
            validated_metrics(metrics_path, job, plan)
            attempt.update(status='completed', metrics_path=str(metrics_path))
            job.update(status='completed', metrics_path=str(metrics_path), metrics_sha256=sha256(metrics_path),
                       batch_size=batch, eval_batch_size=eval_batch, finished_at_unix=time.time())
            publish(args, state)
            return batch, eval_batch
        except Exception as error:
            oom = log_path.exists() and is_oom(log_path)
            attempt.update(status='failed', error=str(error), out_of_memory=oom, finished_at_unix=time.time())
            job.update(status='failed', error=str(error))
            atomic_json(attempt, output / 'queue_attempt_error.json')
            publish(args, state)
            if not oom or (batch == 1 and eval_batch == 1):
                raise
            batch, eval_batch = 1, 1
            job.update(status='retrying_after_oom', retry_note='fresh model; microbatch 1; effective batch stays 32')
            publish(args, state)


def update_summary(args, plan, state):
    rows, metrics_by_mode = [], {}
    for job in state['jobs']:
        if job['is_smoke']:
            continue
        row = {'data': job['data'], 'seed': args.seed, 'status': job['status'],
               'test_auc': '', 'test_auprc': '', 'test_f1': '', 'best_epoch': '',
               'epochs_completed': '', 'metrics_path': job.get('metrics_path', '')}
        if verified_success(job, plan):
            metrics = validated_metrics(job['metrics_path'], job, plan)
            metrics_by_mode[job['data']] = metrics
            row.update({f'test_{name}': metrics['test'][name] for name in ('auc', 'auprc', 'f1')})
            row.update({name: metrics.get(name, '') for name in ('best_epoch', 'epochs_completed')})
        rows.append(row)
    temporary = args.run_root / f'summary.csv.{os.getpid()}.tmp'
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.run_root / 'summary.csv')
    atomic_json({'status': state['status'], 'plan_sha256': plan['plan_sha256'],
                 'results': metrics_by_mode}, args.run_root / 'summary.json')


def supervise(args, plan):
    args.run_root.mkdir(parents=True, exist_ok=True)
    gpu_path = ROOT / 'results/.kip_data_efficiency_gpu.lock'
    gpu_path.parent.mkdir(parents=True, exist_ok=True)
    with (args.run_root / 'queue.lock').open('a') as local_lock, gpu_path.open('a') as gpu_lock:
        try:
            fcntl.flock(local_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('this CKD queue or a surviving trainer is already running') from error
        freeze_plan(args, plan)
        path = args.run_root / 'queue_status.json'
        old = json.loads(path.read_text()) if path.exists() else {}
        if old and old.get('plan_sha256') != plan['plan_sha256']:
            raise RuntimeError('existing queue status belongs to a different protocol')
        prior = {job['key']: job for job in old.get('jobs', [])}
        jobs = [{**job, **prior.get(job['key'], {})} for job in plan['jobs']]
        for job in jobs:
            if not verified_success(job, plan):
                job['status'] = 'pending'
        state = {'schema_version': 1, 'plan_sha256': plan['plan_sha256'],
                 'status': 'waiting_for_kip', 'supervisor_pid': os.getpid(), 'child_pid': None,
                 'started_at_unix': old.get('started_at_unix', time.time()), 'jobs': jobs,
                 'prerequisite_status': str(args.prerequisite_status), 'current_job': None}
        (args.run_root / 'supervisor.pid').write_text(f'{os.getpid()}\n')
        publish(args, state)
        try:
            if not all(verified_success(job, plan) for job in jobs):
                wait_for_gpu(args, state, gpu_lock)
                verify_inputs(args, plan)
                lock_fds = (local_lock.fileno(), gpu_lock.fileno())
                state['model_audit'] = ensure_model_audit(args, state, lock_fds)
                batch, eval_batch = args.batch_size, args.eval_batch_size
                for job in jobs:
                    if verified_success(job, plan):
                        batch, eval_batch = job.get('batch_size', batch), job.get('eval_batch_size', eval_batch)
                    else:
                        batch, eval_batch = run_job(args, plan, state, job, batch, eval_batch, lock_fds)
                    update_summary(args, plan, state)
            state.update(status='completed', current_job=None, child_pid=None)
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                         error=str(error), error_type=type(error).__name__)
            raise
        finally:
            state['finished_at_unix'] = time.time()
            publish(args, state)
            update_summary(args, plan, state)
        return state


def background_command(args):
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve())]
    for name in ('run_root', 'model_path', 'data_dir', 'prerequisite_status', 'seed', 'batch_size',
                  'eval_batch_size', 'epochs', 'patience', 'min_epochs', 'lr', 'max_length',
                  'threads', 'poll_seconds'):
        command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    return command


def main(argv=None):
    args = parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.background:
        args.run_root.mkdir(parents=True, exist_ok=True)
        with (args.run_root / 'supervisor.log').open('a') as log:
            child = subprocess.Popen(background_command(args), cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, env=environment(args),
                                     start_new_session=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            path = args.run_root / 'queue_status.json'
            if path.exists():
                state = json.loads(path.read_text())
                if state.get('supervisor_pid') == child.pid and state.get('status') in {
                        'waiting_for_kip', 'waiting_for_gpu', 'running', 'completed'}:
                    if state['status'] != 'completed' and child.poll() is not None:
                        raise RuntimeError(f'supervisor exited unexpectedly; inspect {args.run_root / "supervisor.log"}')
                    print(json.dumps({'status': state['status'], 'supervisor_pid': child.pid,
                                      'queue_status': str(path), 'run_root': str(args.run_root)}))
                    return
            if child.poll() is not None:
                raise RuntimeError(f'supervisor exited ({child.returncode}); inspect {args.run_root / "supervisor.log"}')
            time.sleep(.1)
        raise RuntimeError(f'supervisor startup not confirmed; inspect PID {child.pid} and supervisor.log before retrying')
    state = supervise(args, plan)
    print(json.dumps({'status': state['status'], 'run_root': str(args.run_root)}))


if __name__ == '__main__':
    main()
