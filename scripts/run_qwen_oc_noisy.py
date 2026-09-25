#!/usr/bin/env python3
"""Queue a fresh Qwen27 QLoRA OC-Noisy fit and attention after Llama OC/CKD."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.deps/kip_eff'), str(ROOT)]
from scripts import run_qwen_oc_clinical_audits as oc
from src.analysis.kip_checkpoint_attention import digest, read_json, write_json

RUN = ROOT / 'results/qwen38_27b_oc_noisy_attention_20260924_seed9950'
UPSTREAM = ROOT / 'results/llama8_oc_ckd_attention_20260924'
SELF = str(Path(__file__).resolve().relative_to(ROOT))


def plan_digest(plan):
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != 'plan_sha256'},
                                    sort_keys=True).encode()).hexdigest()


def upstream_complete(state):
    return (state.get('status') == 'completed' and not state.get('child_pid')
            and not state.get('current_job') and bool(state.get('jobs'))
            and all(j.get('status') == 'completed' for j in state['jobs']))


def prepare(root):
    if root.exists():
        raise FileExistsError('new OC-Noisy queue requires a new output directory')
    previous = oc.RUN_ROOT
    old = read_json(previous / 'launch_plan.json')
    evidence_path = previous / 'preflight.json'
    evidence = read_json(evidence_path)
    # Reuse the exhaustive clean-label/tokenization audit only with identical inputs/oracle.
    for source in ('src/analysis/oc_attention_core.py', 'repro/src/analysis/mechanism_id/scripts/common.py'):
        if digest(ROOT / source) != old['source_sha256'][source]:
            raise ValueError('canonical OC oracle changed; rerun the clean-label preflight')
    data = Path(old['data_dir'])
    export_path = data / 'export_manifest.json'
    if digest(export_path) != old['input_sha256'][str(export_path)]:
        raise ValueError('OC export manifest changed')
    export = read_json(export_path)
    if export['noise_probability'] != 0.3:
        raise ValueError('unexpected OC noise probability')
    inputs = {str(export_path): digest(export_path), str(evidence_path): digest(evidence_path)}
    noise_audit = {}
    for split, count in (('train', 400000), ('val', 50000), ('test', 50000)):
        clean_audit = evidence['oc'][split]
        if (clean_audit['rows'] != count or clean_audit['oracle_mismatches'] != 0
                or clean_audit['max_tokens'] > 64):
            raise ValueError('previous exhaustive clean-label audit is invalid')
        for tag in ('ocdet', 'ocnoise'):
            for kind in ('X', 'y'):
                path = data / f'{kind}_{split}_{tag}.csv'
                actual = digest(path)
                if actual != export['splits'][split]['outputs'][path.name]['sha256']:
                    raise ValueError('export changed: ' + str(path))
                if tag == 'ocdet' and actual != old['input_sha256'][str(path)]:
                    raise ValueError('previously audited clean data changed')
                inputs[str(path)] = actual
        if inputs[str(data / f'X_{split}_ocdet.csv')] != inputs[str(data / f'X_{split}_ocnoise.csv')]:
            raise ValueError('clean/noisy sequence rows are not identical')
        labels = {}
        for tag in ('ocdet', 'ocnoise'):
            with (data / f'y_{split}_{tag}.csv').open() as stream:
                labels[tag] = [int(row['Outcome']) for row in csv.DictReader(stream)]
            if len(labels[tag]) != count or not set(labels[tag]) <= {0, 1}:
                raise ValueError('invalid labels or split size')
        flips = sum(a != b for a, b in zip(labels['ocdet'], labels['ocnoise']))
        if flips != export['splits'][split]['flipped_rows']:
            raise ValueError('label flips differ from audited export')
        noise_audit[split] = dict(rows=count, positives=sum(labels['ocnoise']),
                                 flipped_rows=flips, empirical_flip_rate=flips / count,
                                 identical_clean_noisy_sequences=True)
    plan = {k: v for k, v in old.items() if k not in ('amendments', 'plan_sha256')}
    plan.update(task='qwen_oc_noisy_fit_and_attention', tags=['ocnoise'],
                created_at_unix=time.time(), input_sha256=inputs,
                schedule='complete existing Llama8 OC/CKD queue -> fresh Qwen OC-Noisy QLoRA -> OC-Noisy attention',
                upstream_root=str(UPSTREAM), upstream_plan_sha256=digest(UPSTREAM / 'launch_plan.json'),
                reference_ocdet_plan_sha256=digest(previous / 'launch_plan.json'),
                jobs=[dict(key='fit_ocnoise', kind='fit', tag='ocnoise', status='pending'),
                      dict(key='attention_ocnoise', kind='oc_attention', tag='ocnoise', status='pending')])
    plan['source_sha256'] = {s: digest(ROOT / s) for s in (*oc.SOURCES, SELF)}
    plan['plan_sha256'] = plan_digest(plan)
    write_json(root / 'preflight.json', dict(status='passed', noisy=noise_audit,
               reused_clean_preflight=str(evidence_path), clean_preflight_sha256=digest(evidence_path),
               canonical_source_unchanged=True, all_export_hashes_verified=True))
    write_json(root / 'launch_plan.json', plan)
    write_json(root / 'queue_status.json', dict(status='prepared', jobs=plan['jobs'],
               plan_sha256=plan['plan_sha256'], supervisor_pid=None, child_pid=None,
               current_job=None, error=None, updated_at_unix=time.time()))
    print(json.dumps(dict(status='prepared', root=str(root), jobs=[j['key'] for j in plan['jobs']])))


def verify(root, *, model=False):
    plan = read_json(root / 'launch_plan.json')
    if plan_digest(plan) != plan['plan_sha256']:
        raise ValueError('launch plan digest mismatch')
    oc.verify_plan(plan)
    if digest(Path(plan['upstream_root']) / 'launch_plan.json') != plan['upstream_plan_sha256']:
        raise ValueError('upstream Llama plan changed')
    if model:
        source = plan['model_source']
        snapshot = Path(source['snapshot'])
        for name, expected in source['metadata_sha256'].items():
            if digest(snapshot / name) != expected:
                raise ValueError('model metadata changed: ' + name)
        for name, expected in source['weights'].items():
            path = snapshot / name
            if (not path.is_file() or path.stat().st_size != expected['bytes']
                    or path.resolve().name != expected['expected_hf_blob_sha256']):
                raise ValueError('pinned model shard missing or changed: ' + name)
    return plan


def supervise(root):
    with (root / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = verify(root, model=True)
        state = read_json(root / 'queue_status.json')
        if state['status'] == 'completed':
            raise RuntimeError('queue already completed')
        state.update(supervisor_pid=os.getpid(), child_pid=None, current_job=None, error=None)
        try:
            while True:
                upstream = read_json(Path(plan['upstream_root']) / 'queue_status.json')
                if upstream_complete(upstream):
                    break
                state.update(status='waiting_for_upstream', updated_at_unix=time.time(),
                             upstream_status=upstream.get('status'), upstream_job=upstream.get('current_job'),
                             upstream_error=upstream.get('error'))
                write_json(root / 'queue_status.json', state)
                time.sleep(15)
            verify(root, model=True)
            # This runner only acquires its own queue lock and the shared GPU lock.
            # It never replaces, pauses, resumes, or edits another supervisor.
            oc.run_additional(root, lock.fileno())
        except BaseException as error:
            state = read_json(root / 'queue_status.json')
            state.update(status='failed', error=f'{type(error).__name__}: {error}', updated_at_unix=time.time())
            write_json(root / 'queue_status.json', state)
            raise


def background(root):
    verify(root, model=True)
    with (root / 'queue.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read_json(root / 'queue_status.json')['status'] != 'prepared':
            raise RuntimeError('background launch requires a prepared, never-started queue')
    with (root / 'supervisor.log').open('a') as log:
        child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                                  '--run-root', str(root)], cwd=ROOT, env=oc.prior.environment(root),
                                 start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    (root / 'supervisor.pid').write_text(str(child.pid) + '\n')
    for _ in range(100):
        time.sleep(.1)
        if child.poll() is not None:
            raise RuntimeError('supervisor exited; inspect supervisor.log')
        state = read_json(root / 'queue_status.json')
        if state.get('supervisor_pid') == child.pid and state['status'] in {
                'waiting_for_upstream', 'waiting_for_gpu', 'running'}:
            print(json.dumps(dict(supervisor_pid=child.pid, status=state['status'], root=str(root))))
            return
    raise RuntimeError('startup not acknowledged; inspect status before retrying')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=RUN)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--background', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.prepare:
        prepare(root)
    elif args.check:
        verify(root, model=True)
        print(json.dumps(dict(status='verified', root=str(root))))
    elif args.background:
        background(root)
    else:
        supervise(root)


if __name__ == '__main__':
    main()
