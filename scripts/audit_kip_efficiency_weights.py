#!/usr/bin/env python3
"""Hash a local Hugging Face checkpoint once and verify its SHA256 blob names.

This only reads local model files. Run once per pinned checkpoint before the
training queue; the resulting JSON is the payload-integrity evidence for fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', prefix=f'.{output.name}.',
                                     suffix='.tmp', dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def audit_weights(model_path: Path, output: Path) -> dict:
    started = time.monotonic()
    source = model_path.resolve(strict=True)
    report = {'schema_version': 1, 'model_path': str(source), 'revision': model_path.name,
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'verifying', 'verified': False, 'weights': {}}
    try:
        index_path = source / 'model.safetensors.index.json'
        index_before = _signature(index_path)
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
        weight_map = index.get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('checkpoint index must contain a nonempty weight_map')
        names = list(weight_map.values())
        if any(not isinstance(name, str) or not name or Path(name).name != name
               or name in {'.', '..'} or not name.endswith('.safetensors') for name in names):
            raise ValueError('checkpoint index contains an invalid shard name')
        report['index_sha256'] = hashlib.sha256(index_bytes).hexdigest()
        for name in sorted(set(names)):
            shard = source / name
            resolved = shard.resolve(strict=True)
            if not resolved.is_file():
                raise ValueError(f'checkpoint shard is not a regular file: {name}')
            expected = resolved.name
            if not re.fullmatch(r'[0-9a-f]{64}', expected):
                raise ValueError(f'{name}: target is not a SHA256-named Hugging Face blob')
            before = _signature(resolved)
            digest = sha256(resolved)
            after = _signature(resolved)
            stable = before == after and shard.resolve(strict=True) == resolved
            record = {'blob_path': str(resolved), 'bytes': before[2], 'mtime_ns': before[3],
                      'sha256': digest, 'expected_hf_blob_sha256': expected,
                      'stable_during_hash': stable, 'verified': stable and digest == expected}
            report['weights'][name] = record
            if not stable:
                raise ValueError(f'{name}: checkpoint shard changed during hashing')
            if digest != expected:
                raise ValueError(f'{name}: payload SHA256 does not match Hugging Face blob identity')
        if _signature(index_path) != index_before:
            raise ValueError('checkpoint index changed during hashing')
        report.update(status='verified', verified=True, shards=len(report['weights']),
                      total_bytes=sum(record['bytes'] for record in report['weights'].values()))
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report['wall_seconds'] = time.monotonic() - started
        atomic_json(report, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit_weights(args.model_path, args.output)
    print(json.dumps({'status': result['status'], 'shards': result['shards'],
                      'total_bytes': result['total_bytes'], 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
