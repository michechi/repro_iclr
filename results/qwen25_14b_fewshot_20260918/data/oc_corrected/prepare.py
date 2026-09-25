#!/usr/bin/env python3
"""Export matched corrected OC classification CSVs from canonical parquets.

Run with an interpreter containing pyarrow:
    python -B results/qwen25_14b_fewshot_20260918/data/oc_corrected/prepare.py

Source datasets are read only. Exported row indices are zero based.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def write_csv(path, header, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def counts(values):
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def prepare(source_root, output_dir):
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    assert source_root != output_dir, 'Export must not overwrite the canonical data directory.'
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = source_root / 'dataset_manifest.json'
    noise_path = source_root / 'noise_manifest.json'
    dataset_manifest = json.loads(dataset_path.read_text())
    noise_manifest = json.loads(noise_path.read_text())
    assert dataset_manifest['data_seed'] == noise_manifest['data_seed'] == 9550
    assert noise_manifest['noise_seed'] == 9650
    assert 0.3 in noise_manifest['noise_levels']
    manifest = {
        'schema_version': 1,
        'task_family': 'ordered_compliance_binary_classification',
        'source_root': str(source_root),
        'data_seed': 9550,
        'noise_seed': 9650,
        'noise_probability': 0.3,
        'noise_construction': 'Y_pi_0p3 = Y_star XOR (noise_uniform < 0.3)',
        'source_manifests': {'dataset': record(dataset_path), 'noise': record(noise_path)},
        'mechanism': dataset_manifest['mechanism'],
        'column_mapping': {
            'Sequences': 'splits/{split}.parquet:X; unchanged original U+001F-separated 20-letter string',
            'ocdet.Outcome': 'splits/{split}.parquet:Y_star',
            'ocnoise.Outcome': 'noise/{split}.parquet:Y_pi_0p3, joined one-to-one by sequence_id',
        },
        'row_order': 'Original base parquet row order, shared by ocdet and ocnoise exports.',
        'index_mapping': 'sequence_id_{split}.csv: row_index (zero based), sequence_id',
        'preparation_script': record(Path(__file__).resolve()),
        'splits': {},
    }
    for split in ('train', 'val', 'test'):
        base_path = source_root / 'splits' / f'{split}.parquet'
        noisy_path = source_root / 'noise' / f'{split}.parquet'
        source_records = {'base': record(base_path), 'noise': record(noisy_path)}
        base = pq.read_table(base_path).to_pydict()
        noise = pq.read_table(noisy_path).to_pydict()
        base_ids, noise_ids = base['sequence_id'], noise['sequence_id']
        n = len(base_ids)
        assert n == dataset_manifest['sizes'][split]
        assert len(set(base_ids)) == n, f'{split}: duplicate base sequence_id'
        assert len(noise_ids) == n and len(set(noise_ids)) == n, f'{split}: invalid noise IDs'
        assert set(base_ids) == set(noise_ids), f'{split}: source ID sets differ'
        noise_index = {key: i for i, key in enumerate(noise_ids)}
        alignment = [noise_index[key] for key in base_ids]
        x, clean = base['X'], base['Y_star']
        noisy = [noise['Y_pi_0p3'][i] for i in alignment]
        assert len(x) == len(clean) == len(noisy) == n
        for i, (sequence, y_clean, y_noisy) in enumerate(zip(x, clean, noisy)):
            letters = sequence.split('\x1f')
            assert len(letters) == 20 and all(len(c) == 1 and 'A' <= c <= 'Z' for c in letters)
            assert y_clean in (0, 1) and y_noisy in (0, 1)
            j = alignment[i]
            assert noise['Y_pi_0p0'][j] == y_clean, f'{split}: clean label mismatch'
            flipped = y_clean != y_noisy
            assert flipped == noise['flip_pi_0p3'][j]
            assert flipped == (noise['noise_uniform'][j] < 0.3)
        # Reject changes to source bytes during preparation.
        assert sha256(base_path) == source_records['base']['sha256']
        assert sha256(noisy_path) == source_records['noise']['sha256']
        x_clean = output_dir / f'X_{split}_ocdet.csv'
        x_noisy = output_dir / f'X_{split}_ocnoise.csv'
        y_clean = output_dir / f'y_{split}_ocdet.csv'
        y_noisy = output_dir / f'y_{split}_ocnoise.csv'
        indices = output_dir / f'sequence_id_{split}.csv'
        write_csv(x_clean, ['Sequences'], ((value,) for value in x))
        shutil.copyfile(x_clean, x_noisy)
        write_csv(y_clean, ['Outcome'], ((value,) for value in clean))
        write_csv(y_noisy, ['Outcome'], ((value,) for value in noisy))
        write_csv(indices, ['row_index', 'sequence_id'], enumerate(base_ids))
        outputs = {path.name: record(path) for path in (x_clean, x_noisy, y_clean, y_noisy, indices)}
        assert outputs[x_clean.name]['sha256'] == outputs[x_noisy.name]['sha256']
        manifest['splits'][split] = {
            'rows': n,
            'source_files': source_records,
            'source_ids_already_in_same_order': base_ids == noise_ids,
            'join': 'Validated unique IDs, equal ID sets, explicit one-to-one alignment, complete coverage.',
            'clean_class_counts': counts(clean),
            'noisy_class_counts': counts(noisy),
            'flipped_rows': sum(a != b for a, b in zip(clean, noisy)),
            'outputs': outputs,
        }
        print(f'{split}: {n} rows; clean {counts(clean)}; noisy {counts(noisy)}', flush=True)
    manifest_path = output_dir / 'export_manifest.json'
    temporary = manifest_path.with_name(manifest_path.name + '.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    temporary.replace(manifest_path)
    print(manifest_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=Path(__file__).resolve().parents[4] / 'data/simulation/oc_hole_audit')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prepare(args.source_root, args.output_dir)
