"""CPU checks of fixed-reference sampling for the thinking extension."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments import qwen_thinking_fewshot_eval as q


TASKS = ['oc_det', 'oc_noise', 'gpc', 'kip4', 'kip6']


def sequence(index):
    letters = []
    for _ in range(20):
        letters.append(chr(65 + index % 26))
        index //= 26
    return ''.join(letters)


@pytest.fixture
def reference(tmp_path, monkeypatch):
    (tmp_path / 'samples').mkdir()
    labels = np.arange(1000, dtype=np.int64) % 2
    sequences = [sequence(i) for i in range(1500)]
    datasets, provenance = {}, {}
    for task in TASKS:
        datasets[task] = {
            'train_x': sequences[:], 'train_y': np.arange(1500, dtype=np.int64) % 2,
            'eligible': np.arange(1500) >= 1000, 'queries': sequences[:1000],
            'labels': labels.copy(), 'test_indices': list(range(10000, 11000)),
        }
        provenance[task] = {
            split: {kind: {'sha256': f'{task}-{split}-{kind}'} for kind in ('features', 'labels')}
            for split in ('train', 'test')
        }
        provenance[task].update(test_indices=list(range(10000, 11000)),
                                sampled_test_positives=500,
                                excluded_training_rows_matching_sampled_test=1000)
        (tmp_path / 'samples' / f'{task}.json').write_text(json.dumps({
            'indices': datasets[task]['test_indices'], 'labels': labels.tolist(),
            'sequences': datasets[task]['queries'], 'test_seed': 9950,
        }))
    config = {'model_name': 'Qwen/Qwen3.8-27B', 'n_test': 1000,
              'data': copy.deepcopy(provenance)}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    calls = []

    def prepare(args):
        calls.append(copy.copy(args))
        assert args.n_test == 1000, 'The entire reference must be prepared before subsetting'
        return ({task: copy.deepcopy(datasets[task]) for task in args.tasks},
                {task: copy.deepcopy(provenance[task]) for task in args.tasks})

    monkeypatch.setattr(q.core, 'prepare_data', prepare)
    args = SimpleNamespace(reference_run=tmp_path, model_name='Qwen/Qwen3.8-27B',
                           tasks=TASKS[:], n_test=1000, test_seed=9950)
    return args, datasets, provenance, config, calls


def test_full_main_preserves_all_five_reference_samples(reference):
    args, full, original_provenance, _, calls = reference
    data, provenance, evidence = q.prepare_reference_data(args)
    assert len(calls) == 1
    assert set(data) == set(TASKS)
    for task in TASKS:
        assert data[task]['queries'] == full[task]['queries']
        assert data[task]['labels'].tolist() == full[task]['labels'].tolist()
        assert data[task]['test_indices'] == full[task]['test_indices']
        assert np.array_equal(data[task]['eligible'], full[task]['eligible'])
        assert provenance[task] == original_provenance[task]
        assert evidence['tasks'][task]['reference_positions'] == list(range(1000))


def test_pilot_is_stratified_subset_and_preserves_main_demo_exclusion_pool(reference):
    args, full, _, _, _ = reference
    args.tasks, args.n_test = ['kip6'], 32
    data, provenance, evidence = q.prepare_reference_data(args)
    rng = np.random.default_rng(9950)
    expected = np.sort(np.concatenate([
        rng.choice(np.arange(1, 1000, 2), 16, replace=False),
        rng.choice(np.arange(0, 1000, 2), 16, replace=False),
    ])).tolist()
    current = data['kip6']
    assert evidence['tasks']['kip6']['reference_positions'] == expected
    assert current['test_indices'] == [10000 + i for i in expected]
    assert current['queries'] == [full['kip6']['queries'][i] for i in expected]
    assert current['labels'].tolist() == [int(full['kip6']['labels'][i]) for i in expected]
    assert provenance['kip6']['sampled_test_positives'] == 16
    assert np.array_equal(current['eligible'], full['kip6']['eligible'])
    assert not current['eligible'][:1000].any()  # Includes 968 excluded rows outside the pilot.
    for seed in (0, 1, 2):
        cell = {'shots': 40, 'demo_seed': seed}
        pilot = q.core.demonstration_payload(current, cell)
        main = q.core.demonstration_payload(full['kip6'], cell)
        assert pilot == main
        assert pilot['positive_labels'] == pilot['negative_labels'] == 20
        assert len(set(pilot['train_indices'])) == 40


@pytest.mark.parametrize('field', ['indices', 'labels', 'sequences'])
def test_reference_sample_changes_are_rejected(reference, field):
    args, _, _, _, _ = reference
    path = args.reference_run / 'samples/kip6.json'
    sample = json.loads(path.read_text())
    sample[field].reverse()
    path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match='full reference test sample differs: kip6'):
        q.prepare_reference_data(args)


def test_reference_source_hash_changes_are_rejected(reference):
    args, _, _, config, _ = reference
    config['data']['kip6']['train']['labels']['sha256'] = 'changed'
    (args.reference_run / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='reference input file differs: kip6/train/labels'):
        q.prepare_reference_data(args)


@pytest.mark.parametrize('change,match', [
    ('model', 'reference must use the same checkpoint name'),
    ('size', 'requested sample exceeds fixed reference test sample'),
])
def test_invalid_reference_requests_rejected_before_data_load(reference, change, match):
    args, _, _, _, calls = reference
    if change == 'model':
        args.model_name = 'different-model'
    else:
        args.n_test = 1001
    with pytest.raises(ValueError, match=match):
        q.prepare_reference_data(args)
    assert calls == []


@pytest.fixture
def interrupted_cell(tmp_path, monkeypatch):
    """Exercise the real cell writer/resume path without loading a model."""
    args = SimpleNamespace(output_dir=tmp_path / 'run', test_seed=9950,
        reasoning_effort='medium', max_thinking_tokens=8, max_length=64,
        batch_size=32, score_batch_size=4, generation_seed=9951, device='cpu')
    cell = {'task': 'kip6', 'shots': 40, 'demo_seed': 0,
            'key': 'kip6/shots40/demo_seed0'}
    task = {'queries': [sequence(i) for i in range(9)],
            'labels': np.arange(9, dtype=np.int64) % 2,
            'test_indices': list(range(10000, 10009))}
    examples = [{'sequence': sequence(100 + i), 'label': i % 2} for i in range(40)]
    demo = {'examples': examples, 'train_indices': list(range(100, 140))}
    directory = args.output_dir / 'cells' / cell['key']
    pending = directory / 'pending_reasoning_000000.json'
    completed = directory / 'batches/batch_000000.json'
    state = {'generation_calls': 0, 'score_sizes': [], 'fail_stage': 'score',
             'failure_injected': False, 'durable_batch_writes': 0}

    class Tokenizer:
        pad_token_id = 0

        def encode(self, prompt, add_special_tokens=False):
            assert add_special_tokens is False
            assert prompt.endswith('<think>\n')
            return [100 + task['queries'].index(prompt.removesuffix('<think>\n'))]

    monkeypatch.setattr(q, 'build_prompt',
                        lambda tokenizer, examples, query, effort: query + '<think>\n')

    def generate(model, tokenizer, prompt_ids, device, max_new_tokens, seed, progress):
        state['generation_calls'] += 1
        assert len(prompt_ids) == 9  # Generation stays one batch; scoring must split it.
        records = []
        for prompt in prompt_ids:
            records.append({'prompt_ids': prompt[:], 'generated_ids': [701, 248069],
                'reasoning_ids': [701], 'scoring_prefix_ids': prompt + [701, 248069, 271],
                'reasoning_token_count': 1, 'generated_token_count': 2,
                'thinking_completed': True, 'forced_close': False,
                'stop_reason': 'thinking_completed', 'stop_token_id': 248069})
        return records

    def score(model, scoring_ids, label_ids, pad_id, device):
        state['score_sizes'].append(len(scoring_ids))
        assert 1 <= len(scoring_ids) <= 4
        assert label_ids == [[15, 16]] * len(scoring_ids)
        assert pending.exists(), 'Reasoning must be durable before the first score call'
        draft = json.loads(pending.read_text())
        assert draft['records_sha256'] == q.core.identity(draft['records'])
        if state['fail_stage'] == 'score' and not state['failure_injected']:
            state['failure_injected'] = True
            raise RuntimeError('injected scoring interruption')
        odds = np.array([(row[0] - 104) / 4 for row in scoring_ids], dtype=np.float32)
        logits = np.column_stack([np.zeros(len(odds), dtype=np.float32), odds])
        probs = 1 / (1 + np.exp(-odds))
        return logits, odds, probs

    real_atomic_json = q.core.atomic_json

    def atomic_json(path, payload):
        if path == completed:
            assert pending.exists(), 'Pending reasoning must survive until completed batch commit'
            draft = json.loads(pending.read_text())
            assert [row['thinking'] for row in payload['rows']] == draft['records']
            if state['fail_stage'] == 'commit' and not state['failure_injected']:
                state['failure_injected'] = True
                raise OSError('injected batch commit interruption')
            real_atomic_json(path, payload)
            assert path.exists() and pending.exists()
            state['durable_batch_writes'] += 1
        else:
            real_atomic_json(path, payload)

    monkeypatch.setattr(q, 'generate_reasoning_batch', generate)
    monkeypatch.setattr(q.core, 'score_batch', score)
    monkeypatch.setattr(q.core, 'atomic_json', atomic_json)

    def evaluate():
        return q.evaluate_cell(args, cell, 'cell-id', 'run-id', task,
            {'train': {}, 'test': {}}, demo, object(), Tokenizer(), {}, {},
            lambda *values, **details: None)

    return SimpleNamespace(evaluate=evaluate, state=state, pending=pending,
                           completed=completed, directory=directory, task=task)


@pytest.mark.parametrize('stage,error', [('score', RuntimeError), ('commit', OSError)])
def test_reasoning_survives_failure_and_resume_scores_in_four_row_chunks(interrupted_cell, stage, error):
    run = interrupted_cell
    run.state['fail_stage'] = stage
    with pytest.raises(error, match='injected'):
        run.evaluate()
    assert run.state['generation_calls'] == 1
    assert run.pending.exists() and not run.completed.exists()
    saved = json.loads(run.pending.read_text())
    assert saved['records_sha256'] == q.core.identity(saved['records'])
    assert len(saved['records']) == 9

    result = run.evaluate()
    assert result['status'] == 'completed'
    assert run.state['generation_calls'] == 1, 'Resume must reuse already generated reasoning'
    expected_sizes = [4, 4, 4, 1] if stage == 'score' else [4, 4, 1, 4, 4, 1]
    assert run.state['score_sizes'] == expected_sizes
    assert run.state['durable_batch_writes'] == 1
    assert run.completed.exists() and not run.pending.exists()
    payload = json.loads(run.completed.read_text())
    assert [row['thinking'] for row in payload['rows']] == saved['records']
    assert [row['test_row_index'] for row in payload['rows']] == run.task['test_indices']
    assert [row['label'] for row in payload['rows']] == run.task['labels'].tolist()
    assert [row['log_odds_1_minus_0'] for row in payload['rows']] == [(i - 4) / 4 for i in range(9)]
    assert payload['rows_sha256'] == q.core.identity(payload['rows'])
    assert q.completed_cell(run.directory, 'cell-id') == result
    before = copy.deepcopy(run.state)
    run.evaluate()  # A complete batch is reusable without generation or scoring.
    assert run.state == before


@pytest.mark.parametrize('corruption', [
    'records', 'prompt_with_updated_hash', 'cell_identity', 'generation_seed', 'input_ids_sha256',
])
def test_corrupted_pending_reasoning_rejected_before_generation_or_scoring(interrupted_cell, corruption):
    run = interrupted_cell
    with pytest.raises(RuntimeError, match='injected scoring interruption'):
        run.evaluate()
    draft = json.loads(run.pending.read_text())
    if corruption == 'records':
        draft['records'][0]['reasoning_ids'] = [999]
    elif corruption == 'prompt_with_updated_hash':
        draft['records'][0]['prompt_ids'] = [999]
        draft['records_sha256'] = q.core.identity(draft['records'])
    elif corruption == 'generation_seed':
        draft['generation_seed'] += 1
    else:
        draft[corruption] = 'corrupted'
    run.pending.write_text(json.dumps(draft))
    before = copy.deepcopy(run.state)
    with pytest.raises(ValueError, match='pending reasoning checkpoint differs'):
        run.evaluate()
    assert run.state == before
    assert run.pending.exists() and not run.completed.exists()
