"""CPU checks of native reasoning generation and explicit forced closing."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from src.experiments import qwen_thinking_generation as q


class Tokenizer:
    pad_token_id, eos_token_id = 248044, 248046

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        # Arbitrary decoded generation must not be retokenized.
        return {'</think>\n\n': [248069, 271], '</think>\n\n0': [248069, 271, 15],
                '</think>\n\n1': [248069, 271, 16]}[text]

    def decode(self, ids, **kwargs):
        assert kwargs == {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}
        return 'archive:' + ','.join(map(str, ids))


class Model:
    training = False
    generation_config = SimpleNamespace(eos_token_id=[248046, 248047])

    def __init__(self, rows):
        self.rows, self.calls = rows, []
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16), requires_grad=False)

    def parameters(self):
        return iter([self.weight])

    def generate(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append(kwargs)
        torch.rand(3)  # exercise seeded RNG isolation
        return torch.cat([kwargs['input_ids'], torch.tensor(self.rows, dtype=torch.long)], dim=1)


def test_mixed_stops_preserve_ids_force_close_and_strip_only_post_stop_padding():
    model = Model([[41, 248069, 248044, 248044], [42, 43, 44, 45],
                   [46, 248046, 248044, 248044], [47, 248047, 248044, 248044]])
    progress = []
    records = q.generate_reasoning_batch(model, Tokenizer(), [[7, 8], [9], [10], [11]],
                                          'cpu', 4, 9950, progress.append)
    assert [r['stop_reason'] for r in records] == ['thinking_completed', 'token_cap', 'early_eos', 'early_eos']
    assert [r['generated_token_count'] for r in records] == [2, 4, 2, 2]
    assert [r['forced_close'] for r in records] == [False, True, True, True]
    assert records[0]['generated_ids'] == [41, 248069]
    assert records[1]['scoring_prefix_ids'] == [9, 42, 43, 44, 45, 248069, 271]
    assert records[2]['generated_ids'] == [46, 248046]
    assert records[2]['scoring_prefix_ids'] == [10, 46, 248069, 271]
    assert records[2]['forced_close_reason'] == 'early_eos'
    assert records[0]['reasoning_text'] == 'archive:41'
    assert all(r['label_ids'] == [15, 16] for r in records)
    kwargs = model.calls[0]
    assert kwargs['input_ids'].tolist() == [[7, 8], [248044, 9], [248044, 10], [248044, 11]]
    assert kwargs['attention_mask'].tolist() == [[1, 1], [0, 1], [0, 1], [0, 1]]
    assert kwargs['eos_token_id'] == [248044, 248046, 248047, 248069]
    assert kwargs['use_cache'] and kwargs['logits_to_keep'] == 1
    for name in ('output_scores', 'output_logits', 'output_hidden_states', 'output_attentions', 'return_dict_in_generate'):
        assert kwargs[name] is False
    assert all(kwargs[key] == value for key, value in q.SAMPLING.items())
    assert kwargs['forced_eos_token_id'] is None
    assert progress[0]['event'] == 'reasoning_started'
    assert progress[1]['thinking_completed'] == 1 and progress[1]['forced_close'] == 3


def test_completion_on_last_budget_token_is_natural_and_immediate_eos_is_explicit():
    rows = q.generate_reasoning_batch(Model([[30, 248069], [248044, 248044]]), Tokenizer(),
                                     [[1], [2]], 'cpu', 2, 0)
    assert rows[0]['thinking_completed'] and not rows[0]['forced_close']
    assert rows[1]['reasoning_ids'] == [] and rows[1]['generated_ids'] == [248044]
    assert rows[1]['stop_reason'] == 'early_eos'
    assert rows[1]['scoring_prefix_ids'] == [2, 248069, 271]


def test_fixed_batch_seed_reproduces_sampling_without_mutating_callers_rng():
    class RandomModel(Model):
        def generate(self, **kwargs):
            random_ids = torch.randint(100, 1000, (1, 5))
            return torch.cat([kwargs['input_ids'], random_ids], dim=1)
    model = RandomModel([])
    state = torch.random.get_rng_state().clone()
    a = q.generate_reasoning_batch(model, Tokenizer(), [[1]], 'cpu', 5, 42)
    assert torch.equal(state, torch.random.get_rng_state())
    b = q.generate_reasoning_batch(model, Tokenizer(), [[1]], 'cpu', 5, 42)
    c = q.generate_reasoning_batch(model, Tokenizer(), [[1]], 'cpu', 5, 43)
    assert a[0]['generated_ids'] == b[0]['generated_ids'] != c[0]['generated_ids']


def test_suffix_must_have_exact_label_continuation_without_prefix_retokenization():
    class Bad(Tokenizer):
        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            return result[:-2] + [999] if text.endswith('1') else result
    with pytest.raises(ValueError, match='label 1'):
        q.generate_reasoning_batch(Model([[248069]]), Bad(), [[1]], 'cpu', 1, 0)


@pytest.mark.parametrize('kind', ['train', 'gradient', 'dtype'])
def test_generation_requires_frozen_bfloat16_eval_model(kind):
    model = Model([[248069]])
    if kind == 'train':
        model.training = True
    elif kind == 'gradient':
        model.weight.requires_grad_(True)
    else:
        model.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    with pytest.raises(ValueError, match='eval|frozen BF16'):
        q.generate_reasoning_batch(model, Tokenizer(), [[1]], 'cpu', 1, 0)


def test_unrecognized_short_generation_is_not_mislabeled_as_budget_cap():
    with pytest.raises(ValueError, match='without a recognized stop'):
        q.generate_reasoning_batch(Model([[31]]), Tokenizer(), [[1]], 'cpu', 2, 0)
