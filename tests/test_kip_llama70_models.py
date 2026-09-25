"""CPU checks for pinned 70B prequantization and adapter-only optimization."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from src.experiments import kip_llama70_models as factory


def correct_config():
    return {**factory.ARCHITECTURE, 'architectures': ['LlamaForCausalLM'],
            '_name_or_path': factory.SOURCE_REPO,
            'quantization_config': {'quant_method': 'bitsandbytes', 'load_in_4bit': True,
                'load_in_8bit': False, 'bnb_4bit_quant_type': 'nf4',
                'bnb_4bit_use_double_quant': True, 'bnb_4bit_compute_dtype': 'bfloat16'}}


@pytest.mark.parametrize('changes', [
    {'model_type': 'qwen2'}, {'hidden_size': 4096}, {'num_hidden_layers': 32},
    {'num_key_value_heads': 64}, {'intermediate_size': 24576},
    {'_name_or_path': 'meta-llama/Llama-3.1-70B-Instruct'},
    {'architectures': ['LlamaForSequenceClassification']},
])
def test_wrong_base_or_architecture_rejected(changes):
    with pytest.raises(ValueError):
        factory.validate_checkpoint_config({**correct_config(), **changes})


@pytest.mark.parametrize('changes', [
    {'quant_method': 'gptq'}, {'load_in_4bit': False}, {'load_in_8bit': True},
    {'bnb_4bit_quant_type': 'fp4'}, {'bnb_4bit_use_double_quant': False},
])
def test_wrong_prequantization_rejected(changes):
    config = correct_config()
    config['quantization_config'].update(changes)
    with pytest.raises(ValueError, match='NF4'):
        factory.validate_checkpoint_config(config)


def test_correct_base_config_copies_quantization_and_accepts_serialized_private_flags():
    config = correct_config()
    quantization = factory.validate_checkpoint_config(config)
    quantization['bnb_4bit_quant_type'] = 'bad'
    assert config['quantization_config']['bnb_4bit_quant_type'] == 'nf4'
    config['quantization_config']['_load_in_4bit'] = config['quantization_config'].pop('load_in_4bit')
    assert factory.validate_checkpoint_config(config)['_load_in_4bit'] is True


def complete_index():
    plain, quantized = factory.expected_checkpoint_keys()
    return {'weight_map': {name: 'model-00001-of-00001.safetensors'
            for name in plain | quantized | {name + suffix for name in quantized
                                            for suffix in factory.QUANT_SUFFIXES}}}


def test_complete_index_checks_all_layers_and_quantization_state():
    index = complete_index()
    report = factory.validate_checkpoint_index(index)
    assert report['quantized_projections'] == 560
    assert report['unquantized_tensors'] == 163
    assert report['tensor_keys'] == 3523
    for removed in ['model.embed_tokens.weight', 'lm_head.weight',
                    'model.layers.79.self_attn.q_proj.weight',
                    'model.layers.79.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4',
                    'model.layers.0.mlp.down_proj.weight.nested_absmax']:
        incomplete = copy.deepcopy(index)
        del incomplete['weight_map'][removed]
        with pytest.raises(ValueError, match='mapping mismatch'):
            factory.validate_checkpoint_index(incomplete)
    index['weight_map']['model.layers.80.self_attn.q_proj.weight'] = 'model.safetensors'
    with pytest.raises(ValueError, match='unexpected'):
        factory.validate_checkpoint_index(index)


@pytest.mark.parametrize('bad_shard', ['../outside.safetensors', '/absolute.safetensors', 42, 'model.bin'])
def test_checkpoint_index_rejects_unsafe_or_non_safetensors_paths(bad_shard):
    index = complete_index()
    index['weight_map']['lm_head.weight'] = bad_shard
    with pytest.raises(ValueError, match='shard name'):
        factory.validate_checkpoint_index(index)


@pytest.mark.parametrize('field', ['missing_keys', 'unexpected_keys', 'mismatched_keys',
                                  'error_msgs', 'conversion_errors'])
def test_loading_never_ignores_unconsumed_tensors_or_initialized_weights(field):
    with pytest.raises(ValueError, match=field):
        factory.validate_loading({field: ['model.layers.0.self_attn.q_proj.weight.quant_state.bitsandbytes__nf4']})
    assert all(value == [] for value in factory.validate_loading({}).values())


def test_logical_numel_uses_original_quantized_shape():
    parameter = torch.nn.Parameter(torch.ones(7, 1), requires_grad=False)
    parameter.quant_state = SimpleNamespace(shape=(3, 4))
    assert factory.logical_numel(parameter) == 12
    assert factory.logical_numel(torch.ones(7, 1)) == 7


def test_attention_lora_and_outcome_head_only_are_trainable_and_receive_gradients():
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaModel

    torch.set_num_threads(1)
    torch.manual_seed(8)
    backbone = LlamaModel(LlamaConfig(vocab_size=16, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, pad_token_id=2, eos_token_id=2))
    backbone.requires_grad_(False)
    targets = factory.adapter_module_names(backbone, hybrid=False)
    assert len(targets) == 8
    backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=8, lora_alpha=16, lora_dropout=0.1, bias='none', target_modules=targets))
    model = factory.LastTokenClassifier(backbone)
    counts = factory.validate_trainable_parameters(model, torch.device('cpu'))
    assert counts['trainable_parameters'] > 0
    assert set(name.rsplit('.', 1)[0].split('.')[-1] for name in targets) == {'self_attn'}
    result = model(torch.tensor([[1, 4, 5], [1, 6, 2]]), torch.tensor([[1, 1, 1], [1, 1, 0]]))
    torch.nn.functional.cross_entropy(result.logits.float(), torch.tensor([0, 1])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == ('lora_' in name or name == 'score.weight')
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    assert not any('lm_head' in name for name, _ in model.named_parameters())
    model.backbone.get_input_embeddings().weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match='base parameter'):
        factory.validate_trainable_parameters(model, torch.device('cpu'))


def test_trainable_dtype_guard_rejects_half_precision_head():
    model = torch.nn.Module()
    model.score = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='FP32'):
        factory.validate_trainable_parameters(model, torch.device('cpu'))


def test_actual_nf4_double_quantized_checkpoint_round_trip_consumes_all_state_on_cpu(tmp_path):
    """Exercise the production Transformers/bnb deserializer without using CUDA."""
    pytest.importorskip('bitsandbytes')
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(94)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=False))
    model.save_pretrained(tmp_path / 'base')
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    quantized, info = AutoModelForCausalLM.from_pretrained(tmp_path / 'base',
        quantization_config=quantization, device_map={'': 'cpu'}, dtype=torch.bfloat16,
        output_loading_info=True, local_files_only=True)
    factory.validate_loading(info)
    quantized.save_pretrained(tmp_path / 'prequantized')
    restored, info = AutoModelForCausalLM.from_pretrained(tmp_path / 'prequantized',
        quantization_config=quantization, device_map={'': 'cpu'}, dtype=torch.bfloat16,
        output_loading_info=True, local_files_only=True)
    factory.validate_loading(info)
    original_state, restored_state = quantized.state_dict(), restored.state_dict()
    assert set(original_state) == set(restored_state)
    assert sum(name.endswith('.quant_state.bitsandbytes__nf4') for name in restored_state) == 7
    for name in original_state:
        torch.testing.assert_close(restored_state[name], original_state[name], rtol=0, atol=0)
    for projection in factory.ATTENTION_TARGETS:
        module = getattr(restored.model.layers[0].self_attn, projection)
        assert module.weight.device.type == 'cpu'
        assert module.weight.quant_state.quant_type == 'nf4'
        assert module.weight.quant_state.nested
        assert module.compute_dtype == torch.bfloat16
