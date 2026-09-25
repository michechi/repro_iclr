"""Isolated Llama-3.1-70B base QLoRA classifier for the KIP M4 protocol.

The pinned Unsloth redistribution stores the Meta base model already quantized
with bitsandbytes. It is not an instruction-tuned model or an Unsloth runtime.
All inference/training uses Transformers, bitsandbytes and PEFT directly.
"""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch

from src.experiments.kip_efficiency_models import (
    ATTENTION_TARGETS, LastTokenClassifier, _versions, adapter_module_names, normalize_device,
)

ARM = 'llama70b_qlora'
BASE_MODEL = 'meta-llama/Llama-3.1-70B'
SOURCE_REPO = 'unsloth/Meta-Llama-3.1-70B-bnb-4bit'
SOURCE_REVISION = 'a009b8db2439814febe725486a5ed388f12a8744'
ARCHITECTURE = {'model_type': 'llama', 'hidden_size': 8192, 'intermediate_size': 28672,
                'num_hidden_layers': 80, 'num_attention_heads': 64, 'num_key_value_heads': 8,
                'vocab_size': 128256, 'tie_word_embeddings': False}
QUANT_SUFFIXES = ('.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map',
                  '.quant_state.bitsandbytes__nf4')


def validate_checkpoint_config(config: dict) -> dict:
    """Reject wrong architecture, instruction checkpoints and quantization recipes."""
    for field, expected in ARCHITECTURE.items():
        if config.get(field) != expected:
            raise ValueError(f'Llama-3.1-70B requires {field}={expected!r}; got {config.get(field)!r}')
    if config.get('architectures') != ['LlamaForCausalLM']:
        raise ValueError('checkpoint must be LlamaForCausalLM before extracting the text backbone')
    for field in ('_name_or_path', 'name_or_path'):
        if 'instruct' in str(config.get(field, '')).lower():
            raise ValueError('KIP comparison requires the base model, not an Instruct checkpoint')
    quantization = config.get('quantization_config', {})
    if (quantization.get('quant_method') != 'bitsandbytes'
            or quantization.get('load_in_4bit', quantization.get('_load_in_4bit')) is not True
            or quantization.get('load_in_8bit', quantization.get('_load_in_8bit', False)) is not False
            or quantization.get('bnb_4bit_quant_type') != 'nf4'
            or quantization.get('bnb_4bit_use_double_quant') is not True):
        raise ValueError('checkpoint must be prequantized bitsandbytes NF4 with double quantization')
    return dict(quantization)


def expected_checkpoint_keys(layers: int = 80) -> tuple[set[str], set[str]]:
    """Complete unquantized and quantized tensor names in the causal-LM source."""
    plain = {'model.embed_tokens.weight', 'model.norm.weight', 'lm_head.weight'}
    quantized = set()
    for layer in range(layers):
        prefix = f'model.layers.{layer}'
        plain.update({f'{prefix}.input_layernorm.weight', f'{prefix}.post_attention_layernorm.weight'})
        for family, projections in [('self_attn', ATTENTION_TARGETS),
                                    ('mlp', ('gate_proj', 'up_proj', 'down_proj'))]:
            quantized.update(f'{prefix}.{family}.{projection}.weight' for projection in projections)
    return plain, quantized


def validate_checkpoint_index(index: dict) -> dict:
    """Require every base tensor and every nested NF4 tensor; no permissive skips."""
    mapping = index.get('weight_map')
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError('checkpoint requires a nonempty safetensors weight_map')
    for shard in mapping.values():
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith('.safetensors'):
            raise ValueError('invalid safetensors checkpoint shard name')
    plain, quantized = expected_checkpoint_keys()
    required = plain | quantized | {name + suffix for name in quantized for suffix in QUANT_SUFFIXES}
    missing, unexpected = sorted(required - mapping.keys()), sorted(mapping.keys() - required)
    if missing or unexpected:
        raise ValueError(f'checkpoint tensor mapping mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}')
    return {'tensor_keys': len(required), 'quantized_projections': len(quantized),
            'unquantized_tensors': len(plain), 'shards': len(set(mapping.values())),
            'complete_prequantized_mapping': True}


def validate_loading(info: dict) -> dict:
    """Quantization state must be consumed by the loader, never ignored wholesale."""
    result = {}
    for kind in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs', 'conversion_errors'):
        if info.get(kind):
            raise ValueError(f'pretrained model loading failed: {kind}={info[kind]}')
        result[kind] = []
    return result


def logical_numel(parameter) -> int:
    state = getattr(parameter, 'quant_state', None)
    if state is not None:
        return math.prod(state.shape)
    return parameter.numel()


def validate_quantized_backbone(backbone) -> dict:
    from bitsandbytes.nn import Linear4bit

    projections = {name: module for name, module in backbone.named_modules()
                   if isinstance(module, Linear4bit)}
    _, expected = expected_checkpoint_keys()
    expected = {name.removeprefix('model.').removesuffix('.weight') for name in expected}
    if set(projections) != expected:
        raise ValueError('the loaded backbone must contain all 560 quantized transformer projections')
    for name, module in projections.items():
        state = module.weight.quant_state
        if state is None or state.quant_type != 'nf4' or not state.nested:
            raise ValueError(f'{name} is missing its nested NF4 quantization state')
        if tuple(state.shape) != (module.out_features, module.in_features):
            raise ValueError(f'{name} has an incompatible restored weight shape')
        if module.compute_dtype != torch.bfloat16:
            raise ValueError(f'{name} does not use BF16 quantized compute')
    return {'quantized_projections': len(projections), 'all_nested_nf4': True,
            'all_shapes_verified': True, 'all_compute_bfloat16': True}


def validate_trainable_parameters(model, device) -> dict:
    named = list(model.named_parameters())
    if any(parameter.is_meta or parameter.device != device for _, parameter in named):
        raise RuntimeError('every model parameter must be loaded on the requested device')
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable or any(parameter.dtype != torch.float32 for _, parameter in trainable):
        raise RuntimeError('all trainable parameters must be nonempty and FP32')
    if any('lora_' not in name and name != 'score.weight' for name, _ in trainable):
        raise RuntimeError('QLoRA unexpectedly left a base parameter trainable')
    return {'trainable_parameters': sum(p.numel() for _, p in trainable),
            'parameters': sum(logical_numel(p) for _, p in named),
            'trainable_parameter_names': [name for name, _ in trainable]}


def load_classifier(arm: str, model_path: str | Path, gradient_checkpointing: bool = True,
                    device: str | torch.device = 'cuda'):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if arm != ARM:
        raise ValueError(f'expected arm {ARM}, got {arm!r}')
    source = Path(model_path).resolve()
    if source.name != SOURCE_REVISION:
        raise ValueError(f'70B source must be the pinned snapshot {SOURCE_REVISION}')
    raw_config = json.loads((source / 'config.json').read_text())
    serialized_quantization = validate_checkpoint_config(raw_config)
    index_evidence = validate_checkpoint_index(json.loads((source / 'model.safetensors.index.json').read_text()))
    device = torch.device(device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Llama-70B QLoRA requires a BF16-capable CUDA GPU')
    device = normalize_device(device)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('selected GPU does not support BF16')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                              padding_side='right')
    if tokenizer.eos_token_id is None:
        raise ValueError('tokenizer must provide an EOS token for padding')
    tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig.from_dict(serialized_quantization)
    quantization.bnb_4bit_compute_dtype = torch.bfloat16
    pretrained, info = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        device_map={'': str(device)}, attn_implementation='sdpa', dtype=torch.bfloat16,
        quantization_config=quantization, output_loading_info=True)
    loading = validate_loading(info)
    pretrained.requires_grad_(False)
    backbone = pretrained.model
    quantized_evidence = validate_quantized_backbone(backbone)
    discarded_lm_head = sum(logical_numel(p) for p in pretrained.lm_head.parameters())
    for attribute in ('is_loaded_in_4bit', 'is_loaded_in_8bit', 'is_quantized',
                      'quantization_method', 'hf_quantizer'):
        if hasattr(pretrained, attribute):
            setattr(backbone, attribute, getattr(pretrained, attribute))
    backbone.config.quantization_config = quantization.to_dict()
    backbone.config.pad_token_id = tokenizer.pad_token_id
    backbone.config.use_cache = False
    del pretrained
    gc.collect()
    torch.cuda.empty_cache()
    backbone = prepare_model_for_kbit_training(
        backbone, use_gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False})
    adapted_names = adapter_module_names(backbone, hybrid=False)
    if len(adapted_names) != 320:
        raise ValueError('expected four attention adapters in each of 80 layers')
    backbone = get_peft_model(backbone, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16, lora_dropout=0.1,
        bias='none', target_modules=adapted_names), autocast_adapter_dtype=True)
    for parameter in backbone.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model = LastTokenClassifier(backbone)
    counts = validate_trainable_parameters(model, device)
    if counts['trainable_parameters'] != 32784384:
        raise ValueError(f"unexpected attention-adapter/head parameter count: {counts['trainable_parameters']}")
    model.train()
    metadata = {
        'arm': ARM, 'model_name': BASE_MODEL, 'model_path': str(source),
        'model_revision': SOURCE_REVISION, 'redistribution_repo': SOURCE_REPO,
        'redistribution_revision': SOURCE_REVISION,
        'checkpoint_provenance': 'public prequantized NF4 base-model redistribution; not an official Meta BF16 snapshot',
        'prequantized': True, 'runtime_uses_unsloth': False, 'architecture': ARCHITECTURE,
        'checkpoint_model_type': 'llama', 'text_model_type': 'llama',
        'model_class': type(model).__name__, 'text_backbone_class': type(backbone).__name__,
        'device': str(device), 'gpu': torch.cuda.get_device_name(device), 'versions': _versions(),
        'objective': 'last_unmasked_text_hidden_state_linear_2class_cross_entropy',
        'pooling': 'final position with attention_mask=1', 'head_bias': False,
        'head_initialization': {'distribution': 'normal', 'mean': 0.0, 'std': model.config.initializer_range},
        'loss_dtype': 'float32', 'trainable_parameter_dtype': 'float32',
        'backbone_parameter_dtype': 'prequantized NF4 with FP32 unquantized parameters',
        'compute_autocast': 'bfloat16 (controlled by training loop)',
        **counts, 'frozen_parameters': counts['parameters'] - counts['trainable_parameters'],
        'quantization': quantization.to_dict(), 'serialized_quantization': serialized_quantization,
        'lora': {'r': 8, 'alpha': 16, 'dropout': 0.1, 'bias': 'none',
                 'target_modules': list(ATTENTION_TARGETS), 'adapted_module_names': adapted_names,
                 'adapted_modules': len(adapted_names)},
        'gradient_checkpointing': gradient_checkpointing, 'gradient_checkpointing_use_reentrant': False,
        'use_cache': False, 'padding_side': 'right', 'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id, 'max_position_embeddings': model.config.max_position_embeddings,
        'discarded_checkpoint_components': {'lm_head_parameters': discarded_lm_head,
            'policy': 'loaded and validated, then removed before optimizer construction; text-only forward'},
        'loading_info': loading, 'index_validation': index_evidence,
        'loaded_quantization_validation': quantized_evidence,
    }
    return model, tokenizer, metadata
