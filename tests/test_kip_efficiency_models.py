"""Architecture/gradient checks independent of production checkpoint downloads."""
from __future__ import annotations

import json

import pytest
import torch
from torch.nn import functional as F

from src.experiments import kip_efficiency_models as factory


def test_bare_cuda_is_resolved_before_set_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert factory.normalize_device("cuda") == torch.device("cuda:3")
    assert factory.normalize_device(torch.device("cuda")) == torch.device("cuda:3")
    assert factory.normalize_device("cuda:1") == torch.device("cuda:1")
    assert factory.normalize_device("cpu") == torch.device("cpu")


def tiny_backbone(kind, monkeypatch):
    from transformers import LlamaConfig, LlamaModel, Qwen2Config, Qwen2Model

    common = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
                  pad_token_id=2, eos_token_id=2, bos_token_id=1, use_cache=False)
    if kind == "llama":
        return LlamaModel(LlamaConfig(**common))
    if kind == "qwen2":
        return Qwen2Model(Qwen2Config(**common))
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from transformers.models.qwen3_5 import modeling_qwen3_5 as architecture

    # Installed CUDA-only optional kernels must not replace the reference CPU
    # implementation in this independent tiny-configuration test.
    for name in ("FusedRMSNormGated", "causal_conv1d_fn", "causal_conv1d_update",
                 "chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule"):
        monkeypatch.setattr(architecture, name, None)
    return Qwen3_5TextModel(Qwen3_5TextConfig(**common, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
        linear_num_value_heads=2, layer_types=["linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}))


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3_5"])
def test_text_backbones_pool_padding_and_backpropagate_outcome_loss(kind, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = factory.LastTokenClassifier(tiny_backbone(kind, monkeypatch)).eval()
    # A real token equal to the pad ID remains meaningful when its mask is 1.
    ids = torch.tensor([[1, 6, 2, 8], [1, 5, 7, 2]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    logits = model(ids, mask).logits
    assert logits.shape == (2, 2)
    for row, length in enumerate([4, 3]):
        single = model(ids[row:row + 1, :length], torch.ones(1, length, dtype=torch.long)).logits
        torch.testing.assert_close(logits[row], single[0], rtol=1e-5, atol=2e-7)
    loss = F.cross_entropy(logits.float(), torch.tensor([0, 1]))
    loss.backward()
    assert model.score.weight.grad is not None and model.score.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not hasattr(model, "lm_head") and not hasattr(model.backbone, "visual")


def test_pool_uses_last_unmasked_position_and_rejects_empty_rows():
    hidden = torch.arange(24).reshape(2, 4, 3).float()
    mask = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]])
    torch.testing.assert_close(factory.last_unmasked_hidden(hidden, mask), hidden[[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="at least one"):
        factory.last_unmasked_hidden(hidden, torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]))


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3_5"])
def test_lora_adapts_every_attention_projection_and_trainable_restore(kind, monkeypatch):
    from peft import LoraConfig, TaskType, get_peft_model
    from src.experiments.kip_encoder_training import restore_trainable, trainable_state

    backbone = tiny_backbone(kind, monkeypatch)
    names = factory.adapter_module_names(backbone, hybrid=kind == "qwen3_5")
    assert len(names) == (9 if kind == "qwen3_5" else 8)
    backbone.requires_grad_(False)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone = get_peft_model(backbone, LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
        r=8, lora_alpha=16, lora_dropout=0.1, bias="none", target_modules=names))
    model = factory.LastTokenClassifier(backbone).train()
    inputs = torch.tensor([[1, 5, 6, 7], [1, 6, 7, 2]])
    loss = F.cross_entropy(model(inputs, torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])).logits,
                           torch.tensor([0, 1]))
    loss.backward()
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all("lora_" in name or name == "score.weight" for name in trainable)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
    saved = trainable_state(model)
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.1)
    optimizer.step()
    assert any(not torch.equal(before[name], parameter) for name, parameter in trainable.items())
    restore_trainable(model, saved)
    for name, parameter in trainable.items():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


def test_full_factory_loads_local_lm_weights_without_vocabulary_head(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    source = tmp_path / "source"
    backbone = tiny_backbone("llama", monkeypatch)
    pretrained = LlamaForCausalLM(backbone.config)
    pretrained.model.load_state_dict(backbone.state_dict())
    pretrained.save_pretrained(source)
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<bos>": 1, "<eos>": 2, "A": 3}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                        bos_token="<bos>", eos_token="<eos>")
    tokenizer.save_pretrained(source)
    model, loaded_tokenizer, metadata = factory.load_classifier("llama8b_full", source,
                                                               gradient_checkpointing=True, device="cpu")
    assert loaded_tokenizer.padding_side == "right" and loaded_tokenizer.pad_token_id == 2
    assert all(p.dtype == torch.float32 and p.requires_grad for p in model.parameters())
    assert metadata["frozen_parameters"] == 0
    assert metadata["lora"] is None and metadata["quantization"] is None
    json.dumps(metadata)
    for name, parameter in model.backbone.named_parameters():
        torch.testing.assert_close(parameter, dict(backbone.named_parameters())[name], rtol=0, atol=0)
    assert not any("lm_head" in name for name, _ in model.named_parameters())
    F.cross_entropy(model(torch.tensor([[1, 3]]), torch.ones(1, 2, dtype=torch.long)).logits,
                    torch.tensor([1])).backward()
    assert all(p.grad is not None for p in model.parameters())
