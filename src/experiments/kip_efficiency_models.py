"""Pretrained model and classifier wrappers for KIP and OC/CKD experiments."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ARMS = ("qwen27b_qlora", "qwen14b_lora", "llama8b_lora", "llama8b_full")
MODEL_NAMES = {
    "qwen27b_qlora": "Qwen/Qwen3.8-27B",
    "qwen14b_lora": "Qwen/Qwen2.5-14B",
    "llama8b_lora": "meta-llama/Llama-3.1-8B",
    "llama8b_full": "meta-llama/Llama-3.1-8B",
}
_ALIASES = {
    "qwen27_qlora": "qwen27b_qlora", "qwen38_27b_qlora": "qwen27b_qlora",
    "qwen14_lora": "qwen14b_lora", "qwen25_14b_lora": "qwen14b_lora",
    "llama8_lora": "llama8b_lora", "llama8_full": "llama8b_full",
}
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
HYBRID_TARGETS = ATTENTION_TARGETS + (
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)


def normalize_arm(arm: str) -> str:
    arm = _ALIASES.get(arm, arm)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    return arm


def normalize_device(device: str | torch.device) -> torch.device:
    """Resolve bare CUDA to a concrete index before CUDA APIs consume it."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def last_unmasked_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool by the explicit mask, including content tokens equal to the pad ID."""
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("expected hidden states [batch, time, hidden] and matching mask [batch, time]")
    if hidden_states.shape[1] == 0:
        raise ValueError("cannot pool an empty sequence")
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    last = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("every sequence must contain at least one unmasked token")
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last]


def adapter_module_names(backbone: nn.Module, hybrid: bool) -> list[str]:
    """Find exact text projection paths and reject an architecture mismatch."""
    targets = HYBRID_TARGETS if hybrid else ATTENTION_TARGETS
    names = sorted(name for name, module in backbone.named_modules()
                   if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear))
    present = {name.rsplit(".", 1)[-1] for name in names}
    missing = set(targets) - present
    if missing:
        raise ValueError(f"checkpoint lacks expected attention adapter targets: {sorted(missing)}")
    if any("visual" in name or "vision" in name for name in names):
        raise ValueError("adapter targets unexpectedly include a vision tower")
    return names


def load_classifier(arm: str, model_path: str | Path, gradient_checkpointing: bool = True,
                    device: str | torch.device = "cuda", protocol: str = "paper") -> tuple[nn.Module, object, dict]:
    """Load a local checkpoint using the paper's decoder classifier and return model/tokenizer/metadata."""
    if protocol != "paper":
        raise ValueError("protocol must be 'paper'")
    arm = normalize_arm(arm)
    from src.experiments.paper_pretrained_training import load_paper_model

    key = "qwen27b" if arm.startswith("qwen27") else "qwen14b" if arm.startswith("qwen14") else "llama8b"
    model, tokenizer, metadata = load_paper_model(
        key, model_path, tuning=arm.rsplit("_", 1)[1],
        gradient_checkpointing=gradient_checkpointing, device=device,
    )
    metadata["arm"] = arm
    return model, tokenizer, metadata
