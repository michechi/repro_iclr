"""Text-only outcome classifiers for the KIP data-efficiency experiment.

The supervised objective is a two-class head on the final unmasked text token.
Qwen's language-model logits and vision tower never participate in this model.
Pretrained checkpoints must already exist locally; this module downloads nothing.
"""
from __future__ import annotations

import gc
import importlib.metadata
from pathlib import Path

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


ARMS = ("qwen27b_qlora", "qwen14b_qlora", "llama8b_lora", "llama8b_full")
MODEL_NAMES = {
    "qwen27b_qlora": "Qwen/Qwen3.8-27B",
    "qwen14b_qlora": "Qwen/Qwen2.5-14B",
    "llama8b_lora": "meta-llama/Llama-3.1-8B",
    "llama8b_full": "meta-llama/Llama-3.1-8B",
}
_ALIASES = {
    "qwen27_qlora": "qwen27b_qlora", "qwen38_27b_qlora": "qwen27b_qlora",
    "qwen14_qlora": "qwen14b_qlora", "qwen25_14b_qlora": "qwen14b_qlora",
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


class LastTokenClassifier(nn.Module):
    """Uniform interface over Llama, Qwen2 and Qwen3.5 text backbones."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.config.num_labels = 2
        self.config.use_cache = False
        self.config.problem_type = "single_label_classification"
        embedding = backbone.get_input_embeddings().weight
        self.score = nn.Linear(self.config.hidden_size, 2, bias=False,
                               device=embedding.device, dtype=torch.float32)
        nn.init.normal_(self.score.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SequenceClassifierOutput:
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask shapes must match")
        # Mask-derived positions make both left and right padding equivalent to
        # an unpadded example. Qwen3.5 expands these to its text-only MRoPE axes.
        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids, use_cache=False, return_dict=True)
        pooled = last_unmasked_hidden(outputs.last_hidden_state, attention_mask)
        # FP32 trainable head, matching the existing Llama outcome classifier.
        logits = self.score(pooled.to(self.score.weight.dtype))
        return SequenceClassifierOutput(logits=logits)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()


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


def _validate_loading(info: dict, *, multimodal: bool) -> dict:
    for kind in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"):
        if info.get(kind):
            raise ValueError(f"pretrained model loading failed: {kind}={info[kind]}")
    unexpected = sorted(info.get("unexpected_keys", []))
    allowed = (lambda name: name.startswith("mtp.")) if multimodal else (lambda name: name == "lm_head.weight")
    if any(not allowed(name) for name in unexpected):
        raise ValueError(f"unexpected pretrained checkpoint keys: {unexpected}")
    return {"missing_keys": [], "mismatched_keys": [], "error_msgs": [], "unexpected_keys": unexpected}


def _versions() -> dict:
    result = {}
    for package in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "fla-core", "causal-conv1d"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _logical_numel(parameter: nn.Parameter) -> int:
    # bitsandbytes packs two 4-bit values into every storage byte. Its wrapper
    # numel therefore differs from the number of pretrained scalar weights.
    factor = 2 * parameter.element_size() if type(parameter).__name__ == "Params4bit" else 1
    return parameter.numel() * factor


def load_classifier(arm: str, model_path: str | Path, gradient_checkpointing: bool = True,
                    device: str | torch.device = "cuda") -> tuple[LastTokenClassifier, object, dict]:
    """Load a local checkpoint on one device and return model/tokenizer/metadata.

    QLoRA uses NF4, nested quantization and BF16 compute. The nonquantized LoRA
    backbone stays BF16; adapters and the head are FP32. Full tuning uses FP32
    parameters; the caller controls BF16 autocast and the optimizer placement.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

    arm = normalize_arm(arm)
    source = Path(model_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"pretrained snapshot does not exist: {source}")
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device = normalize_device(device)
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support BF16")
    elif arm != "llama8b_full":
        raise ValueError("production LoRA/QLoRA arms require a BF16-capable CUDA device")

    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    expected_type = "qwen3_5" if arm == "qwen27b_qlora" else ("qwen2" if arm == "qwen14b_qlora" else "llama")
    if config.model_type != expected_type:
        raise ValueError(f"{arm} expects {expected_type}, checkpoint is {config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                              padding_side="right")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must have an EOS token for padding")
    tokenizer.pad_token = tokenizer.eos_token

    quantized = arm.endswith("_qlora")
    full = arm.endswith("_full")
    load_kwargs = dict(local_files_only=True, trust_remote_code=False, use_safetensors=True,
                       device_map={"": str(device)}, attn_implementation="sdpa",
                       dtype=torch.float32 if full else torch.bfloat16, output_loading_info=True)
    quantization = None
    if quantized:
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head", "model.visual"])
        load_kwargs["quantization_config"] = quantization

    discarded = {}
    kernels = None
    if arm == "qwen27b_qlora":
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5 import modeling_qwen3_5 as architecture
        from src.experiments.qwen38_fewshot_eval import kernel_metadata

        pretrained, info = Qwen3_5ForConditionalGeneration.from_pretrained(source, **load_kwargs)
        loading = _validate_loading(info, multimodal=True)
        pretrained.requires_grad_(False)
        backbone = pretrained.model.language_model
        # Retain the quantization attributes used by PEFT's preparation and
        # adapter dispatch when extracting the official text submodel.
        for attribute in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized",
                          "quantization_method", "hf_quantizer"):
            if hasattr(pretrained, attribute):
                setattr(backbone, attribute, getattr(pretrained, attribute))
        backbone.config.quantization_config = quantization.to_dict()
        discarded = {"vision_parameters": sum(_logical_numel(p) for p in pretrained.model.visual.parameters()),
                     "lm_head_parameters": sum(_logical_numel(p) for p in pretrained.lm_head.parameters()),
                     "policy": "frozen and removed before optimizer construction; text-only forward"}
        kernels = kernel_metadata(backbone, architecture)
        del pretrained
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        backbone, info = AutoModel.from_pretrained(source, **load_kwargs)
        loading = _validate_loading(info, multimodal=False)

    backbone.config.pad_token_id = tokenizer.pad_token_id
    backbone.config.use_cache = False
    adapted_names = []
    if not full:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if quantized:
            backbone = prepare_model_for_kbit_training(backbone,
                use_gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            backbone.requires_grad_(False)
            if gradient_checkpointing:
                backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        adapted_names = adapter_module_names(backbone, hybrid=arm == "qwen27b_qlora")
        lora = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16,
                          lora_dropout=0.1, bias="none", target_modules=adapted_names)
        backbone = get_peft_model(backbone, lora, autocast_adapter_dtype=True)
        for parameter in backbone.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    elif gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model = LastTokenClassifier(backbone)
    named_parameters = list(model.named_parameters())
    if any(parameter.is_meta or parameter.device != device for _, parameter in named_parameters):
        raise RuntimeError("every model parameter must be loaded on the requested device")
    if any(parameter.dtype != torch.float32 for _, parameter in named_parameters if parameter.requires_grad):
        raise RuntimeError("all trainable parameters must be FP32")
    if not full and any(parameter.requires_grad and "lora_" not in name and name != "score.weight"
                        for name, parameter in named_parameters):
        raise RuntimeError("LoRA/QLoRA unexpectedly left a base parameter trainable")
    model.train()
    trainable = sum(p.numel() for _, p in named_parameters if p.requires_grad)
    total = sum(_logical_numel(p) for _, p in named_parameters)
    metadata = {
        "arm": arm, "model_name": MODEL_NAMES[arm], "model_path": str(source),
        "model_revision": source.name, "checkpoint_model_type": expected_type,
        "text_model_type": model.config.model_type, "model_class": type(model).__name__,
        "text_backbone_class": type(backbone).__name__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "versions": _versions(), "objective": "last_unmasked_text_hidden_state_linear_2class_cross_entropy",
        "pooling": "final position with attention_mask=1", "head_bias": False,
        "head_initialization": {"distribution": "normal", "mean": 0.0, "std": model.config.initializer_range},
        "loss_dtype": "float32", "trainable_parameter_dtype": "float32",
        "backbone_parameter_dtype": "NF4 with FP32 unquantized parameters" if quantized else ("float32" if full else "bfloat16"),
        "compute_autocast": "bfloat16 (controlled by training loop)",
        "parameters": total, "trainable_parameters": trainable, "frozen_parameters": total - trainable,
        "trainable_parameter_names": [name for name, p in named_parameters if p.requires_grad],
        "quantization": quantization.to_dict() if quantization else None,
        "lora": None if full else {"r": 8, "alpha": 16, "dropout": 0.1, "bias": "none",
            "target_modules": list(HYBRID_TARGETS if arm == "qwen27b_qlora" else ATTENTION_TARGETS),
            "adapted_module_names": adapted_names, "adapted_modules": len(adapted_names),
            "rationale": "attention Q/K/V/output; hybrid layers also adapt QKV and gating/output projections"},
        "gradient_checkpointing": gradient_checkpointing, "gradient_checkpointing_use_reentrant": False,
        "use_cache": False, "padding_side": "right", "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id, "max_position_embeddings": model.config.max_position_embeddings,
        "discarded_checkpoint_components": discarded, "kernel_bindings": kernels, "loading_info": loading,
    }
    return model, tokenizer, metadata
