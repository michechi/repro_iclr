"""Shared training configuration and model specifications for pretrained experiments (Table 1 and Appendix B)."""
from __future__ import annotations

from dataclasses import asdict, dataclass

PROTOCOL_ID = "paper_pretrained_v1"
SEEDS = (9550, 9551, 9552)
TRAIN_SIZES = (4000, 40000, 120000, 200000, 300000, 400000)
MODELS = {
    "bert": {"model_name": "bert-base-uncased", "family": "encoder", "tuning": "lora"},
    "roberta": {"model_name": "FacebookAI/roberta-large", "family": "encoder", "tuning": "lora"},
    # Appendix G attention audits use RoBERTa-base; main tables use RoBERTa-large above.
    "roberta_base": {"model_name": "FacebookAI/roberta-base", "family": "encoder", "tuning": "full"},
    "llama1b": {"model_name": "meta-llama/Llama-3.2-1B", "family": "decoder", "tuning": "lora"},
    "llama8b": {"model_name": "meta-llama/Llama-3.1-8B", "family": "decoder", "tuning": "lora"},
    "llama70b": {"model_name": "meta-llama/Llama-3.1-70B", "family": "decoder", "tuning": "qlora"},
    "qwen4b": {"model_name": "Qwen/Qwen3-4B-Thinking", "family": "decoder", "tuning": "lora"},
    "qwen14b": {"model_name": "Qwen/Qwen2.5-14B", "family": "decoder", "tuning": "lora"},
    "qwen27b": {"model_name": "Qwen/Qwen3.8-27B", "family": "decoder", "tuning": "qlora"},
}
# QLoRA is required for 27B+ models and is also the Appendix G recipe for Qwen2.5-14B on KIP.
QLORA_MODELS = frozenset({"llama70b", "qwen27b", "qwen14b"})


@dataclass(frozen=True)
class PaperConfig:
    protocol_id: str = PROTOCOL_ID
    optimizer: str = "AdamW"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    effective_batch_size: int = 16
    max_epochs: int = 20
    stopping_statistic: str = "validation_loss"
    patience: int = 3
    schedule: str = "linear"
    warmup_ratio: float = 0.06
    max_length: int = 200
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    head_dropout: float = 0.1
    decoder_loss: str = "prompt_next_token_cross_entropy + classification_cross_entropy"
    decoder_loss_weights: tuple[float, float] = (1.0, 1.0)
    encoder_loss: str = "classification_cross_entropy"
    decoder_head: str = "Linear(d,d//2),Tanh,Dropout(0.1),Linear(d//2,2)"
    decoder_pooling: str = "last attention_mask=1 position"

    def to_dict(self):
        return asdict(self)


CONFIG = PaperConfig()


def model_spec(model: str, tuning: str | None = None) -> dict:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose {tuple(MODELS)}")
    spec = dict(MODELS[model])
    spec["model_key"] = model
    spec["tuning"] = tuning or spec["tuning"]
    if spec["tuning"] not in {"full", "lora", "qlora"}:
        raise ValueError("tuning must be full, lora or qlora")
    if spec["tuning"] == "qlora" and model not in QLORA_MODELS:
        raise ValueError(f"QLoRA is supported only for {sorted(QLORA_MODELS)}")
    if model in {"llama70b", "qwen27b"} and spec["tuning"] != "qlora":
        raise ValueError("QLoRA is required for pretrained 27B+ models")
    return spec


def sequence_prompt(sequence: str, *, decoder: bool = False, sequence_length: int = 20) -> str:
    events = sequence.split("\x1f")
    if len(events) != sequence_length or any(len(event) != 1 or not "A" <= event <= "Z" for event in events):
        raise ValueError("inputs must have the declared number of uppercase A-Z events separated by U+001F")
    prompt = "Sequential events: " + " ".join(events)
    return prompt + "\nOutcome (0 or 1):" if decoder else prompt
