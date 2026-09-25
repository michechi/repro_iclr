"""Offline tests for the Ordered Compliance pretrained LoRA runner."""
from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.oc_completion.oracle import SEP
from src.oc_completion import ordered_train_hf as hf


class FakeTokenizer:
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = "[EOS]"
    eos_token_id = 2
    unk_token = "[UNK]"
    unk_token_id = 999
    padding_side = "right"

    def __init__(self, minimum_length: int = 0):
        self.minimum_length = minimum_length

    def _encode(self, text: str, add_special_tokens: bool) -> list[int]:
        words = text.replace("\n", " ").split()
        ids = [10 + (sum(map(ord, word)) % 200) for word in words]
        if "?" in text:
            ids.append(self.unk_token_id)
        if add_special_tokens:
            ids = [101, *ids, 102]
        while len(ids) < self.minimum_length:
            ids.append(7)
        return ids

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        max_length=None,
        return_tensors=None,
    ):
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        rows = [self._encode(text, add_special_tokens) for text in text_list]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        if padding == "max_length":
            rows = [row + [self.pad_token_id] * (max_length - len(row)) for row in rows]
        masks = [[int(token != self.pad_token_id) for token in row] for row in rows]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {
            "input_ids": rows[0] if single else rows,
            "attention_mask": masks[0] if single else masks,
        }

    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_tokenizer.json").write_text("{}")


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        batch, width = input_ids.shape
        positions = torch.arange(width, dtype=torch.float32, device=input_ids.device)
        hidden = positions.view(1, width, 1).expand(batch, width, 4).clone()
        return SimpleNamespace(last_hidden_state=hidden)


class CaptureHead(nn.Module):
    def forward(self, representations):
        return representations[:, :2]


class TinyBertLoRA(nn.Module):
    """Deterministic frozen backbone plus trainable adapter/head."""

    def __init__(self):
        super().__init__()
        self.frozen_backbone = nn.Embedding(1000, 4)
        with torch.no_grad():
            values = torch.arange(4000, dtype=torch.float32).reshape(1000, 4) / 4000
            self.frozen_backbone.weight.copy_(values)
        self.frozen_backbone.weight.requires_grad_(False)
        self.lora_adapter = nn.Linear(4, 4, bias=False)
        self.classification_head = nn.Linear(4, 2)
        with torch.no_grad():
            self.lora_adapter.weight.copy_(torch.eye(4))
            self.classification_head.weight.fill_(0.05)
            self.classification_head.bias.zero_()

    def forward(self, input_ids, attention_mask):
        hidden = self.frozen_backbone(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.classification_head(torch.tanh(self.lora_adapter(pooled)))
        return SimpleNamespace(logits=logits)


def _complete_sequence(index: int) -> str:
    tokens = ["A"] * 20
    tokens[-1] = chr(ord("A") + index % 26)
    return SEP.join(tokens)


def _frame(split: str, n: int) -> pd.DataFrame:
    labels = np.arange(n) % 2
    return pd.DataFrame(
        {
            "sequence_id": [f"{split}_{index}" for index in range(n)],
            "X": [_complete_sequence(index) for index in range(n)],
            "Y_star": labels.astype(np.uint8),
            "noise_uniform": np.linspace(0, 1, n, endpoint=False),
            "flipped": np.zeros(n, dtype=bool),
            "Y_observed": labels.astype(np.uint8),
        }
    )


def test_prompts_use_spaces_and_exact_causal_suffix():
    sequence = SEP.join("ABCDEFGHIJKLMNOPQRST")
    assert hf.build_prompt("bert", sequence) == (
        "Sequential events: A B C D E F G H I J K L M N O P Q R S T"
    )
    assert hf.build_prompt("llama", sequence) == (
        "Sequential events: A B C D E F G H I J K L M N O P Q R S T\n"
        "Outcome (0 or 1):"
    )


def test_tokenization_audit_covers_all_data_and_20_by_26_probes():
    sequences = {
        "train": [_complete_sequence(0), _complete_sequence(1)],
        "val": [_complete_sequence(2)],
        "test": [_complete_sequence(3)],
    }
    audit = hf.audit_tokenization(
        FakeTokenizer(), "bert", sequences, "fake-bert", "fingerprint", batch_size=37
    )
    assert audit["audit_used_truncation"] is False
    assert audit["actual_prompt_count"] == 4
    assert audit["replacement_probe_count"] == 20 * 26
    assert audit["total_prompt_count"] == 524
    assert audit["selected_max_length"] == 64
    assert audit["number_truncated_examples"] == 0
    assert set(audit["token_ids_A_Z"]) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    long_audit = hf.audit_tokenization(
        FakeTokenizer(minimum_length=70),
        "llama",
        sequences,
        "fake-llama",
        "fingerprint",
        batch_size=61,
    )
    assert long_audit["maximum_untruncated_token_length"] == 70
    assert long_audit["selected_max_length"] == 200
    assert long_audit["would_truncate_at_64"] == 1


def test_prompt_dataset_contains_classification_labels_but_no_lm_labels():
    dataset = hf.PromptDataset(
        "llama_lora",
        [_complete_sequence(0)],
        [1],
        [0],
        FakeTokenizer(),
        max_length=64,
    )
    item = dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "labels", "Y_star"}
    assert item["labels"].item() == 1
    assert item["Y_star"].item() == 0
    assert item["input_ids"].shape == (64,)


def test_llama_head_uses_final_non_padding_hidden_state_and_no_lm_labels():
    backbone = FakeBackbone()
    model = hf.LlamaSequenceClassifier(backbone)
    model.classification_head = CaptureHead()
    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    logits = model(input_ids, attention_mask)
    assert logits.tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert set(backbone.calls[0]) == {"input_ids", "attention_mask", "return_dict"}


def test_lora_recipes_and_effective_batches_are_exact():
    assert hf.LEARNING_RATE == 2e-5
    assert hf.EFFECTIVE_BATCH_SIZE == 16
    assert hf.MAX_EPOCHS == 20
    assert hf.PATIENCE == 3
    assert hf.WARMUP_RATIO == 0.06
    assert (hf.LORA_RANK, hf.LORA_ALPHA, hf.LORA_DROPOUT) == (8, 16, 0.1)
    assert hf.MODEL_SPECS["bert_lora"]["lora_targets"] == ("query", "key", "value")
    assert hf.MODEL_SPECS["llama_lora"]["lora_targets"] == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    assert hf.MODEL_SPECS["llama8b_qlora"]["model_name"] == (
        "meta-llama/Llama-3.1-8B"
    )
    assert hf.MODEL_SPECS["qwen14b_qlora"]["model_name"] == "Qwen/Qwen2.5-14B"
    assert hf.MODEL_SPECS["llama70b_qlora"]["model_name"] == (
        "meta-llama/Llama-3.1-70B"
    )
    assert not hf.is_qlora_arm("llama_lora")
    for arm in ("llama8b_qlora", "qwen14b_qlora", "llama70b_qlora"):
        assert hf.is_qlora_arm(arm)
        assert hf.quantization_recipe(arm) == {
            "enabled": True,
            "backend": "bitsandbytes",
            "bits": 4,
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "bfloat16",
            "gradient_checkpointing": True,
            "use_cache": False,
        }
    for spec in hf.MODEL_SPECS.values():
        assert spec["micro_batch"] * spec["gradient_accumulation"] == 16


def test_qlora_builder_uses_nf4_double_quant_and_never_moves_whole_model(
    tmp_path, monkeypatch
):
    captured = {}

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            captured["bnb"] = kwargs

    class QuantizedBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8, use_cache=True, pad_token_id=None)
            self.frozen = nn.Parameter(torch.ones(1))

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            captured["model_name"] = model_name
            captured["load_kwargs"] = kwargs
            return QuantizedBackbone()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            captured["lora"] = kwargs

    def prepare_model_for_kbit_training(model, **kwargs):
        captured["prepare"] = kwargs
        model.frozen.requires_grad_(False)
        return model

    def get_peft_model(model, config):
        captured["adapted"] = True
        return model

    class MovableHead:
        def to(self, **kwargs):
            captured["head_to"] = kwargs
            return self

    class FakeClassifier:
        def __init__(self, backbone):
            self.backbone = backbone
            self.classification_head = MovableHead()

        def to(self, *args, **kwargs):
            raise AssertionError("quantized wrapper must not receive .to()")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoModel=FakeAutoModel,
            BitsAndBytesConfig=FakeBitsAndBytesConfig,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(
            LoraConfig=FakeLoraConfig,
            get_peft_model=get_peft_model,
            prepare_model_for_kbit_training=prepare_model_for_kbit_training,
        ),
    )
    monkeypatch.setattr(hf, "LlamaSequenceClassifier", FakeClassifier)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    tokenizer = FakeTokenizer()
    model = hf.build_hf_model(
        "llama70b_qlora",
        tokenizer,
        torch.device("cuda"),
        torch.bfloat16,
        tmp_path,
    )
    assert isinstance(model, FakeClassifier)
    assert captured["model_name"] == "meta-llama/Llama-3.1-70B"
    assert captured["load_kwargs"]["device_map"] == {"": 0}
    assert captured["load_kwargs"]["low_cpu_mem_usage"] is True
    assert captured["bnb"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": torch.bfloat16,
    }
    assert captured["prepare"] == {
        "use_gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    assert model.backbone.config.use_cache is False
    assert model.backbone.config.pad_token_id == tokenizer.pad_token_id
    assert captured["head_to"] == {
        "device": torch.device("cuda"),
        "dtype": torch.bfloat16,
    }


def test_early_stopping_is_strict_validation_loss_with_patience_three():
    stopper = hf.EarlyStopping(3)
    assert stopper.update(1.0)
    assert not stopper.update(1.0)
    assert not stopper.update(1.1)
    assert not stopper.should_stop
    assert not stopper.update(1.2)
    assert stopper.should_stop
    assert stopper.update(0.9)
    assert stopper.counter == 0


def test_checkpoint_contains_only_trainable_model_tensors_and_restores_logits(tmp_path):
    model = TinyBertLoRA()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=2e-5
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(9)
    tokenizer = FakeTokenizer()
    scorer = hf.HFScorer(
        "bert_lora", model, tokenizer, max_length=64, device=torch.device("cpu")
    )
    sequence = _complete_sequence(0)
    expected = scorer([sequence])
    payload = hf.checkpoint_payload(
        model,
        optimizer,
        scheduler,
        torch.amp.GradScaler("cuda", enabled=False),
        generator,
        "bert_lora",
        0.2,
        9550,
        1,
        0.6,
        1,
        0,
        64,
        [{"epoch": 1}],
        {"sequence": sequence, "classification_logits_fp32": expected[0].tolist()},
    )
    names = set(payload["adapter_and_head_state_dict"])
    assert names
    assert not any(name.startswith("frozen_backbone") for name in names)
    assert all(
        name.startswith("lora_adapter") or name.startswith("classification_head")
        for name in names
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.zero_()
    hf.load_trainable_state_dict(model, payload["adapter_and_head_state_dict"])
    np.testing.assert_allclose(scorer([sequence]), expected, rtol=0, atol=1e-7)


def test_full_fake_training_writes_resume_contract_and_loadable_scorer(
    tmp_path, monkeypatch
):
    frames = {"train": _frame("train", 16), "val": _frame("val", 8), "test": _frame("test", 8)}
    monkeypatch.setattr(hf, "load_split", lambda root, split, pi: frames[split].copy())
    monkeypatch.setattr(hf, "_data_fingerprint", lambda root: "fixed-data")
    monkeypatch.setattr(
        hf,
        "build_tokenizer",
        lambda arm, model_name, cache_dir, tokenizer_path=None: FakeTokenizer(),
    )
    monkeypatch.setattr(
        hf,
        "build_hf_model",
        lambda arm, tokenizer, device, dtype, cache_dir, model_name=None: TinyBertLoRA().to(
            device
        ),
    )
    run_dir = tmp_path / "run"
    args = Namespace(
        arm="bert_lora",
        pi=0.0,
        model_seed=9550,
        data_root=tmp_path / "data",
        checkpoint_root=tmp_path / "checkpoints",
        run_dir=run_dir,
        hf_cache=tmp_path / "cache",
        resume=False,
        smoke=True,
        max_epochs=1,
        patience=3,
        micro_batch=16,
        gradient_accumulation=1,
        eval_batch=8,
        workers=0,
        threads=1,
        audit_batch_size=128,
        max_train_rows=0,
        max_eval_rows=0,
        cpu=True,
    )
    done = hf.train(args)
    assert done["status"] == "complete"
    for relative in (
        "best_standard.pt",
        "last.pt",
        "config.json",
        "history.csv",
        "done.json",
        "tokenization_audit.json",
        "epochs/epoch_001_eval.pt",
    ):
        assert (run_dir / relative).is_file()
    checkpoint = torch.load(run_dir / "best_standard.pt", weights_only=False)
    assert checkpoint["state_format"] == "lora_adapter_and_classification_head_only"
    assert checkpoint["optimizer"]
    assert checkpoint["scheduler"]
    assert checkpoint["rng_states"]
    assert checkpoint["current_epoch"] == 1
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["patience_counter"] == 0
    assert checkpoint["model_name"] == "bert-base-uncased"
    assert checkpoint["lora"] == {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.1,
        "targets": ["query", "key", "value"],
        "bias": "none",
    }
    assert checkpoint["tokenizer_metadata"]["padding_side"] == "right"
    assert not any(
        name.startswith("frozen_backbone")
        for name in checkpoint["adapter_and_head_state_dict"]
    )
    scorer, metadata = hf.load_hf_scorer(run_dir, device="cpu", verify_reload=True)
    scores = scorer.score_sequences([_complete_sequence(0), _complete_sequence(1)])
    assert scores["logits"].shape == (2, 2)
    assert scores["positive_logit"].dtype == np.float32
    assert scores["positive_probability"].dtype == np.float32
    assert metadata["checkpoint_epoch"] == 1
    assert metadata["reload_max_abs_difference"] <= 1e-5
    snapshot_scorer, snapshot_metadata = hf.load_hf_scorer(
        run_dir / "epochs" / "epoch_001_eval.pt", device="cpu", verify_reload=True
    )
    assert snapshot_scorer([_complete_sequence(0)]).shape == (1, 2)
    assert snapshot_metadata["checkpoint_epoch"] == 1
    snapshot_expected = snapshot_scorer([_complete_sequence(0)])
    for parameter in snapshot_scorer.model.parameters():
        if parameter.requires_grad:
            parameter.data.zero_()
    same_scorer, reloaded_metadata = hf.reload_hf_scorer_checkpoint(
        snapshot_scorer,
        run_dir / "epochs" / "epoch_001_eval.pt",
        verify_reload=True,
    )
    assert same_scorer is snapshot_scorer
    np.testing.assert_allclose(
        same_scorer([_complete_sequence(0)]), snapshot_expected, rtol=0, atol=1e-7
    )
    assert reloaded_metadata["checkpoint_epoch"] == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    auto_scorer, _ = hf.load_hf_scorer(run_dir, device="auto", verify_reload=True)
    assert auto_scorer.device.type == "cpu"
    assert hf._valid_done(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["training_objective"] == "binary classification cross entropy only"
    assert config["uses_next_token_loss"] is False

    # Simulate interruption after the last epoch but before final done commit.
    (run_dir / "done.json").unlink()
    args.resume = True
    resumed = hf.train(args)
    assert resumed["status"] == "complete"
    assert resumed["best_epoch"] == 1
    assert hf._valid_done(run_dir)
