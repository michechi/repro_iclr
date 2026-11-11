"""Pretrained encoder and causal decoder training for OC, KIP, and GPC (Section 4.2-4.3 and Appendix B)."""
from __future__ import annotations

import json
import hashlib
import importlib.metadata
import re
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_outputs import SequenceClassifierOutput

from src.experiments.paper_protocol import CONFIG, model_spec, sequence_prompt


class PaperTokenizedDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=200, decoder=True, sequence_length=20):
        if len(sequences) != len(labels) or not len(sequences):
            raise ValueError("nonempty matching sequences and labels required")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.input_ids = []
        for start in range(0, len(sequences), 2048):
            prompts = [sequence_prompt(s, decoder=decoder, sequence_length=sequence_length) for s in sequences[start:start + 2048]]
            encoded = tokenizer(prompts, padding=False, truncation=True, max_length=max_length,
                                add_special_tokens=True, return_attention_mask=False,
                                return_token_type_ids=False)
            self.input_ids.extend(encoded["input_ids"])
        self.min_tokens = min(map(len, self.input_ids))
        self.max_tokens = max(map(len, self.input_ids))
        if self.min_tokens < (2 if decoder else 1) or self.max_tokens > max_length:
            raise ValueError("invalid tokenized prompt length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], int(self.labels[index])


def last_unmasked(hidden, attention_mask):
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
    last = positions.masked_fill(~attention_mask.bool(), -1).amax(dim=1)
    if (last < 0).any():
        raise ValueError("every prompt needs an unmasked token")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def prompt_lm_labels(input_ids, attention_mask):
    """Exclude pads and the first real token, which has no real predecessor."""
    labels = input_ids.masked_fill(~attention_mask.bool(), -100).clone()
    previous_is_real = F.pad(attention_mask[:, :-1].bool(), (1, 0), value=False)
    labels.masked_fill_(~previous_is_real, -100)
    if not (labels[:, 1:] != -100).any():
        raise ValueError("prompt LM loss needs at least one unmasked token transition")
    return labels


class PaperDecoderClassifier(nn.Module):
    paper_objective = True

    def __init__(self, causal_lm):
        super().__init__()
        self.backbone = causal_lm
        self.config = getattr(causal_lm.config, "text_config", causal_lm.config)
        self.config.num_labels = 2
        width = self.config.hidden_size
        embedding = causal_lm.get_input_embeddings().weight
        self.score = nn.Sequential(nn.Linear(width, width // 2), nn.Tanh(), nn.Dropout(CONFIG.head_dropout),
                                   nn.Linear(width // 2, 2)).to(device=embedding.device, dtype=torch.float32)

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels=None, output_attentions=False):
        positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=positions, output_hidden_states=True,
                               output_attentions=output_attentions,
                               use_cache=False, return_dict=True)
        pooled = last_unmasked(output.hidden_states[-1], attention_mask)
        logits = self.score(pooled.float()).float()
        loss = None
        if labels is not None:
            lm_labels = prompt_lm_labels(input_ids, attention_mask)
            targets = lm_labels[:, 1:]
            token_losses = F.cross_entropy(
                output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
                targets.reshape(-1), ignore_index=-100, reduction="none").reshape_as(targets)
            counts = (targets != -100).sum(dim=1)
            if (counts == 0).any():
                raise ValueError("every prompt needs a next-token target")
            lm_loss = (token_losses.sum(dim=1) / counts).mean()
            loss = lm_loss + F.cross_entropy(logits, labels.long())
        return SequenceClassifierOutput(loss=loss, logits=logits, attentions=output.attentions)


def load_paper_model(model_key, model_path, *, tuning=None, gradient_checkpointing=True, device="cuda",
                     allow_smoke_model=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from src.experiments.kip_efficiency_models import normalize_device

    spec = model_spec(model_key, tuning)
    source = Path(model_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"local pretrained snapshot missing: {source}")
    device = normalize_device(device)
    if device.type == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("a BF16-capable CUDA device is required")
    if spec["tuning"] == "qlora" and device.type != "cuda":
        raise ValueError("QLoRA requires CUDA")
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    text_config = getattr(config, "text_config", config)
    family_types = {"bert": {"bert"}, "roberta": {"roberta"}, "roberta_base": {"roberta"},
                    "llama1b": {"llama"}, "llama8b": {"llama"}, "llama70b": {"llama"},
                    "qwen14b": {"qwen2"}, "qwen4b": {"qwen3"}, "qwen27b": {"qwen3_5"}}
    if config.model_type not in family_types[model_key]:
        raise ValueError(f"{model_key}: unexpected checkpoint model_type {config.model_type}")
    expected_shapes = {"bert": (768, 12), "roberta": (1024, 24), "roberta_base": (768, 12),
                       "llama1b": (2048, 16), "llama8b": (4096, 32),
                       "llama70b": (8192, 80), "qwen14b": (5120, 48)}
    actual_shape = (text_config.hidden_size, text_config.num_hidden_layers)
    if not allow_smoke_model and model_key in expected_shapes and actual_shape != expected_shapes[model_key]:
        raise ValueError(f"{model_key}: expected hidden width/layers {expected_shapes[model_key]}, got {actual_shape}")
    if model_key in {"roberta", "roberta_base"} and config.model_type != "roberta":
        raise ValueError(f"{model_key} requires a RoBERTa checkpoint")
    from src.experiments.paper_data import sha256
    weight_files = sorted(source.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError("local snapshot has no safetensors weights")
    weight_hashes = {p.name: sha256(p) for p in weight_files}
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False,
                                              padding_side="right")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(local_files_only=True, trust_remote_code=False, use_safetensors=True,
                  dtype=torch.float32 if spec["tuning"] == "full" or device.type == "cpu" else torch.bfloat16,
                  attn_implementation="sdpa", output_loading_info=True)
    quantization = None
    if spec["tuning"] == "qlora":
        from transformers import BitsAndBytesConfig
        quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
                         llm_int8_skip_modules=["lm_head", "model.visual"])
        kwargs.update(quantization_config=quantization, device_map={"": str(device)})
    if spec["family"] == "encoder":
        base, loading_info = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2,
                    problem_type="single_label_classification", **kwargs)
    elif config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration
        base, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(source, **kwargs)
    else:
        base, loading_info = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    for field in ("mismatched_keys", "error_msgs", "conversion_errors"):
        if loading_info.get(field):
            raise ValueError(f"pretrained checkpoint loading failed: {field}={loading_info[field]}")
    missing = loading_info.get("missing_keys", [])
    if any(spec["family"] != "encoder" or not name.startswith("classifier.") for name in missing):
        raise ValueError(f"pretrained checkpoint has missing backbone weights: {missing}")
    unexpected = loading_info.get("unexpected_keys", [])
    allowed_unused = ("cls.", "lm_head.", "roberta.pooler.") if spec["family"] == "encoder" else ("mtp.",)
    if any(not name.startswith(allowed_unused) for name in unexpected):
        raise ValueError(f"pretrained checkpoint has unexpected backbone weights: {unexpected}")
    logical_parameters = sum(p.numel() * (2 * p.element_size() if type(p).__name__ == "Params4bit" else 1)
                             for p in base.parameters())
    size_bounds = {"qwen4b": (3_500_000_000, 4_500_000_000),
                   "qwen27b": (24_000_000_000, 30_000_000_000)}
    if not allow_smoke_model and model_key in size_bounds:
        lower, upper = size_bounds[model_key]
        if not lower <= logical_parameters <= upper:
            raise ValueError(f"{model_key}: checkpoint parameter count {logical_parameters} is outside declared model scale")
    if quantization is None:
        base.to(device)
    base.config.pad_token_id = tokenizer.pad_token_id
    text_config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = False
    targets = ["query", "key", "value"] if spec["family"] == "encoder" else ["q_proj", "k_proj", "v_proj", "o_proj"]
    adapted_names = []
    if spec["tuning"] != "full":
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        if quantization is not None:
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=gradient_checkpointing,
                    gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            base.requires_grad_(False)
        adapted_names = sorted(name for name, module in base.named_modules()
                               if name.rsplit(".", 1)[-1] in targets and "visual" not in name
                               and "vision" not in name and hasattr(module, "weight"))
        if set(targets) - {name.rsplit(".", 1)[-1] for name in adapted_names}:
            raise ValueError("model checkpoint lacks LoRA targets")
        base = get_peft_model(base, LoraConfig(r=CONFIG.lora_r, lora_alpha=CONFIG.lora_alpha,
                    lora_dropout=CONFIG.lora_dropout, bias="none",
                    target_modules=adapted_names,
                    task_type=TaskType.SEQ_CLS if spec["family"] == "encoder" else TaskType.CAUSAL_LM))
        for parameter in base.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PaperDecoderClassifier(base) if spec["family"] == "decoder" else base
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if spec["family"] == "decoder" and spec["tuning"] != "full":
        if any("lora_" not in name and not name.startswith("score.") for name in trainable):
            raise RuntimeError("unexpected trainable decoder base parameter")
    metadata = {**CONFIG.to_dict(), **spec, "model_path": str(source),
                "model_revision": getattr(config, "_commit_hash", None) or (source.name if re.fullmatch(r"[0-9a-f]{40}", source.name) else None),
                "snapshot_directory_name": source.name, "device": str(device),
                "compute_dtype": "bfloat16" if device.type == "cuda" else "float32",
                "tokenizer_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in source.iterdir() if p.is_file() and
                    (p.name.startswith("tokenizer") or p.name in {"vocab.json", "vocab.txt", "merges.txt", "special_tokens_map.json"})},
                "model_config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
                "pretrained_weight_files_sha256": weight_hashes,
                "pretrained_parameter_count": logical_parameters,
                "allow_smoke_model": allow_smoke_model,
                "actual_architecture": {"model_type": config.model_type, "hidden_size": text_config.hidden_size,
                                        "num_hidden_layers": text_config.num_hidden_layers,
                                        "vocab_size": text_config.vocab_size},
                "loading_info": {key: sorted(str(item) for item in loading_info.get(key, [])) for key in
                    ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs", "conversion_errors")},
                "lm_reduction": "mean over valid next-token targets per example, then mean over examples",
                "environment_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy")},
                "checkpoint_model_type": config.model_type, "model_class": type(model).__name__,
                "objective": CONFIG.decoder_loss if spec["family"] == "decoder" else CONFIG.encoder_loss,
                "lora": None if spec["tuning"] == "full" else {"r": CONFIG.lora_r, "alpha": CONFIG.lora_alpha,
                    "dropout": CONFIG.lora_dropout,
                    "target_modules": targets, "adapted_module_names": adapted_names},
                "quantization": quantization.to_dict() if quantization else None,
                "trainable_parameters": sum(p.numel() for p in trainable.values()),
                "trainable_parameter_names": list(trainable),
                "gradient_checkpointing": gradient_checkpointing, "padding_side": "right",
                "head_dtype": "float32"}
    return model, tokenizer, metadata


def train_paper(args):
    from transformers import get_linear_schedule_with_warmup
    from src.experiments.kip_encoder_training import (
        DynamicPadding, EarlyStopping, atomic_json, metric_values, restore_trainable,
        seed_all, sha256, subset_indices, trainable_state, validation_threshold,
    )
    from src.experiments.kip_llama_training import accumulation_samples, classifier_loss, evaluate
    from src.experiments.paper_data import latent_label, read_paper_split
    import pandas as pd

    spec = model_spec(args.model, args.tuning)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("output directory must be empty")
    seed_all(args.seed)
    device = torch.device(args.device)
    selected, provenance = {}, {}
    for split, expected in (("train", 400000), ("val", 50000), ("test", 50000)):
        sequences, labels, evidence = read_paper_split(args.data_dir, args.tag, split)
        if not args.smoke and len(labels) != expected:
            raise ValueError(f"{split}: expected {expected} source rows, got {len(labels)}")
        limit = args.train_size if split == "train" else None
        if args.smoke:
            limit = min(len(labels), 32 if split == "train" else 16)
        # Remark A.4: stratify on latent Y* before noise, not the observed label.
        latent = np.asarray([latent_label(s, evidence["rule"]) for s in sequences], dtype=np.int64)
        indices = subset_indices(labels, limit, args.seed, stratify_by=latent)
        selected[split] = ([sequences[i] for i in indices], labels[indices])
        provenance[split] = evidence
        np.save(output / f"{split}_indices.npy", indices, allow_pickle=False)
    model, tokenizer, model_metadata = load_paper_model(args.model, args.model_path, tuning=args.tuning,
                                                       device=device, allow_smoke_model=args.smoke)
    datasets = {split: PaperTokenizedDataset(*data, tokenizer, CONFIG.max_length,
                        decoder=spec["family"] == "decoder", sequence_length=provenance[split]["rule"]["n"]) for split, data in selected.items()}
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(data, batch_size=args.microbatch if split == "train" else args.eval_batch_size,
                   shuffle=split == "train", generator=generator if split == "train" else None,
                   collate_fn=DynamicPadding(tokenizer.pad_token_id, CONFIG.max_length))
               for split, data in datasets.items()}
    manifest = {**CONFIG.to_dict(), "model": model_metadata, "tag": args.tag, "seed": args.seed,
                "rows": {k: len(v) for k, v in datasets.items()}, "data": provenance,
                "microbatch": args.microbatch,
                "gradient_accumulation_steps": CONFIG.effective_batch_size // args.microbatch,
                "evaluation_batch_size": args.eval_batch_size, "requested_train_size": args.train_size,
                "is_smoke": args.smoke,
                "selected_indices_sha256": {split: sha256(output / f"{split}_indices.npy") for split in selected},
                "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("paper_protocol.py"), Path(__file__).with_name("paper_data.py"),
                    Path(__file__).with_name("kip_llama_training.py"), Path(__file__).with_name("kip_encoder_training.py"))}}
    tokenizer.save_pretrained(output / "tokenizer")
    atomic_json(manifest, output / "manifest.json")
    epochs = 1 if args.smoke else CONFIG.max_epochs
    accumulation = CONFIG.effective_batch_size // args.microbatch
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CONFIG.learning_rate,
                                  weight_decay=CONFIG.weight_decay, betas=CONFIG.adam_betas, eps=CONFIG.adam_epsilon)
    steps = math.ceil(len(loaders["train"]) / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(CONFIG.warmup_ratio * steps), steps)
    stopper, best_state, history = EarlyStopping(CONFIG.patience), None, []
    precision = "bf16" if device.type == "cuda" else "fp32"
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loaders["train"]):
            stop, group_samples = accumulation_samples(index, len(loaders["train"]), args.microbatch,
                                                        len(datasets["train"]), accumulation)
            loss, _ = classifier_loss(model, batch, device, precision)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            (loss * len(batch["labels"]) / group_samples).backward()
            if index + 1 == stop:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        val_loss, _, _ = evaluate(model, loaders["val"], device, precision)
        improved = stopper.update(val_loss, epoch)
        if improved:
            best_state = trainable_state(model)
        history.append({"epoch": epoch, "validation_loss": val_loss, "improved": improved})
        atomic_json(history, output / "history.json")
        print(json.dumps(history[-1]), flush=True)
        if stopper.stopped:
            break
    if best_state is None:
        raise RuntimeError("no finite validation-best checkpoint")
    restore_trainable(model, best_state)
    torch.save(best_state, output / "selected_trainable.pt")
    val_loss, val_prob, val_y = evaluate(model, loaders["val"], device, precision)
    threshold = validation_threshold(val_y, val_prob)
    test_loss, test_prob, test_y = evaluate(model, loaders["test"], device, precision)
    for split, y, prob in (("val", val_y, val_prob), ("test", test_y, test_prob)):
        pd.DataFrame({"label": y, "probability": prob}).to_csv(output / f"{split}_predictions.csv", index=False)
    metrics = {"status": "completed", "protocol_id": CONFIG.protocol_id, "model": spec,
               "tag": args.tag, "seed": args.seed, "rows": manifest["rows"], "is_smoke": args.smoke,
               "label_convention": provenance["test"]["label_convention"],
               "best_epoch": stopper.best_epoch, "selected_by": "validation_loss",
               "stopping_rule": "validation_loss_patience",
               "threshold_selected_on": "validation",
               "max_length": CONFIG.max_length, "effective_batch_size": CONFIG.effective_batch_size,
               "paper_recipe": f"{spec['model_key']}_{spec['tuning']}",
               "validation": {"loss": val_loss, **metric_values(val_y, val_prob, threshold)},
               "test": {"loss": test_loss, **metric_values(test_y, test_prob, threshold)},
               "training_seconds": time.monotonic() - started,
               "manifest_sha256": sha256(output / "manifest.json"),
               "selected_trainable_sha256": sha256(output / "selected_trainable.pt"),
               "predictions_sha256": {split: sha256(output / f"{split}_predictions.csv") for split in ("val", "test")}}
    atomic_json(metrics, output / "metrics.json")
    return metrics
