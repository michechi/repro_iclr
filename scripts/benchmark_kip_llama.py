#!/usr/bin/env python3
"""Bounded local Llama classification GPU compatibility/throughput check."""
from pathlib import Path
import json
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    torch.set_num_threads(4)
    torch.manual_seed(9950)
    source = str(Path(__file__).resolve().parents[1] / "data/model_cache/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=2, dtype=torch.float32, local_files_only=True,
    ).cuda()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    assert all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    print(json.dumps({"gpu": torch.cuda.get_device_name(0),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "parameter_dtype": str(next(model.parameters()).dtype)}), flush=True)
    for size in (64, 128, 256):
        torch.cuda.reset_peak_memory_stats()
        batch = tokenizer(["Sequential events: A B C D E F G H I J K L M N O P Q R S T"] * size,
                          padding="max_length", max_length=32, return_tensors="pt").to("cuda")
        assert not (batch.input_ids[batch.attention_mask.bool()] == tokenizer.pad_token_id).any()
        labels = torch.arange(size, device="cuda") % 2
        for _ in range(2):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.monotonic()
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
            loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        assert torch.isfinite(loss)
        print(json.dumps({"batch_size": size, "token_shape": list(batch.input_ids.shape),
                          "seconds_10_steps": elapsed, "samples_per_second": size * 10 / elapsed,
                          "loss": loss.item(), "loss_dtype": str(loss.dtype),
                          "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3}), flush=True)


if __name__ == "__main__":
    main()
