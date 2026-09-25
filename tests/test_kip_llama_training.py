from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments import kip_llama_training as runner


def tiny_model():
    from transformers import LlamaConfig, LlamaForSequenceClassification

    return LlamaForSequenceClassification(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        num_labels=2, pad_token_id=2, eos_token_id=2, bos_token_id=1, use_cache=False))


def tiny_inputs(root: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    source, ordered, shuffled = root / "source", root / "ordered", root / "shuffled"
    for path in (source, ordered, shuffled):
        path.mkdir()
    vocabulary = ["<unk>", "<bos>", "<eos>", "Sequential", "events", ":"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(vocabulary)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 1)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<bos>",
                                       eos_token="<eos>", model_max_length=64)
    tokenizer.save_pretrained(source)
    torch.manual_seed(12)
    tiny_model().save_pretrained(source)
    for split in ("train", "val", "test"):
        sequences = [list("ABCDEFGHIJKLMNOPQRST") if i % 2 else list("TSRQPONMLKJIHGFEDCBA") for i in range(8)]
        for directory, values in ((ordered, sequences), (shuffled, [x[3:] + x[:3] for x in sequences])):
            pd.DataFrame({"Sequences": [runner.SEP.join(x) for x in values]}).to_csv(
                directory / f"X_{split}_kip_m4.csv", index=False)
            pd.DataFrame({"Outcome": np.arange(8) % 2}).to_csv(directory / f"y_{split}_kip_m4.csv", index=False)
    return source, ordered, shuffled


def arguments(root, source, ordered, shuffled, output_name="run"):
    return runner.parse_args([
        "--model-name", "test-tiny-llama", "--model-path", str(source), "--tag", "kip_m4",
        "--mode", "ordered", "--ordered-dir", str(ordered), "--shuffled-dir", str(shuffled),
        "--output-dir", str(root / output_name), "--device", "cpu", "--precision", "fp32",
        "--smoke", "--epochs", "2", "--threads", "1", "--batch-size", "3",
        "--gradient-accumulation-steps", "2", "--skip-completed",
    ])


def test_full_fp32_gradients_optimizer_and_right_padding_pooling():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = tiny_model().float().eval()
    rows = [([1, 6, 7, 8], 0), ([1, 10, 11], 1)]
    batch = runner.DynamicPadding(2, 64)(rows)
    loss, logits = runner.classifier_loss(model, batch, torch.device("cpu"), "fp32")
    assert loss.dtype == torch.float32
    for i, row in enumerate(rows):
        single = runner.DynamicPadding(2, 64, multiple=1)([row])
        _, unpadded_logits = runner.classifier_loss(model, single, torch.device("cpu"), "fp32")
        torch.testing.assert_close(logits[i], unpadded_logits[0], rtol=1e-5, atol=1e-7)
    loss.backward()
    assert all(p.requires_grad and p.dtype == torch.float32 and p.grad is not None
               and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    optimizer.step()
    assert all(state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
               for state in optimizer.state.values())


def test_sample_correct_accumulation_including_partial_final_group():
    torch.manual_seed(19)
    features, labels = torch.randn(8, 4), torch.arange(8) % 2
    model = torch.nn.Linear(4, 2)
    reference = torch.nn.Linear(4, 2)
    reference.load_state_dict(model.state_dict())
    for index, start in enumerate(range(0, 8, 3)):
        stop, samples = runner.accumulation_samples(index, 3, 3, 8, 2)
        x, y = features[start:start + 3], labels[start:start + 3]
        (torch.nn.functional.cross_entropy(model(x), y) * len(x) / samples).backward()
        if index + 1 == stop:
            group_start = index // 2 * 6
            torch.nn.functional.cross_entropy(reference(features[group_start:start + len(x)]),
                                              labels[group_start:start + len(x)]).backward()
            for actual, expected in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad)
            model.zero_grad()
            reference.zero_grad()


def test_padding_token_cannot_appear_in_content_and_no_truncation():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": [[1, 2, 6] for _ in texts]}

    dataset = runner.TokenizedDataset([runner.SEP.join(["A"] * 20)], np.array([0]), Tokenizer(), 64)
    with pytest.raises(ValueError, match="inside unpadded content"):
        runner.validate_token_ids({"train": dataset}, 2)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        runner.TokenizedDataset([runner.SEP.join(["A"] * 20)], np.array([0]), Tokenizer(), 2)


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_selected_checkpoint_reload_and_strict_completed_skip(tmp_path, monkeypatch, cpu_offload):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    source, ordered, shuffled = tiny_inputs(tmp_path)
    args = arguments(tmp_path, source, ordered, shuffled)
    args.optimizer_offload = "cpu" if cpu_offload else "none"
    args.gradient_checkpointing = cpu_offload
    original_evaluate, calls, selected_state = runner.evaluate, 0, {}

    def controlled_validation(model, loader, device, precision):
        nonlocal calls, selected_state
        calls += 1
        loss, probabilities, labels = original_evaluate(model, loader, device, precision)
        if calls == 1:
            selected_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            loss = 0.2
        elif calls == 2:
            loss = 0.8
        return loss, probabilities, labels

    monkeypatch.setattr(runner, "evaluate", controlled_validation)
    metrics = runner.train(args)
    assert metrics["best_epoch"] == 1
    assert metrics["rows"] == {"train": 8, "val": 8, "test": 8}
    assert metrics["is_smoke"] and metrics["selected_by"] == "validation_loss"
    assert metrics["checkpoint_policy"] == runner.CHECKPOINT_POLICY
    assert metrics["test"]["threshold"] == metrics["shuffled_test"]["threshold"] == metrics["validation"]["threshold"]
    output = args.output_dir
    assert not list(output.glob("*.pt")) and not (output / ".best_model.staging").exists()
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    model = AutoModelForSequenceClassification.from_pretrained(output / "best_model", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(output / "best_model", local_files_only=True)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, selected_state[name], rtol=0, atol=0)
    sequences, labels, _ = runner.read_split(ordered, "kip_m4", "test")
    loader = torch.utils.data.DataLoader(runner.TokenizedDataset(sequences, labels, tokenizer, 64), batch_size=64,
                                        collate_fn=runner.DynamicPadding(tokenizer.pad_token_id, 64))
    _, probabilities, _ = original_evaluate(model, loader, torch.device("cpu"), "fp32")
    predictions = pd.read_csv(output / "test_predictions.csv")
    np.testing.assert_allclose(probabilities, predictions["probability"], rtol=1e-7, atol=1e-7)
    assert predictions["row_index"].tolist() == list(range(8))
    assert runner.train(args)["test"] == metrics["test"]
    args.lr *= 2
    with pytest.raises(ValueError, match="configuration differs"):
        runner.train(args)
    args.lr /= 2
    (output / "test_predictions.csv").write_text("broken")
    with pytest.raises(ValueError, match="artifact missing or changed"):
        runner.train(args)


def test_interrupted_runs_require_restart_and_smoke_can_omit_checkpoint(tmp_path, monkeypatch):
    source, ordered, shuffled = tiny_inputs(tmp_path)
    args = arguments(tmp_path, source, ordered, shuffled)
    original = runner.evaluate

    def interrupted(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(runner, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train(args)
    with pytest.raises(FileExistsError, match="unfinished run must restart"):
        runner.train(args)
    monkeypatch.setattr(runner, "evaluate", original)
    fresh = arguments(tmp_path, source, ordered, shuffled, "fresh")
    fresh.no_save_model = True
    metrics = runner.train(fresh)
    assert metrics["is_smoke"] and metrics["best_model"] is None
    assert not (fresh.output_dir / "best_model").exists()
    assert runner.train(fresh)["test"] == metrics["test"]


def test_production_rejects_partial_datasets(tmp_path):
    source, ordered, shuffled = tiny_inputs(tmp_path)
    args = arguments(tmp_path, source, ordered, shuffled)
    args.smoke = False
    with pytest.raises(ValueError, match="400000/50000/50000"):
        runner.train(args)
