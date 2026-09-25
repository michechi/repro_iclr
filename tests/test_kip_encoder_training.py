from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.experiments import kip_encoder_training as runner


def test_dynamic_padding_keeps_every_token_and_masks_only_padding():
    rows = [([2, 8, 9, 3], 1), ([2, 7, 3], 0)]
    batch = runner.DynamicPadding(pad_token_id=0, max_length=64)(rows)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["input_ids"][0, :4].tolist() == rows[0][0]
    assert batch["input_ids"][1, :3].tolist() == rows[1][0]
    assert batch["attention_mask"].sum(dim=1).tolist() == [4, 3]
    assert batch["labels"].tolist() == [1, 0]
    with pytest.raises(ValueError, match="truncation"):
        runner.DynamicPadding(0, 3)(rows)


def test_pretokenization_rejects_length_overflow_instead_of_truncating():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            assert kwargs["padding"] is False
            return {"input_ids": [[1] * 65 for _ in texts]}

    with pytest.raises(ValueError, match="truncation is forbidden"):
        runner.TokenizedDataset([runner.SEP.join(["A"] * 20)], np.array([0]), Tokenizer(), 64)


def test_prompt_rejects_dropped_events():
    assert runner.sequence_prompt(runner.SEP.join(["A"] * 20)) == "Sequential events: " + " ".join(["A"] * 20)
    with pytest.raises(ValueError, match="exactly 20"):
        runner.sequence_prompt(runner.SEP.join(["A"] * 19))


def test_early_stopping_is_strict_validation_loss_selection():
    stopper = runner.EarlyStopping(patience=2)
    assert stopper.update(0.6, 1)
    assert not stopper.update(0.6, 2)
    assert not stopper.stopped
    assert stopper.update(0.5, 3)
    assert not stopper.update(0.7, 4)
    assert not stopper.update(0.8, 5)
    assert stopper.stopped
    assert stopper.best_epoch == 3
    assert stopper.best_loss == 0.5
    with pytest.raises(ValueError, match="non-finite"):
        stopper.update(float("nan"), 6)


def test_smoke_subsampling_is_stratified_and_keeps_original_indices():
    labels = np.array([0] * 80 + [1] * 20)
    selected = runner.subset_indices(labels, 10, 9950)
    assert len(selected) == 10
    assert labels[selected].sum() == 2
    assert np.array_equal(selected, np.sort(selected))
    assert np.array_equal(selected, runner.subset_indices(labels, 10, 9950))


def _tiny_local_inputs(root: Path) -> tuple[Path, Path, Path]:
    transformers = pytest.importorskip("transformers")
    source = root / "pretrained"
    source.mkdir()
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sequential", "events", ":"] + list("abcdefghijklmnopqrstuvwxyz")
    tokenizer = transformers.BertTokenizer(vocab={token: index for index, token in enumerate(vocabulary)},
                                          model_max_length=64)
    tokenizer.save_pretrained(source)
    torch.manual_seed(12)
    model = transformers.BertForSequenceClassification(transformers.BertConfig(
        vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_hidden_layers=1, max_position_embeddings=64, num_labels=2))
    model.save_pretrained(source)
    ordered, shuffled = root / "ordered", root / "shuffled"
    ordered.mkdir()
    shuffled.mkdir()
    for split in ("train", "val", "test"):
        sequences = [list("ABCDEFGHIJKLMNOPQRST") if index % 2 else list("TSRQPONMLKJIHGFEDCBA")
                     for index in range(8)]
        for directory, values in [(ordered, sequences), (shuffled, [sequence[3:] + sequence[:3] for sequence in sequences])]:
            pd.DataFrame({"Sequences": [runner.SEP.join(sequence) for sequence in values]}).to_csv(
                directory / f"X_{split}_kip_m4.csv", index=False)
            pd.DataFrame({"Outcome": np.arange(8) % 2}).to_csv(directory / f"y_{split}_kip_m4.csv", index=False)
    return source, ordered, shuffled


@pytest.mark.parametrize("tuning", ["lora", "full"])
def test_epoch_resume_matches_uninterrupted_training_and_saves_selected_model(tmp_path, monkeypatch, tuning):
    if tuning == "lora":
        pytest.importorskip("peft")
    source, ordered, shuffled = _tiny_local_inputs(tmp_path)

    def arguments(output: Path):
        return runner.parse_args([
            "--model-name", "test-tiny-bert", "--model-path", str(source),
            "--tag", "kip_m4", "--mode", "ordered", "--tuning", tuning,
            "--ordered-dir", str(ordered), "--shuffled-dir", str(shuffled),
            "--output-dir", str(output), "--cache-dir", str(tmp_path / "cache"),
            "--device", "cpu", "--precision", "fp32", "--epochs", "2", "--threads", "1",
            "--batch-size", "3", "--gradient-accumulation-steps", "2", "--resume",
        ])

    complete_dir, resumed_dir = tmp_path / "complete", tmp_path / "resumed"
    baseline = runner.train(arguments(complete_dir))
    original_evaluate = runner.evaluate
    validation_calls = 0

    def interrupted_evaluate(*args, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise RuntimeError("simulated interruption before second epoch checkpoint")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate", interrupted_evaluate)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.train(arguments(resumed_dir))
    assert torch.load(resumed_dir / "last.pt", weights_only=False)["epoch"] == 1
    monkeypatch.setattr(runner, "evaluate", original_evaluate)
    resumed = runner.train(arguments(resumed_dir))
    assert resumed["best_epoch"] == baseline["best_epoch"]
    assert resumed["validation"] == baseline["validation"]
    assert resumed["test"] == baseline["test"]
    assert resumed["shuffled_test"] == baseline["shuffled_test"]
    for split in ("test", "shuffled_test"):
        pd.testing.assert_frame_equal(pd.read_csv(complete_dir / f"{split}_predictions.csv"),
                                      pd.read_csv(resumed_dir / f"{split}_predictions.csv"))
    final_checkpoint = torch.load(resumed_dir / "last.pt", weights_only=False)
    assert final_checkpoint["optimizer_steps"] == 4
    assert json.loads((resumed_dir / "status.json").read_text())["status"] == "completed"
    model_filename = "adapter_model.safetensors" if tuning == "lora" else "model.safetensors"
    assert (resumed_dir / "best_model" / model_filename).is_file()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    selected_dir = resumed_dir / "best_model"
    if tuning == "lora":
        from peft import PeftModel
        reloaded_model = PeftModel.from_pretrained(
            AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True),
            selected_dir, local_files_only=True)
    else:
        reloaded_model = AutoModelForSequenceClassification.from_pretrained(selected_dir, local_files_only=True)
    reloaded_tokenizer = AutoTokenizer.from_pretrained(selected_dir, local_files_only=True)
    sequences, labels, _ = runner.read_split(ordered, "kip_m4", "test")
    reload_loader = torch.utils.data.DataLoader(
        runner.TokenizedDataset(sequences, labels, reloaded_tokenizer, 64), batch_size=128,
        collate_fn=runner.DynamicPadding(reloaded_tokenizer.pad_token_id, 64))
    _, probabilities, _ = runner.evaluate(reloaded_model, reload_loader, torch.device("cpu"), "fp32")
    np.testing.assert_allclose(probabilities, pd.read_csv(resumed_dir / "test_predictions.csv")["probability"],
                               rtol=1e-7, atol=1e-7)
    # A completed invocation is resumable/idempotent, but a changed recipe is rejected.
    assert runner.train(arguments(resumed_dir))["test"] == resumed["test"]
    changed = arguments(resumed_dir)
    changed.lr *= 2
    with pytest.raises(ValueError, match="resume configuration differs"):
        runner.train(changed)
