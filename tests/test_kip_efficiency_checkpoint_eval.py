from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import evaluate_kip_efficiency_checkpoint as evaluator
from src.experiments.kip_encoder_training import SEP


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Linear(1, 2, bias=False)

    def forward(self, input_ids, attention_mask):
        feature = (input_ids[:, 1].float() * 2 - 5).unsqueeze(-1)
        return SimpleNamespace(logits=self.score(feature))


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": [[1, 3 if text.startswith("Sequential events: B") else 2] for text in texts]}


def source_run(tmp_path, monkeypatch):
    run, data = tmp_path / "run", tmp_path / "data"
    run.mkdir()
    data.mkdir()
    expected = {"train": 16, "val": 8, "test": 8}
    monkeypatch.setattr(evaluator, "EXPECTED_SOURCE_ROWS", expected)
    provenance, counts, index_hashes = {}, {}, {}
    for split, count in expected.items():
        labels = np.arange(count) % 2
        sequences = [SEP.join((["B"] if y else ["A"]) + list("CDEFGHIJKLMNOPQRSTU")) for y in labels]
        pd.DataFrame({"Sequences": sequences}).to_csv(data / f"X_{split}_kip_m4.csv", index=False)
        pd.DataFrame({"Outcome": labels}).to_csv(data / f"y_{split}_kip_m4.csv", index=False)
        _, _, provenance[split] = evaluator.read_split(data, "kip_m4", split)
        selected = np.arange(count // 2 if split == "train" else count)
        np.save(run / f"{split}_indices.npy", selected, allow_pickle=False)
        counts[split] = len(selected)
        index_hashes[split] = evaluator.sha256(run / f"{split}_indices.npy")
    config = {"arm": "qwen27b_qlora", "fraction": 0.5, "seed": 9950,
              "is_smoke": False, "tag": "kip_m4", "mode": "ordered",
              "selection": "minimum_validation_loss", "threshold_selection": "validation_f1",
              "data": provenance, "rows": counts, "index_sha256": index_hashes,
              "max_length": 64, "model_source": {"snapshot": str(tmp_path / "base")}}
    source_names = [f"src/experiments/{name}" for name in ("kip_efficiency_models.py", "kip_encoder_training.py",
                                                         "kip_llama_training.py", "kip_data_efficiency.py")]
    manifest = {"config": config, "source_sha256": {name: evaluator.sha256(evaluator.ROOT / name) for name in source_names},
                "model": {"trainable_parameter_names": ["score.weight"]}}
    (run / "manifest.json").write_text(json.dumps(manifest))
    history = [{"epoch": 1, "val_loss": 0.5, "epoch_seconds": 12},
               {"epoch": 2, "val_loss": 0.1, "epoch_seconds": 10}]
    (run / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    torch.save({"arm": config["arm"], "epoch": 2,
                "state_dict": {"score.weight": torch.tensor([[-1.0], [1.0]])}}, run / "best_trainable.pt")
    return run, manifest


def test_full_saved_checkpoint_evaluation_restores_exact_weights_and_preserves_source(tmp_path, monkeypatch):
    run, manifest = source_run(tmp_path, monkeypatch)
    source_hashes = {path: evaluator.sha256(path) for path in run.iterdir()}
    monkeypatch.setattr(evaluator, "model_provenance", lambda _: manifest["config"]["model_source"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None)
    loaded = TinyClassifier()
    monkeypatch.setattr(evaluator, "load_classifier", lambda *a, **kw: (loaded, TinyTokenizer(), {}))
    original_evaluate = evaluator.evaluate
    batches = []

    def cpu_evaluate(model, loader, device, precision):
        batches.append(len(loader.dataset))
        loader.pin_memory = False
        return original_evaluate(model, loader, torch.device("cpu"), "fp32")

    monkeypatch.setattr(evaluator, "evaluate", cpu_evaluate)
    args = evaluator.parse_args(["--run-dir", str(run), "--output-dir", str(tmp_path / "final_evaluation"),
                                 "--eval-batch-size", "3", "--threads", "1"])
    metrics = evaluator.run_evaluation(args)
    assert batches == [8, 8]
    assert metrics["rows"] == {"train": 8, "val": 8, "test": 8}
    assert metrics["test"]["auc"] == 1.0 and metrics["stopping_rule"]["met"]
    assert metrics["completion_method"] == evaluator.COMPLETION_METHOD
    assert metrics["test_informed_stopping"] and metrics["threshold_selected_on"] == "validation"
    assert metrics["best_epoch"] == metrics["epochs_completed"] == 2
    assert metrics["validation"]["threshold"] == metrics["test"]["threshold"]
    assert metrics["reference_checkpoint"]["sha256"] == source_hashes[run / "best_trainable.pt"]
    torch.testing.assert_close(loaded.score.weight, torch.tensor([[-1.0], [1.0]]), rtol=0, atol=0)
    for path, digest in source_hashes.items():
        assert evaluator.sha256(path) == digest
    for relative, digest in metrics["artifact_sha256"].items():
        assert not relative.startswith("/") and evaluator.sha256(args.output_dir / relative) == digest
    assert pd.read_csv(args.output_dir / "test_predictions.csv")["row_index"].tolist() == list(range(8))
    with pytest.raises(FileExistsError, match="fresh empty"):
        evaluator.run_evaluation(args)


@pytest.mark.parametrize("mutation,match", [("indices", "row index hash"), ("data", "data hash"),
                                           ("arm", "arm/epoch"), ("epoch", "minimum-validation-loss")])
def test_saved_checkpoint_evaluation_rejects_source_or_checkpoint_mismatch(tmp_path, monkeypatch, mutation, match):
    run, manifest = source_run(tmp_path, monkeypatch)
    checkpoint_path = run / "best_trainable.pt"
    if mutation == "indices":
        np.save(run / "test_indices.npy", np.arange(7), allow_pickle=False)
    elif mutation == "data":
        with open(manifest["config"]["data"]["test"]["labels"]["path"], "a") as handle:
            handle.write("0\n")
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        checkpoint[mutation] = "llama8b_lora" if mutation == "arm" else 1
        torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match=match):
        evaluator.verify_run(run, checkpoint_path)


def test_run_directory_cannot_be_evaluation_output(tmp_path, monkeypatch):
    run, _ = source_run(tmp_path, monkeypatch)
    args = evaluator.parse_args(["--run-dir", str(run), "--output-dir", str(run)])
    with pytest.raises(ValueError, match="outside"):
        evaluator.run_evaluation(args)
