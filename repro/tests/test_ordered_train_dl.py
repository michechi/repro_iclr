"""Focused tests for the persistent Ordered Compliance scratch trainer."""

from __future__ import annotations

import numpy as np
import torch

from src.oc_completion import ordered_train_dl as trainer
from src.oc_completion.oracle import SEP


def _run_metadata(model_name: str = "LSTM") -> dict:
    recipe = trainer.PAPER_RECIPES[model_name]
    return {
        "model_name": model_name,
        "model_config": recipe["model_config"],
        "noise_pi": 0.1,
        "model_seed": 9550,
        "data_fingerprints": {"train": "a", "val": "b", "test": "c"},
        "split_rows": {"train": 4, "val": 2, "test": 2},
        "batch_size": 64,
        "optimizer": "AdamW",
        "learning_rate": recipe["learning_rate"],
        "encoding": trainer.ENCODING,
    }


def test_paper_scratch_recipes_are_the_requested_anchor_configs():
    assert trainer.BATCH_SIZE == 64
    assert trainer.DEFAULT_MAX_EPOCHS == 30
    assert trainer.DEFAULT_PATIENCE == 5

    lstm = trainer.PAPER_RECIPES["LSTM"]
    assert lstm["model_config"] == {
        "vocab_size": 27,
        "embedding_dim": 32,
        "hidden_dim": 128,
        "num_layers": 2,
        "num_classes": 2,
        "dropout": 0.2,
    }
    assert lstm["learning_rate"] == 2e-3

    transformer = trainer.PAPER_RECIPES["Transformer"]
    assert transformer["model_config"]["embedding_dim"] == 64
    assert transformer["model_config"]["dim_feedforward"] == 128
    assert transformer["model_config"]["num_layers"] == 3
    assert transformer["model_config"]["dropout"] == 0.3
    assert transformer["learning_rate"] == 5e-4

    hybrid = trainer.PAPER_RECIPES["RNNTransformer"]
    assert hybrid["model_config"]["embedding_dim"] == 128
    assert hybrid["model_config"]["rnn_hidden_dim"] == 32
    assert hybrid["paper_spec"]["recurrent_layers"] == 1
    assert hybrid["model_config"]["dim_feedforward"] == 256
    assert hybrid["model_config"]["num_transformer_layers"] == 3
    assert hybrid["model_config"]["dropout"] == 0.3
    assert hybrid["learning_rate"] == 1e-4

    sample = trainer.encode_complete_sequence("ABCDEFGHIJKLMNOPQRST").unsqueeze(0)
    for model_name in trainer.MODEL_NAMES:
        model = trainer.build_scratch_model(model_name).eval()
        with torch.inference_mode():
            assert model(sample).shape == (1, 2)


def test_complete_encoder_accepts_compact_and_sep_forms_identically():
    compact = "ABCDEFGHIJKLMNOPQRST"
    separated = SEP.join(compact)
    compact_ids = trainer.encode_complete_sequence(compact)
    separated_ids = trainer.encode_complete_sequence(separated)

    assert torch.equal(compact_ids, separated_ids)
    assert compact_ids.shape == (20,)
    assert compact_ids.min().item() >= 1
    assert not compact_ids.eq(trainer.PAD_ID).any()


def test_validation_threshold_is_selected_only_from_labels_and_probabilities():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.array([0.05, 0.30, 0.40, 0.90], dtype=np.float64)

    threshold, score = trainer.validation_f1_threshold(labels, probabilities)

    assert threshold == 0.40
    assert score == 1.0


def test_resumable_checkpoint_reload_and_scorer_reproduce_logits(tmp_path):
    trainer.set_model_seed(9550)
    model = trainer.build_scratch_model("LSTM")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer.PAPER_RECIPES["LSTM"]["learning_rate"],
    )
    generator = torch.Generator().manual_seed(9550)
    inputs = torch.stack(
        [
            trainer.encode_complete_sequence("ABCDEFGHIJKLMNOPQRST"),
            trainer.encode_complete_sequence("ZABCDEFGHIJKLMNOPQRS"),
        ]
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(inputs), labels)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        reference = model(inputs).float()

    payload = trainer.checkpoint_payload(
        role="best_standard",
        model=model,
        optimizer=optimizer,
        loader_generator=generator,
        run_metadata=_run_metadata(),
        current_epoch=1,
        best_validation_f1=0.75,
        best_epoch=1,
        best_validation_threshold=0.42,
        patience_counter=0,
        history=[{"epoch": 1}],
        training_time_seconds=1.25,
    )
    checkpoint_path = tmp_path / "best_standard.pt"
    trainer.atomic_torch_save(payload, checkpoint_path)

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["optimizer_state_dict"]
    assert saved["scheduler_state_dict"] is None
    assert saved["gradient_scaler_state_dict"] is None
    assert saved["current_epoch"] == 1
    assert saved["best_validation_f1"] == 0.75
    assert saved["patience_counter"] == 0
    assert set(saved["rng_state"]) == {
        "python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"
    }

    difference = trainer.verify_checkpoint_logits(
        checkpoint_path, inputs, reference, device="cpu"
    )
    assert difference == 0.0

    resumed_model = trainer.build_scratch_model("LSTM")
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=trainer.PAPER_RECIPES["LSTM"]["learning_rate"],
    )
    resumed_generator = torch.Generator().manual_seed(1)
    restored = trainer.restore_training_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        loader_generator=resumed_generator,
        device=torch.device("cpu"),
        expected_metadata=_run_metadata(),
    )
    resumed_model.eval()
    with torch.inference_mode():
        resumed_logits = resumed_model(inputs).float()
    torch.testing.assert_close(resumed_logits, reference, atol=1e-6, rtol=1e-5)
    assert restored["current_epoch"] == 1
    assert resumed_optimizer.state_dict()["state"]
    assert torch.equal(
        resumed_generator.get_state(), saved["rng_state"]["loader_generator"]
    )

    scorer, metadata = trainer.load_scratch_scorer(
        checkpoint_path, device="cpu", batch_size=1
    )
    scored = scorer(
        ["ABCDEFGHIJKLMNOPQRST", SEP.join("ABCDEFGHIJKLMNOPQRST")]
    )
    assert scored.shape == (2, 2)
    assert scored.dtype == np.float32
    np.testing.assert_array_equal(scored[0], scored[1])
    # The scorer intentionally changes the batch shape here (1 versus 2), so
    # BLAS may differ in the last few FP32 bits while remaining numerically
    # equivalent.
    np.testing.assert_allclose(scored[0], reference[0].numpy(), atol=1e-7, rtol=1e-6)
    assert metadata["model_name"] == "LSTM"
    assert metadata["model_seed"] == 9550
    assert metadata["noise_pi"] == 0.1
    assert metadata["checkpoint_epoch"] == 1
    assert metadata["validation_selected_threshold"] == 0.42
    assert metadata["positive_class_index"] == 1


def test_epoch_snapshot_is_model_only_and_evaluation_ready():
    model = trainer.build_scratch_model("LSTM")
    snapshot = trainer.epoch_snapshot_payload(
        model=model,
        run_metadata=_run_metadata(),
        epoch=3,
        validation_metrics={"observed_f1": 0.7},
    )

    assert snapshot["checkpoint_role"] == "epoch_evaluation_snapshot"
    assert snapshot["epoch"] == 3
    assert snapshot["model_state_dict"]
    assert "optimizer_state_dict" not in snapshot
    assert snapshot["standard_validation"] == {"observed_f1": 0.7}


def test_random_untrained_scorer_is_seeded_fp32_and_does_not_move_torch_rng():
    before = torch.get_rng_state().clone()
    scorer_a, metadata = trainer.make_random_scorer(
        "Transformer", 9550, device="cpu", batch_size=2
    )
    after = torch.get_rng_state()
    scorer_b, _ = trainer.make_random_scorer(
        "Transformer", 9550, device="cpu", batch_size=1
    )
    sequences = ["ABCDEFGHIJKLMNOPQRST", "TSRQPONMLKJIHGFEDCBA"]

    logits_a = scorer_a(sequences)
    logits_b = scorer_b(sequences)

    assert torch.equal(before, after)
    assert logits_a.dtype == np.float32
    np.testing.assert_allclose(logits_a, logits_b, atol=1e-6, rtol=1e-5)
    assert metadata["checkpoint_role"] == "random_untrained_control"
    assert metadata["checkpoint_epoch"] == 0
    assert metadata["model_seed"] == 9550
