from __future__ import annotations

import numpy as np
import pytest

from src.mimic.aki.models import train_sequence_classifier, train_tabular_classifier


def _sequence_config() -> dict:
    return {
        "models": {
            "sequence_preprocessing": {"scaling": "standard", "missing_values": "error"},
            "sequence_training": {
                "batch_size": 2,
                "max_epochs": 2,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "optimizer": "adam",
                "optimizer_parameters": {},
                "patience": 1,
                "min_delta": 0.0,
                "early_stopping_metric": "validation_loss",
                "device": "cpu",
                "num_workers": 0,
                "pin_memory": False,
                "deterministic_algorithms": True,
                "class_weight": None,
                "gradient_clip_norm": None,
            },
            "lstm": {
                "hidden_dim": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "projection_dim": None,
            },
            "transformer": {
                "d_model": 4,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 8,
                "dropout": 0.0,
                "max_sequence_length": 8,
                "positional_encoding": "sinusoidal",
                "pooling": "mean",
            },
        }
    }


def _sequences() -> tuple[list[np.ndarray], np.ndarray]:
    values = [
        [1.0, 1.4, 1.0],
        [1.0, 1.5, 1.1],
        [0.9, 1.3, 0.9],
        [1.0, 1.4, 1.5],
        [1.1, 1.5, 1.6],
        [0.9, 1.3, 1.4],
    ]
    sequences = [
        np.column_stack([row, np.arange(len(row), dtype=float)]).astype(np.float32)
        for row in values
    ]
    labels = np.asarray(["transient"] * 3 + ["persistent"] * 3)
    return sequences, labels


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ["lstm", "transformer"])
def test_real_sequence_backend_smoke(model_name: str) -> None:
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"working PyTorch is unavailable: {exc}")
    sequences, labels = _sequences()

    fitted = train_sequence_classifier(
        model_name,
        sequences,
        labels,
        sequences,
        labels,
        _sequence_config(),
        seed=17,
        class_labels=("transient", "persistent"),
    )
    probabilities = fitted.predict_proba(sequences)

    assert probabilities.shape == (6, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_real_xgboost_backend_smoke() -> None:
    try:
        import xgboost  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"XGBoost is unavailable: {exc}")
    features = np.asarray([[1.0, 0.0], [1.1, 0.1], [0.9, -0.1], [1.5, 0.5], [1.6, 0.6], [1.4, 0.4]])
    labels = np.asarray(["transient"] * 3 + ["persistent"] * 3)
    config = {
        "models": {
            "tabular_preprocessing": {"scaling": "standard", "missing_values": "error"},
            "xgboost": {
                "parameters": {
                    "n_estimators": 2,
                    "max_depth": 1,
                    "learning_rate": 0.2,
                    "min_child_weight": 1,
                    "subsample": 1.0,
                    "colsample_bytree": 1.0,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "n_jobs": 1,
                    "tree_method": "hist",
                    "verbosity": 0,
                },
                "fit_parameters": {},
                "class_weight": None,
            },
        }
    }

    fitted = train_tabular_classifier(
        "xgboost",
        features,
        labels,
        config,
        seed=19,
        class_labels=("transient", "persistent"),
        validation_data=(features, labels),
    )
    probabilities = fitted.predict_proba(features)

    assert probabilities.shape == (6, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
