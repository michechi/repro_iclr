import numpy as np
import pytest

from src.mimic.aki.models import _resolve_class_weights


def test_one_explicit_weight_map_supports_binary_and_multiclass_tasks() -> None:
    labels = ("transient", "persistent", "relapsing")
    weights = {"transient": 1.0, "persistent": 2.0, "relapsing": 3.0}

    binary = _resolve_class_weights(
        np.asarray([0, 1, 0]),
        labels[:2],
        weights,
        allowed_labels=labels,
    )
    multiclass = _resolve_class_weights(
        np.asarray([0, 1, 2]),
        labels,
        weights,
        allowed_labels=labels,
    )

    np.testing.assert_array_equal(binary, [1.0, 2.0])
    np.testing.assert_array_equal(multiclass, [1.0, 2.0, 3.0])


def test_explicit_weight_map_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="extra=.*unknown"):
        _resolve_class_weights(
            np.asarray([0, 1]),
            ("transient", "persistent"),
            {"transient": 1.0, "persistent": 1.0, "unknown": 1.0},
            allowed_labels=("transient", "persistent", "relapsing"),
        )
