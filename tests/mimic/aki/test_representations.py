from collections import Counter

import numpy as np
import pandas as pd

from src.mimic.aki.representations import (
    assert_value_multisets_preserved,
    build_representations,
    build_tabular_feature_sets,
    reverse_values_preserve_timestamps,
    shuffle_values_preserve_timestamps,
)


def _events() -> pd.DataFrame:
    base = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "episode_id": ["e1"] * 5 + ["e2"] * 4,
            "subject_id": [1] * 5 + [2] * 4,
            "hadm_id": [11] * 5 + [22] * 4,
            "specimen_time": [base + pd.Timedelta(hours=h) for h in (0, 6, 12, 24, 48)]
            + [base + pd.Timedelta(hours=h) for h in (0, 8, 16, 32)],
            "creatinine_mg_dl": [1.0, 1.4, 1.4, 1.1, 1.0, 0.8, 1.2, 0.9, 1.3],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": ["e1", "e2"],
            "subject_id": [1, 2],
            "hadm_id": [11, 22],
            "label": ["transient", "relapsing"],
            "baseline_creatinine_mg_dl": [1.0, 0.8],
            "onset_time": [pd.Timestamp("2025-01-01 06:00"), pd.Timestamp("2025-01-01 08:00")],
        }
    )


def _config() -> dict:
    return {
        "representations": {
            "max_length": 10,
            "truncation": "error",
            "value_transform": "raw",
            "time_features": ["elapsed_hours", "delta_hours", "time_since_onset_hours"],
            "summary_features": [
                "count",
                "distinct_count",
                "min",
                "max",
                "mean",
                "std",
                "range",
                "repeated_max_count",
                "duration_hours",
                "gap_mean_hours",
            ],
            "slope_hours_per_unit": 24,
        }
    }


def test_shuffle_preserves_timestamp_grid_count_and_value_multiset() -> None:
    events = _events()
    shuffled = shuffle_values_preserve_timestamps(events, seed=9550)
    assert_value_multisets_preserved(events, shuffled)

    for episode_id, original_group in events.groupby("episode_id"):
        shuffled_group = shuffled[shuffled["episode_id"] == episode_id]
        assert len(original_group) == len(shuffled_group)
        assert original_group["specimen_time"].tolist() == shuffled_group["specimen_time"].tolist()
        assert Counter(original_group["creatinine_mg_dl"]) == Counter(
            shuffled_group["creatinine_mg_dl"]
        )


def test_reversal_preserves_multiset_and_invariant_features() -> None:
    events = _events()
    reversed_events = reverse_values_preserve_timestamps(events)
    assert_value_multisets_preserved(events, reversed_events)

    original_features = build_tabular_feature_sets(events, _labels(), _config())[
        "invariant_summary"
    ]
    reversed_features = build_tabular_feature_sets(reversed_events, _labels(), _config())[
        "invariant_summary"
    ]
    feature_columns = [
        column
        for column in original_features
        if column not in {"episode_id", "subject_id", "hadm_id", "label"}
    ]
    np.testing.assert_allclose(
        original_features[feature_columns], reversed_features[feature_columns]
    )


def test_bundle_preserves_duplicate_values_and_channel_shapes() -> None:
    bundle = build_representations(_events(), _labels(), _config(), shuffle_seed=9551)
    assert len(bundle.ordered) == len(bundle.shuffled) == len(bundle.reversed) == 2
    assert bundle.ordered[0]["features"].shape == (5, 4)
    assert Counter(bundle.ordered[0]["values_mg_dl"]) == Counter(bundle.shuffled[0]["values_mg_dl"])
    assert set(bundle.tabular) == {
        "invariant_summary",
        "count_only",
        "first_value",
        "last_value",
        "first_to_last_change",
        "linear_slope",
        "position_combined",
    }


def test_truncation_selects_common_window_before_shuffle_and_reversal() -> None:
    config = _config()
    config["representations"]["max_length"] = 3
    config["representations"]["truncation"] = "head"

    bundle = build_representations(_events(), _labels(), config, shuffle_seed=19)

    for ordered, shuffled, reversed_record in zip(bundle.ordered, bundle.shuffled, bundle.reversed):
        assert ordered["timestamps"] == shuffled["timestamps"] == reversed_record["timestamps"]
        assert Counter(ordered["values_mg_dl"]) == Counter(shuffled["values_mg_dl"])
        assert Counter(ordered["values_mg_dl"]) == Counter(reversed_record["values_mg_dl"])
    counts = bundle.tabular["count_only"].set_index("episode_id")["count"]
    assert counts.to_dict() == {"e1": 5.0, "e2": 4.0}


def test_value_transform_is_shared_by_sequence_and_tabular_controls() -> None:
    config = _config()
    config["representations"]["value_transform"] = "delta_from_baseline"

    bundle = build_representations(_events(), _labels(), config, shuffle_seed=7)

    assert bundle.ordered[0]["features"][0, 0] == 0.0
    invariant = bundle.tabular["invariant_summary"].set_index("episode_id")
    assert invariant.loc["e1", "min"] == 0.0
    first = bundle.tabular["first_value"].set_index("episode_id")
    assert first.loc["e2", "first_value"] == 0.0
