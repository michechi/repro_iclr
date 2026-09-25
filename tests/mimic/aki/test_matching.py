import pandas as pd
import pytest

from src.mimic.aki.matching import coarsened_exact_match, select_first_eligible_episode


def _config() -> dict:
    return {
        "matching": {
            "class_a": "transient",
            "class_b": "persistent",
            "selection_seed": 9550,
            "coarsening": {
                "count": {"method": "exact"},
                "baseline": {"method": "width", "width": 0.5, "origin": 0.0},
                "peak": {"method": "width", "width": 1.0, "origin": 0.0},
                "duration_hours": {
                    "method": "edges",
                    "edges": [0, 72, 168, 1000],
                    "right": True,
                },
            },
            "minimum_pairs_per_split": {"train": 1, "validation": 1, "test": 1},
            "maximum_absolute_smd": 10.0,
        }
    }


def _features() -> pd.DataFrame:
    rows = []
    episode = 0
    for split in ("train", "validation", "test"):
        for pair in range(3):
            for label in ("transient", "persistent"):
                rows.append(
                    {
                        "episode_id": f"e{episode}",
                        "subject_id": episode,
                        "label": label,
                        "split": split,
                        "count": 5,
                        "baseline": 1.0 + 0.01 * pair,
                        "peak": 2.0 + 0.02 * pair,
                        "duration_hours": 120 + pair,
                    }
                )
                episode += 1
    return pd.DataFrame(rows)


def test_coarsened_matching_stays_within_splits_and_balances_classes() -> None:
    result = coarsened_exact_match(_features(), _config())
    assert result.estimable
    assert result.assignments.groupby("pair_id")["split"].nunique().eq(1).all()
    assert result.assignments.groupby("pair_id")["label"].nunique().eq(2).all()
    counts = result.matched.groupby(["split", "label"]).size().unstack(fill_value=0)
    assert counts["transient"].equals(counts["persistent"])
    assert result.flow["n_unmatched"].eq(0).all()


def test_first_eligible_episode_is_selected_per_subject() -> None:
    episodes = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 2],
            "episode_id": ["late", "early", "relapse", "primary"],
            "onset_time": pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-01", "2025-01-02"]),
            "label": ["persistent", "transient", "relapsing", "persistent"],
        }
    )
    selected = select_first_eligible_episode(episodes, eligible_labels={"transient", "persistent"})
    assert selected.set_index("subject_id")["episode_id"].to_dict() == {
        1: "early",
        2: "primary",
    }


def test_no_matching_strata_returns_auditable_empty_tables() -> None:
    features = _features()
    features.loc[features["label"] == "persistent", "count"] = 99

    result = coarsened_exact_match(features, _config())

    assert not result.estimable
    assert result.assignments.empty
    assert {"episode_id", "pair_id", "split", "label"}.issubset(result.assignments.columns)
    assert not result.flow.empty


def test_feature_specific_smd_override_is_used_for_estimability() -> None:
    features = _features()
    features["duration_hours"] = features["duration_hours"].astype(float)
    features.loc[features["label"] == "persistent", "duration_hours"] += 0.1
    config = _config()
    config["matching"]["maximum_absolute_smd"] = 0.1

    without_override = coarsened_exact_match(features, config)
    assert not without_override.estimable

    config["matching"]["maximum_absolute_smd_overrides"] = {"duration_hours": 0.15}
    with_override = coarsened_exact_match(features, config)

    assert with_override.estimable
    duration_rows = with_override.balance[with_override.balance["feature"] == "duration_hours"]
    assert duration_rows["maximum_absolute_smd"].eq(0.15).all()
    assert duration_rows["passes_threshold"].all()
    other_rows = with_override.balance[with_override.balance["feature"] != "duration_hours"]
    assert other_rows["maximum_absolute_smd"].eq(0.1).all()


def test_balance_reports_every_coarsening_feature_with_object_dtype() -> None:
    features = _features()
    configured_features = list(_config()["matching"]["coarsening"])
    for feature in configured_features:
        features[feature] = features[feature].astype("object")

    result = coarsened_exact_match(features, _config())

    expected = {
        (split, feature)
        for split in ("train", "validation", "test")
        for feature in configured_features
    }
    assert set(result.balance[["split", "feature"]].itertuples(index=False, name=None)) == expected
    assert result.balance["passes_threshold"].all()


def test_non_finite_post_match_smd_fails_estimability() -> None:
    features = _features().assign(site="ward-a")
    config = _config()
    config["matching"]["coarsening"]["site"] = {"method": "exact"}

    result = coarsened_exact_match(features, config)

    site_rows = result.balance[result.balance["feature"] == "site"]
    assert site_rows["smd_after"].isna().all()
    assert not site_rows["passes_threshold"].any()
    assert not result.estimable
    assert any("non-finite" in reason for reason in result.non_estimable_reasons)


def test_matching_rejects_smd_override_for_unconfigured_feature() -> None:
    config = _config()
    config["matching"]["maximum_absolute_smd_overrides"] = {"not_coarsened": 0.15}

    with pytest.raises(ValueError, match="configured matching.coarsening features"):
        coarsened_exact_match(_features(), config)
