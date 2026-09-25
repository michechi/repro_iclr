import pandas as pd

from src.mimic.aki.splits import attach_patient_splits, make_patient_splits


def _config() -> dict:
    return {
        "splitting": {
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "split_seed": 2026,
            "stratification": "single_label",
            "rare_stratum_policy": "error",
        }
    }


def test_patient_splits_are_reproducible_and_do_not_overlap() -> None:
    episodes = pd.DataFrame(
        {
            "subject_id": range(60),
            "episode_id": [f"e{i}" for i in range(60)],
            "label": [["transient", "persistent", "relapsing"][i % 3] for i in range(60)],
        }
    )
    first = make_patient_splits(episodes, _config())
    second = make_patient_splits(episodes.sample(frac=1, random_state=9), _config())
    pd.testing.assert_frame_equal(
        first.sort_values("subject_id").reset_index(drop=True),
        second.sort_values("subject_id").reset_index(drop=True),
    )
    split_sets = {
        name: set(first.loc[first["split"] == name, "subject_id"])
        for name in ("train", "validation", "test")
    }
    assert split_sets["train"].isdisjoint(split_sets["validation"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["validation"].isdisjoint(split_sets["test"])


def test_all_patient_episodes_receive_the_same_split() -> None:
    subjects = list(range(60))
    primary = pd.DataFrame(
        {
            "subject_id": subjects,
            "episode_id": [f"first-{i}" for i in subjects],
            "label": [["transient", "persistent", "relapsing"][i % 3] for i in subjects],
        }
    )
    splits = make_patient_splits(primary, _config())
    repeated = pd.concat(
        [
            primary,
            primary.assign(episode_id=lambda frame: "second-" + frame["subject_id"].astype(str)),
        ],
        ignore_index=True,
    )
    attached = attach_patient_splits(repeated, splits)
    assert attached.groupby("subject_id")["split"].nunique().eq(1).all()
