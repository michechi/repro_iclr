import pandas as pd

from src.mimic.aki.sensitivity import build_recovery_timing_sensitivity


def _config() -> dict:
    return {
        "episodes": {
            "recovery_timing_sensitivity": {
                "enabled": True,
                "reference_time_column": "peak_time",
                "baseline_column": "baseline_value",
                "threshold_above_baseline_mg_dl": 0.3,
                "value_comparison": "lt",
                "fast_max_hours": 48,
                "intermediate_max_hours": 240,
                "time_boundary": "inclusive",
                "sustained_hours": 0,
                "max_gap_hours": 24,
                "endpoint_tolerance_hours": 0,
                "minimum_measurements": 1,
                "duplicate_policy": "error",
            }
        }
    }


def test_peak_anchored_recovery_sensitivity_marks_all_timing_outcomes() -> None:
    start = pd.Timestamp("2025-01-01")
    episodes = pd.DataFrame(
        {
            "episode_id": ["fast", "intermediate", "none", "censored"],
            "subject_id": [1, 2, 3, 4],
            "hadm_id": [11, 22, 33, 44],
            "phenotype": ["transient", "persistent", "persistent", "persistent"],
            "peak_time": [start] * 4,
            "baseline_value": [1.0] * 4,
        }
    )
    rows = []
    for subject_id, hadm_id, recovery_hour, last_hour in (
        (1, 11, 24, 240),
        (2, 22, 72, 240),
        (3, 33, None, 240),
        (4, 44, None, 120),
    ):
        for hour in range(0, last_hour + 1, 24):
            value = 1.4 if recovery_hour is None or hour < recovery_hour else 1.0
            rows.append(
                {
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "specimen_time": start + pd.Timedelta(hours=hour),
                    "creatinine_mg_dl": value,
                }
            )

    result = build_recovery_timing_sensitivity(pd.DataFrame(rows), episodes, _config())
    categories = result.set_index("episode_id")["recovery_timing_category"].to_dict()

    assert categories == {
        "fast": "fast",
        "intermediate": "intermediate",
        "none": "no_recovery_by_intermediate_horizon",
        "censored": "censored",
    }
