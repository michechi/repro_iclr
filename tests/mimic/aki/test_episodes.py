import copy

import pandas as pd
import pytest

from src.mimic.aki.episodes import construct_aki_episodes


_START = pd.Timestamp("2025-01-01 00:00:00")
_DENSE_HOURS = list(range(0, 193, 12))


def _episode_config() -> dict:
    """A fully explicit synthetic protocol; no production choice is inferred."""

    return {
        "episodes": {
            "columns": {
                "timestamp": "specimen_time",
                "value": "creatinine_mg_dl",
                "subject_id": "subject_id",
                "hadm_id": "hadm_id",
            },
            "admission_bounded": True,
            "baseline": {
                "method": "minimum",
                "lookback_hours": 48,
                "minimum_measurements": 1,
            },
            "onset": {
                "absolute_rise_mg_dl": 0.3,
                "window_hours": 48,
                "boundary": "inclusive",
                "tie_breaker": "earliest",
            },
            "recovery": {
                "threshold_above_baseline_mg_dl": 0.3,
                "window_hours": 48,
                "time_boundary": "inclusive",
                "value_comparison": "lt",
                "sustained_hours": 48,
            },
            "follow_up": {"duration_days": 7, "boundary": "inclusive"},
            "peak": {
                "window_end": "follow_up",
                "boundary": "inclusive",
                "tie_breaker": "earliest",
            },
            "relapse": {
                "window_days": 7,
                "anchor": "onset",
                "comparator": "recovery_nadir",
                "boundary": "inclusive",
            },
            "observation": {
                "max_gap_hours": 24,
                "endpoint_tolerance_hours": 0,
                "minimum_post_onset_measurements": 1,
            },
            "duplicates": {
                "exact": "drop",
                "conflicting_timestamp": "ambiguous",
            },
        }
    }


def _measurements(values: list[float], hours: list[int] | None = None) -> pd.DataFrame:
    if hours is None:
        hours = _DENSE_HOURS
    assert len(values) == len(hours)
    return pd.DataFrame(
        {
            "subject_id": 101,
            "hadm_id": 1001,
            "specimen_time": [_START + pd.Timedelta(hours=hour) for hour in hours],
            "creatinine_mg_dl": values,
        }
    )


def _transient_values(*, recovery_hour: int = 36) -> list[float]:
    return [1.0 if hour < 24 or hour >= recovery_hour else 1.4 for hour in _DENSE_HOURS]


def test_transient_requires_sustained_early_recovery_and_complete_follow_up() -> None:
    result = construct_aki_episodes(_measurements(_transient_values()), _episode_config())

    assert len(result.episodes) == 1
    episode = result.episodes.iloc[0]
    assert episode["phenotype"] == "transient"
    assert episode["status"] == "labeled"
    assert bool(episode["include_in_analysis"])
    assert episode["onset_time"] == _START + pd.Timedelta(hours=24)
    assert episode["recovery_time"] == _START + pd.Timedelta(hours=36)
    assert episode["recovery_confirmed_time"] == _START + pd.Timedelta(hours=84)
    assert episode["reason_code"] == "recovery_within_window_without_recurrence"
    assert bool(result.audit.iloc[0]["observation_adequate"])


def test_persistent_when_no_sustained_recovery_occurs_by_48_hours() -> None:
    values = [1.0 if hour < 24 else 1.4 for hour in _DENSE_HOURS]
    result = construct_aki_episodes(_measurements(values), _episode_config())

    episode = result.episodes.iloc[0]
    assert episode["phenotype"] == "persistent"
    assert pd.isna(episode["recovery_time"])
    assert pd.isna(episode["recurrence_time"])
    assert episode["reason_code"] == "no_sustained_recovery_within_window"
    assert bool(episode["include_in_analysis"])


def test_relapsing_takes_precedence_after_confirmed_recovery() -> None:
    values = []
    for hour in _DENSE_HOURS:
        if hour < 24:
            values.append(1.0)
        elif hour < 36:
            values.append(1.4)
        elif hour < 96:
            values.append(1.0)
        else:
            values.append(1.4)

    result = construct_aki_episodes(_measurements(values), _episode_config())

    episode = result.episodes.iloc[0]
    assert episode["phenotype"] == "relapsing"
    assert episode["recovery_time"] == _START + pd.Timedelta(hours=36)
    assert episode["recovery_confirmed_time"] == _START + pd.Timedelta(hours=84)
    assert episode["recurrence_time"] == _START + pd.Timedelta(hours=96)
    assert episode["reason_code"] == "sustained_recovery_then_recurrence"
    assert "takes phenotype precedence" in result.audit.iloc[0]["reason_detail"]


def test_incomplete_admission_follow_up_is_censored_not_persistent() -> None:
    hours = list(range(0, 85, 12))
    values = [1.0 if hour < 24 else 1.4 for hour in hours]
    result = construct_aki_episodes(_measurements(values, hours), _episode_config())

    episode = result.episodes.iloc[0]
    audit = result.audit.iloc[0]
    assert episode["phenotype"] == "censored"
    assert episode["status"] == "censored"
    assert not bool(episode["include_in_analysis"])
    assert episode["reason_code"] == "follow_up_ends_before_required_horizon"
    assert not bool(audit["observation_adequate"])
    assert audit["required_observation_end_time"] == _START + pd.Timedelta(hours=192)


def test_exact_duplicate_is_removed_and_audited_without_changing_label() -> None:
    measurements = _measurements(_transient_values())
    measurements = pd.concat([measurements, measurements.iloc[[2]]], ignore_index=True)

    result = construct_aki_episodes(measurements, _episode_config())

    assert result.episodes.iloc[0]["phenotype"] == "transient"
    assert result.audit.iloc[0]["exact_duplicates_removed"] == 1
    assert result.audit.iloc[0]["conflicting_timestamp_groups"] == 0


def test_conflicting_same_timestamp_values_mark_admission_ambiguous() -> None:
    measurements = _measurements(_transient_values())
    conflict = measurements.iloc[[2]].copy()
    conflict["creatinine_mg_dl"] = 1.7
    measurements = pd.concat([measurements, conflict], ignore_index=True)

    result = construct_aki_episodes(measurements, _episode_config())

    assert len(result.episodes) == 1
    episode = result.episodes.iloc[0]
    audit = result.audit.iloc[0]
    assert episode["phenotype"] == "ambiguous"
    assert episode["status"] == "ambiguous"
    assert not bool(episode["include_in_analysis"])
    assert episode["reason_code"] == "conflicting_simultaneous_values"
    assert audit["conflicting_timestamp_groups"] == 1


def test_recovery_exactly_at_48_hours_obeys_configured_time_boundary() -> None:
    measurements = _measurements(_transient_values(recovery_hour=72))

    inclusive = construct_aki_episodes(measurements, _episode_config())
    assert inclusive.episodes.iloc[0]["recovery_time"] == _START + pd.Timedelta(hours=72)
    assert inclusive.episodes.iloc[0]["phenotype"] == "transient"

    exclusive_config = copy.deepcopy(_episode_config())
    exclusive_config["episodes"]["recovery"]["time_boundary"] = "exclusive"
    exclusive = construct_aki_episodes(measurements, exclusive_config)
    assert exclusive.episodes.iloc[0]["phenotype"] == "persistent"


def test_peak_ties_obey_configured_tie_breaker_and_are_audited() -> None:
    values = [1.0 if hour < 24 else 1.4 for hour in _DENSE_HOURS]
    config = _episode_config()
    config["episodes"]["peak"]["tie_breaker"] = "latest"

    result = construct_aki_episodes(_measurements(values), config)

    assert result.episodes.iloc[0]["peak_time"] == _START + pd.Timedelta(hours=192)
    assert result.audit.iloc[0]["peak_tie_breaker"] == "latest"
    assert result.audit.iloc[0]["peak_window_end"] == "follow_up"


@pytest.mark.parametrize(
    ("relapse_boundary", "expected_phenotype"),
    [("inclusive", "relapsing"), ("exclusive", "transient")],
)
def test_recurrence_exactly_at_day_seven_obeys_configured_boundary(
    relapse_boundary: str, expected_phenotype: str
) -> None:
    values = _transient_values()
    values[_DENSE_HOURS.index(192)] = 1.4
    config = _episode_config()
    config["episodes"]["relapse"]["boundary"] = relapse_boundary

    result = construct_aki_episodes(_measurements(values), config)

    assert result.episodes.iloc[0]["phenotype"] == expected_phenotype
    if relapse_boundary == "inclusive":
        assert result.episodes.iloc[0]["recurrence_time"] == _START + pd.Timedelta(hours=192)
    else:
        assert pd.isna(result.episodes.iloc[0]["recurrence_time"])
