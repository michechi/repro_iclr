"""Smoke contract for the shared-feature baseline evaluation grid."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.oc_completion import ordered_eval_baseline_grid as grid
from src.oc_completion.ordered_data import enumerate_candidate_labels
from src.oc_completion.ordered_features import build_features
from src.oc_completion.ordered_io import PI_LEVELS, pi_slug
from src.oc_completion.oracle import ALPHABET, N_EVENTS, SEP, oc_label_tokens
from src.oc_completion.strict_pairs import generate_strict_family


def _random_sequences(n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(ALPHABET)
    return [SEP.join(row) for row in
            alphabet[rng.integers(0, len(alphabet), size=(n, N_EVENTS))]]


def _hole_artifacts(result_root):
    tokens = []
    first = ["A"] * N_EVENTS
    tokens.append(first)
    second = list(first)
    second[0] = "W"
    tokens.append(second)
    third = list(second)
    third[7] = "D"
    tokens.append(third)
    fourth = ["B"] * N_EVENTS
    fourth[2], fourth[9] = "Q", "J"
    tokens.append(fourth)
    manifest = pd.DataFrame({
        "oracle_row": np.arange(len(tokens)),
        "base_sequence_id": [f"hole-{i}" for i in range(len(tokens))],
        "split": "hole_test",
        "X": [SEP.join(row) for row in tokens],
        "base_Y_star": [oc_label_tokens(row) for row in tokens],
    })
    oracle = np.stack([enumerate_candidate_labels(row) for row in tokens])
    manifest.to_parquet(result_root / "hole_manifest.parquet", index=False)
    np.savez_compressed(result_root / "hole_oracle_labels.npz",
                        oracle_labels=oracle)


def _strict_artifact(result_root):
    clean, _ = generate_strict_family(
        "clean", "test", 6, np.random.default_rng(301)
    )
    bases = _random_sequences(50, 302)
    heldout, _ = generate_strict_family(
        "heldout",
        "test",
        6,
        np.random.default_rng(303),
        heldout_bases=bases,
        base_sequence_ids=[f"base-{i}" for i in range(len(bases))],
    )
    pd.concat([clean, heldout], ignore_index=True).to_parquet(
        result_root / "strict_pair_manifest.parquet", index=False
    )


def _baseline_artifacts(checkpoint_root):
    negative = []
    positive = []
    for index in range(12):
        n = [chr(65 + (index % 20))] * N_EVENTS
        negative.append(SEP.join(n))
        p = [chr(65 + (index % 20))] * N_EVENTS
        p[0], p[7] = "W", "D"
        positive.append(SEP.join(p))
    sequences = negative + positive
    labels = np.r_[np.zeros(len(negative)), np.ones(len(positive))]
    features = build_features(sequences, "letter_count")
    for pi in PI_LEVELS:
        estimator = LogisticRegression(random_state=9550).fit(features, labels)
        artifact = {
            "family": "letter_count_logreg",
            "feature_family": "letter_count",
            "model_kind": "logreg",
            "estimator": estimator,
            "training_mode": "baseline",
            "noise_pi": pi,
            "seed": 9550,
            "smoke": True,
            "validation_threshold": 0.5,
        }
        path = (
            checkpoint_root / "baselines" / f"pi_{pi_slug(pi)}"
            / "letter_count_logreg_smoke" / "artifact.joblib"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)


def test_smoke_grid_builds_each_chunk_once_for_all_four_noise_levels(
    tmp_path, monkeypatch
):
    result_root = tmp_path / "results"
    checkpoint_root = result_root / "checkpoints"
    data_root = tmp_path / "data"
    result_root.mkdir(parents=True)
    data_root.mkdir()
    _hole_artifacts(result_root)
    _strict_artifact(result_root)
    _baseline_artifacts(checkpoint_root)

    feature_calls = []
    real_build_features = grid.build_features

    def recording_build_features(sequences, family):
        feature_calls.append((len(sequences), family))
        return real_build_features(sequences, family)

    monkeypatch.setattr(grid, "build_features", recording_build_features)
    summary = grid.evaluate_baseline_grid(
        "letter_count_logreg",
        PI_LEVELS,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        chunk_bases=2,
        pair_chunk=4,
        bootstrap_iters=12,
        smoke=True,
        command="pytest baseline-grid smoke",
    )
    # Two hole chunks plus two chunks for each of clean and held-out strict
    # pairs.  Each call serves all four pi artifacts.
    assert summary["completed"] == 4
    assert summary["skipped_complete"] == 0
    assert summary["hole_feature_builds"] == 2
    assert summary["strict_feature_builds"] == 4
    assert len(feature_calls) == 6
    assert [rows for rows, _ in feature_calls[:2]] == [1040, 1040]
    assert all(family == "letter_count" for _, family in feature_calls)

    for run_directory in map(grid.Path, summary["run_directories"]):
        done = json.loads((run_directory / "done.json").read_text())
        assert done["status"] == "complete"
        assert done["evaluation_mode"] == "baseline_shared_feature_grid"
        assert done["metadata"]["model"] == "letter_count_logreg"
        assert done["metadata"]["noise_pi"] in PI_LEVELS
        assert all(grid.Path(path).is_file() for path in done["outputs"].values())

        hole_rows = sum(
            len(pd.read_parquet(path))
            for path in sorted((run_directory / "hole_prediction_shards").glob("*.parquet"))
        )
        assert hole_rows == 4 * N_EVENTS * len(ALPHABET)
        repair = pd.read_csv(run_directory / "hole_repair_metrics_test.csv")
        assert "all_targets" in set(repair.segment)
        assert {"macro_candidate_auc_ci_lo", "macro_candidate_auc_ci_hi"} \
            <= set(repair)
        position = pd.read_csv(run_directory / "hole_position_metrics_test.csv")
        assert {"position_auroc_ci_lo", "position_auroc_ci_hi"} <= set(position)
        stability = pd.read_csv(run_directory / "hole_stability_metrics_test.csv")
        assert {"prediction_flip_rate_ci_lo", "prediction_flip_rate_ci_hi"} \
            <= set(stability)
        for background in ("clean", "heldout"):
            strict = pd.read_parquet(
                run_directory
                / f"strict_pair_predictions_{background}_test.parquet"
            )
            assert len(strict) == 6
            assert {"model", "noise_level", "model_seed", "checkpoint_epoch",
                    "strict_win", "tie", "loss"} <= set(strict)
            metrics = pd.read_csv(
                run_directory / f"strict_pair_metrics_{background}_test.csv"
            )
            assert metrics.loc[0, "bootstrap_replicates"] == 12

    calls_after_first_run = len(feature_calls)
    resumed = grid.evaluate_baseline_grid(
        "letter_count_logreg",
        PI_LEVELS,
        data_root=data_root,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        chunk_bases=2,
        pair_chunk=4,
        bootstrap_iters=12,
        smoke=True,
    )
    assert resumed["completed"] == 4
    assert resumed["skipped_complete"] == 4
    assert resumed["hole_feature_builds"] == 0
    assert resumed["strict_feature_builds"] == 0
    assert len(feature_calls) == calls_after_first_run
