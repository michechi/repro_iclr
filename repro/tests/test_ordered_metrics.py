"""Focused contracts for exhaustive-hole and strict-pair evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.oc_completion.ordered_data import enumerate_candidate_labels
from src.oc_completion.ordered_eval import (
    _score_pairs_same_batch,
    enumerate_position_candidates,
    evaluate_holes,
)
from src.oc_completion.ordered_features import (
    build_features,
    maximal_run_histogram,
    per_chain_unordered_bags,
)
from src.oc_completion.ordered_metrics import (
    calibrate_tie_tolerance,
    hole_cluster_bootstrap_ci,
    position_localization_metrics,
    stability_metrics,
    strict_pair_metrics,
    valid_filling_metrics,
)
from src.oc_completion.oracle import ALPHABET, N_EVENTS, SEP, oc_label, oc_label_tokens
from src.oc_completion.strict_pairs import generate_strict_family


def _hole_examples() -> tuple[list[str], pd.DataFrame, np.ndarray]:
    # Fixed-zero sequence, negative sequence with one create position, and
    # positive sequence with destroy positions plus many fixed-one positions.
    all_non_key = ["A"] * N_EVENTS
    one_key = ["A"] * N_EVENTS
    one_key[0] = "W"
    ordered_pair = list(one_key)
    ordered_pair[7] = "D"
    token_rows = [all_non_key, one_key, ordered_pair]
    sequences = [SEP.join(tokens) for tokens in token_rows]
    manifest = pd.DataFrame({
        "base_sequence_id": ["fixed-zero", "create", "destroy"],
        "base_Y_star": [oc_label_tokens(tokens) for tokens in token_rows],
        "X": sequences,
        "split": "hole_test",
    })
    oracle = np.stack([enumerate_candidate_labels(tokens) for tokens in token_rows])
    return sequences, manifest, oracle


def _oracle_scorer(sequences: list[str]) -> np.ndarray:
    labels = np.asarray([oc_label(sequence) for sequence in sequences], dtype=np.float32)
    # Two-class logits give positive logits 0/1 and probabilities below/above .5.
    return np.stack([1.0 - labels, labels], axis=1).astype(np.float32)


def test_position_enumeration_is_complete_and_changes_only_the_hole():
    original_tokens = list("ABCDEFGHIJKLMNOPQRST")
    original = SEP.join(original_tokens)
    position = 7
    candidates = enumerate_position_candidates(original, position)

    assert len(candidates) == len(ALPHABET) == 26
    candidate_tokens = [candidate.split(SEP) for candidate in candidates]
    assert [tokens[position] for tokens in candidate_tokens] == list(ALPHABET)
    assert len(set(candidates)) == 26
    assert original in candidates
    assert candidates[ALPHABET.index(original_tokens[position])] == original
    for tokens in candidate_tokens:
        assert len(tokens) == N_EVENTS
        assert all(tokens[index] == original_tokens[index]
                   for index in range(N_EVENTS) if index != position)
    with pytest.raises(IndexError):
        enumerate_position_candidates(original, -1)
    with pytest.raises(IndexError):
        enumerate_position_candidates(original, N_EVENTS)


def test_oracle_hole_metrics_are_perfect_and_targets_partition_alphabet():
    _, manifest, oracle = _hole_examples()
    positive_logits = oracle.astype(np.float32)
    probabilities = oracle.astype(np.float32)

    for labels in oracle.reshape(-1, len(ALPHABET)):
        target_zero = set(np.flatnonzero(labels == 0))
        target_one = set(np.flatnonzero(labels == 1))
        assert target_zero.isdisjoint(target_one)
        assert target_zero | target_one == set(range(len(ALPHABET)))

    position = position_localization_metrics(
        manifest,
        oracle,
        positive_logits,
        probabilities,
        bootstrap_iters=40,
        bootstrap_seed=700,
    ).set_index("segment")
    for segment in ("all", "negative_create", "positive_destroy"):
        assert position.loc[segment, "position_auroc"] == pytest.approx(1.0)
        assert position.loc[segment, "position_auprc"] == pytest.approx(1.0)
        assert position.loc[segment, "within_sequence_average_precision"] \
            == pytest.approx(1.0)
        assert position.loc[segment, "top1_localization_accuracy"] \
            == pytest.approx(1.0)
        assert position.loc[segment, "top3_decisive_recall"] == pytest.approx(1.0)
        assert position.loc[segment, "position_auroc_ci_lo"] == pytest.approx(1.0)
        assert position.loc[segment, "position_auroc_ci_hi"] == pytest.approx(1.0)
    assert position.loc["all", "mean_delta_decisive"] == pytest.approx(1.0)
    assert position.loc["all", "mean_delta_fixed_zero"] == pytest.approx(0.0)
    assert position.loc["all", "mean_delta_fixed_one"] == pytest.approx(0.0)

    repair = valid_filling_metrics(
        manifest,
        oracle,
        positive_logits,
        bootstrap_iters=40,
        bootstrap_seed=701,
    ).set_index("segment")
    assert {
        "all_targets",
        "target_0",
        "target_1",
        "create_compliance",
        "destroy_compliance",
        "preserve_base_label",
    } <= set(repair.index)
    for segment in repair.index:
        assert repair.loc[segment, "macro_candidate_auc"] == pytest.approx(1.0)
        assert repair.loc[segment, "valid_invalid_pairwise_win_rate"] \
            == pytest.approx(1.0)
        assert repair.loc[segment, "top1_valid_filling_accuracy"] \
            == pytest.approx(1.0)
        assert repair.loc[segment, "tie_aware_top1_accuracy"] \
            == pytest.approx(1.0)
        assert repair.loc[segment, "mrr_first_valid_filling"] \
            == pytest.approx(1.0)
        assert repair.loc[segment, "mean_valid_minus_invalid_logit_margin"] \
            == pytest.approx(1.0)
        assert repair.loc[segment, "macro_candidate_auc_ci_lo"] \
            == pytest.approx(1.0)
        assert repair.loc[segment, "macro_candidate_auc_ci_hi"] \
            == pytest.approx(1.0)
    assert repair.loc["all_targets", "n_positions"] == 8
    assert repair.loc["target_0", "n_positions"] == 4
    assert repair.loc["target_1", "n_positions"] == 4

    stability = stability_metrics(
        manifest,
        oracle,
        positive_logits,
        probabilities,
        {"fixed_0p5": 0.5, "validation_selected": 0.6},
        bootstrap_iters=40,
        bootstrap_seed=702,
    )
    assert set(stability.category) == {"fixed_zero", "fixed_one"}
    assert set(stability.threshold_name) == {"fixed_0p5", "validation_selected"}
    assert np.allclose(stability.prediction_flip_rate, 0.0)
    assert np.allclose(stability.all_candidates_correct_fraction, 1.0)
    assert np.allclose(stability.individual_candidate_accuracy, 1.0)
    assert np.allclose(stability.prediction_flip_rate_ci_lo, 0.0)
    assert np.allclose(stability.prediction_flip_rate_ci_hi, 0.0)


def test_hole_cluster_bootstrap_resamples_bases_and_is_reproducible():
    clusters = np.repeat(np.asarray(["a", "b", "c", "d"]), 3)
    values = np.repeat(np.asarray([0.0, 0.25, 0.75, 1.0]), 3)
    first = hole_cluster_bootstrap_ci(
        clusters,
        statistic="mean",
        values=values,
        n_boot=300,
        seed=91,
    )
    second = hole_cluster_bootstrap_ci(
        clusters,
        statistic="mean",
        values=values,
        n_boot=300,
        seed=91,
    )
    assert first == second
    assert first["estimate"] == pytest.approx(values.mean())
    assert first["bootstrap_clusters"] == 4
    assert first["bootstrap_replicates"] == 300
    assert first["confidence_level"] == pytest.approx(0.95)
    assert 0.0 <= first["ci_lo"] <= first["estimate"] <= first["ci_hi"] <= 1.0

    labels = np.tile(np.asarray([0, 1, 0]), 4)
    scores = np.asarray([
        0.0, 0.9, 0.1,
        0.2, 0.8, 0.3,
        0.4, 0.7, 0.5,
        0.6, 1.0, 0.65,
    ])
    auc = hole_cluster_bootstrap_ci(
        clusters,
        statistic="auroc",
        labels=labels,
        scores=scores,
        n_boot=250,
        seed=92,
    )
    assert auc["estimate"] == pytest.approx(roc_auc_score(labels, scores))
    assert 0.0 <= auc["ci_lo"] <= auc["ci_hi"] <= 1.0
    assert auc["bootstrap_finite_replicates"] == 250


def test_oracle_strict_scoring_is_perfect_and_feature_shortcuts_tie_exactly():
    pairs, _ = generate_strict_family(
        "clean", "test", 128, np.random.default_rng(9001)
    )
    scorer_calls: list[list[str]] = []

    def recording_oracle(sequences: list[str]) -> np.ndarray:
        scorer_calls.append(list(sequences))
        return _oracle_scorer(sequences)

    scored = _score_pairs_same_batch(recording_oracle, pairs, pair_chunk=31)
    for call in scorer_calls:
        assert len(call) % 2 == 0
        # Candidate pairs remain adjacent inside every scorer call.
        for index in range(0, len(call), 2):
            source = pairs[
                (pairs.cand0.eq(call[index])) & (pairs.cand1.eq(call[index + 1]))
            ]
            assert len(source) == 1
    assert np.allclose(scored.positive_logit, 1.0)
    assert np.allclose(scored.negative_logit, 0.0)
    assert np.allclose(scored.margin, 1.0)
    metrics = strict_pair_metrics(
        scored,
        tolerance=0.0,
        cluster_col="pair_id",
        n_boot=100,
    )
    assert metrics["strict_win_rate"] == pytest.approx(1.0)
    assert metrics["tie_rate"] == pytest.approx(0.0)
    assert metrics["pair_accuracy"] == pytest.approx(1.0)
    assert metrics["flattened_candidate_auc"] == pytest.approx(1.0)
    assert metrics["pair_accuracy_ci_lo"] == pytest.approx(1.0)
    assert metrics["pair_accuracy_ci_hi"] == pytest.approx(1.0)

    positive = [getattr(row, f"cand{int(row.positive_index)}")
                for row in pairs.itertuples(index=False)]
    negative = [getattr(row, f"cand{1 - int(row.positive_index)}")
                for row in pairs.itertuples(index=False)]
    for family in ("letter_count", "lag_pair", "chain_occupancy",
                   "chain_counts_xgb"):
        pos_features = build_features(positive, family)
        neg_features = build_features(negative, family)
        assert np.array_equal(pos_features, neg_features), family
    for pos_sequence, neg_sequence in zip(positive, negative):
        pos_tokens, neg_tokens = pos_sequence.split(SEP), neg_sequence.split(SEP)
        assert np.array_equal(maximal_run_histogram(pos_tokens),
                              maximal_run_histogram(neg_tokens))
        assert per_chain_unordered_bags(pos_tokens) == \
            per_chain_unordered_bags(neg_tokens)

    # Any deterministic score based only on an exact-tie representation must
    # itself tie and receive half-credit on every strict pair.
    def count_scorer(sequences: list[str]) -> np.ndarray:
        features = build_features(sequences, "letter_count")
        positive_logit = features @ np.linspace(-0.5, 0.5, 26)
        return np.column_stack([np.zeros(len(sequences)), positive_logit])

    tied = _score_pairs_same_batch(count_scorer, pairs, pair_chunk=47)
    assert np.array_equal(tied.positive_logit.to_numpy(),
                          tied.negative_logit.to_numpy())
    tied_metrics = strict_pair_metrics(tied, tolerance=0.0, n_boot=50)
    assert tied_metrics["strict_win_rate"] == pytest.approx(0.0)
    assert tied_metrics["tie_rate"] == pytest.approx(1.0)
    assert tied_metrics["pair_accuracy"] == pytest.approx(0.5)


def test_tie_jitter_calibration_uses_measured_duplicate_range():
    exact = calibrate_tie_tolerance(np.ones((5, 4), dtype=np.float32))
    assert exact["tie_tolerance"] == 0.0
    assert exact["mean_duplicate_range"] == 0.0
    assert exact["n_jitter_candidates"] == 5
    assert exact["n_repeats"] == 4

    duplicated = np.asarray([
        [1.0, 1.0002, 0.9999, 1.0001],
        [-2.0, -2.0001, -2.0, -1.9999],
    ], dtype=np.float64)
    calibrated = calibrate_tie_tolerance(duplicated)
    assert calibrated["tie_tolerance"] == pytest.approx(0.0003)
    margins = pd.DataFrame({
        "pair_id": ["tie", "win", "loss"],
        "margin": [0.0002, 0.0004, -0.0004],
        "positive_logit": [1.0, 1.0, 0.0],
        "negative_logit": [0.9998, 0.9996, 0.0004],
    })
    metrics = strict_pair_metrics(
        margins,
        tolerance=calibrated["tie_tolerance"],
        n_boot=50,
    )
    assert metrics["tie_rate"] == pytest.approx(1 / 3)
    assert metrics["strict_win_rate"] == pytest.approx(1 / 3)
    assert metrics["loss_rate"] == pytest.approx(1 / 3)
    assert metrics["pair_accuracy"] == pytest.approx(0.5)


def test_evaluate_holes_writes_core_cluster_confidence_intervals(tmp_path):
    _, manifest, oracle = _hole_examples()
    manifest_path = tmp_path / "hole_manifest.parquet"
    oracle_path = tmp_path / "hole_oracle_labels.npz"
    manifest.to_parquet(manifest_path, index=False)
    np.savez_compressed(oracle_path, oracle_labels=oracle)

    paths = evaluate_holes(
        _oracle_scorer,
        {"model": "oracle", "seed": 0, "validation_threshold": 0.5},
        pi=0.0,
        manifest_path=manifest_path,
        oracle_path=oracle_path,
        split="test",
        out_dir=tmp_path,
        chunk_bases=2,
        bootstrap_iters=30,
    )
    position = pd.read_csv(paths["hole_position_metrics"])
    repair = pd.read_csv(paths["hole_repair_metrics"])
    stability = pd.read_csv(paths["hole_stability_metrics"])
    assert {"position_auroc_ci_lo", "position_auroc_ci_hi"} <= set(position)
    assert {"macro_candidate_auc_ci_lo", "macro_candidate_auc_ci_hi",
            "top1_valid_filling_accuracy_ci_lo",
            "top1_valid_filling_accuracy_ci_hi"} <= set(repair)
    assert {"prediction_flip_rate_ci_lo", "prediction_flip_rate_ci_hi"} \
        <= set(stability)
    assert "all_targets" in set(repair.segment)
    assert np.allclose(position.position_auroc, 1.0)
    assert np.allclose(repair.macro_candidate_auc, 1.0)
    assert np.allclose(stability.prediction_flip_rate, 0.0)
