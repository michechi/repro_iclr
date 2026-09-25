"""Invariant tests for the strict four-chain Ordered-Compliance family."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.analysis.mechanism_id.scripts.common import feat_count26, feat_lag_pair
from src.oc_completion.oracle import (
    ALPHABET,
    KEY_SET,
    LAG,
    N_EVENTS,
    SEP,
    oc_label_tokens,
)
from src.oc_completion.strict_pairs import (
    COMPLETE_CHAIN_STARTS,
    LEGACY_CONTROL_METADATA,
    LEGACY_FAMILY,
    STRICT_FAMILY,
    assert_manifest_row_invariants,
    assert_strict_pair_invariants,
    chain_positions,
    construct_strict_pair,
    generate_strict_family,
    key_membership_mask,
    legacy_family_name,
    main,
    maximal_run_length_histogram,
    per_chain_key_counts,
    per_chain_unordered_letter_bags,
    record_from_manifest_row,
)


def _random_bases(n: int, seed: int = 7300) -> list[str]:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(ALPHABET)
    return [SEP.join(row) for row in
            alphabet[rng.integers(0, len(alphabet), size=(n, N_EVENTS))]]


def _candidate_tokens(row: dict, positive: bool) -> list[str]:
    pos_index = int(row["positive_index"])
    index = pos_index if positive else 1 - pos_index
    return str(row[f"cand{index}"]).split(SEP)


def test_explicit_strict_template_has_every_required_exact_invariant():
    keys = ("W", "D", "Q", "J")
    # Deliberately non-sorted assignment: template row -> chain start.
    template_to_chain = (5, 1, 4, 0)
    rec = construct_strict_pair(
        ["A"] * N_EVENTS,
        keys,
        template_to_chain,
        background="clean",
    )
    assert_strict_pair_invariants(rec)

    expected_negative = (
        ("W", "Q", "D"),
        ("W", "J", "Q"),
        ("J", "W", "D"),
        ("J", "D", "Q"),
    )
    expected_positive = (
        ("W", "D", "Q"),
        ("J", "W", "Q"),
        ("W", "J", "D"),
        ("J", "Q", "D"),
    )
    for row_index, start in enumerate(template_to_chain):
        positions = chain_positions(start)
        assert tuple(rec["negative"][p] for p in positions) == \
            expected_negative[row_index]
        assert tuple(rec["positive"][p] for p in positions) == \
            expected_positive[row_index]

    negative, positive = rec["negative"], rec["positive"]
    assert oc_label_tokens(negative) == 0
    assert oc_label_tokens(positive) == 1
    assert np.array_equal(feat_count26(negative), feat_count26(positive))
    assert np.array_equal(feat_lag_pair(negative, LAG),
                          feat_lag_pair(positive, LAG))
    assert np.array_equal(key_membership_mask(negative),
                          key_membership_mask(positive))
    assert np.array_equal(per_chain_key_counts(negative),
                          per_chain_key_counts(positive))
    assert np.array_equal(maximal_run_length_histogram(negative),
                          maximal_run_length_histogram(positive))
    assert per_chain_unordered_letter_bags(negative) == \
        per_chain_unordered_letter_bags(positive)

    # The stated aggregated selected-edge multiset is realized exactly.
    def selected_edges(toks):
        return Counter(
            (toks[p], toks[p + LAG])
            for start in template_to_chain
            for p in chain_positions(start)[:-1]
        )

    expected_edges = Counter({
        ("W", "Q"): 1,
        ("Q", "D"): 1,
        ("W", "J"): 1,
        ("J", "Q"): 1,
        ("J", "W"): 1,
        ("W", "D"): 1,
        ("J", "D"): 1,
        ("D", "Q"): 1,
    })
    assert selected_edges(negative) == expected_edges
    assert selected_edges(positive) == expected_edges


def test_ten_thousand_clean_pairs_have_zero_invariant_violations():
    frame, stats = generate_strict_family(
        "clean", "test", 10_000, np.random.default_rng(20260831)
    )
    assert len(frame) == 10_000
    assert stats["attempted"] == 10_000
    assert stats["accepted"] == 10_000
    assert stats["rejected"] == 0
    assert stats["rejection_rate"] == 0.0
    assert stats["rejection_reasons"] == {}
    assert frame["pair_id"].is_unique
    assert set(frame["family"]) == {STRICT_FAMILY}
    assert set(frame["positive_index"]) == {0, 1}
    assert not np.array_equal(frame["pair_id_order"].to_numpy(),
                              np.arange(len(frame)))

    # All 15 choices of four ranks and all 15 choices of four complete chains
    # should be exercised by this seeded large invariant audit.
    key_choices = set(zip(frame.key_a, frame.key_b, frame.key_c, frame.key_d))
    chain_choices = set(frame.selected_chains)
    assert len(key_choices) == 15
    assert len(chain_choices) == 15
    assert frame["template1_chain"].nunique() == len(COMPLETE_CHAIN_STARTS)

    violations = []
    oracle_correct = 0
    for row in frame.to_dict("records"):
        try:
            assert_manifest_row_invariants(row)
        except AssertionError as exc:  # retain pair IDs for actionable failure
            violations.append((row["pair_id"], str(exc)))
        negative = _candidate_tokens(row, positive=False)
        positive = _candidate_tokens(row, positive=True)
        oracle_correct += int(
            oc_label_tokens(negative) == 0 and oc_label_tokens(positive) == 1
        )
    assert violations == []
    assert oracle_correct / len(frame) == 1.0

    # Clean backgrounds contain no keys outside the four selected chains.
    for row in frame.head(250).to_dict("records"):
        selected = {
            p
            for i in range(1, 5)
            for p in chain_positions(int(row[f"template{i}_chain"]) - 1)
        }
        for toks in (_candidate_tokens(row, False),
                     _candidate_tokens(row, True)):
            assert all(toks[p] not in KEY_SET
                       for p in range(N_EVENTS) if p not in selected)


def test_heldout_generation_retains_background_and_avoids_base_reuse():
    bases = _random_bases(1_000)
    base_ids = [f"heldout-{i:04d}" for i in range(len(bases))]
    by_id = dict(zip(base_ids, bases))
    frame, stats = generate_strict_family(
        "heldout",
        "val",
        300,
        np.random.default_rng(4401),
        heldout_bases=bases,
        base_sequence_ids=base_ids,
    )

    assert len(frame) == 300
    assert frame["base_sequence_id"].nunique() == 300
    assert frame["base_id"].equals(frame["base_sequence_id"])
    assert stats["unique_base_sequences"] == 300
    assert stats["repeated_base_pairs"] == 0
    assert stats["at_most_one_pair_per_base_when_possible"] is True
    assert "conditioned" in stats["conditioning"].lower()
    assert sum(stats["background_key_count_distribution"]["attempted"].values()) \
        == stats["attempted"]
    assert sum(stats["background_key_count_distribution"]["accepted"].values()) \
        == stats["accepted"]
    assert sum(stats["background_ordered_run_distribution"]["attempted"].values()) \
        == stats["attempted"]

    for row in frame.to_dict("records"):
        assert_manifest_row_invariants(row)
        base = by_id[row["base_sequence_id"]].split(SEP)
        selected = {
            p
            for i in range(1, 5)
            for p in chain_positions(int(row[f"template{i}_chain"]) - 1)
        }
        negative = _candidate_tokens(row, False)
        positive = _candidate_tokens(row, True)
        assert all(negative[p] == base[p] == positive[p]
                   for p in range(N_EVENTS) if p not in selected)


def test_legacy_two_hole_family_is_explicitly_a_nonprimary_control():
    assert legacy_family_name("two_hole") == LEGACY_FAMILY
    assert legacy_family_name("one_hole") == "one_hole"
    assert LEGACY_CONTROL_METADATA["source_family"] == "two_hole"
    assert LEGACY_CONTROL_METADATA["audit_name"] == LEGACY_FAMILY
    assert LEGACY_CONTROL_METADATA["role"] == "legacy_control"
    assert LEGACY_CONTROL_METADATA["primary_matched_test"] is False
    assert "occupancy" in LEGACY_CONTROL_METADATA["warning"]


def test_smoke_cli_writes_one_evaluator_compatible_parquet(tmp_path):
    val = pd.DataFrame({
        "sequence_id": [f"v{i}" for i in range(40)],
        "Sequences": _random_bases(40, seed=881),
    })
    test = pd.DataFrame({
        "sequence_id": [f"t{i}" for i in range(40)],
        "Sequences": _random_bases(40, seed=882),
    })
    val_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    main([
        "--smoke",
        "--n_val", "8",
        "--n_test", "12",
        "--pair_seed", "9981",
        "--val_file", str(val_path),
        "--test_file", str(test_path),
        "--out_dir", str(tmp_path),
    ])
    path = tmp_path / "strict_pair_manifest_smoke.parquet"
    report_path = tmp_path / "strict_pair_generation_report_smoke.json"
    assert path.exists()
    assert report_path.exists()
    manifest = pd.read_parquet(path)
    assert len(manifest) == 2 * (8 + 12)
    assert set(manifest["family"]) == {STRICT_FAMILY}
    assert set(manifest["background"]) == {"clean", "heldout"}
    assert set(manifest["split"]) == {"val", "test"}
    required = {
        "pair_id",
        "split",
        "background",
        "base_sequence_id",
        "base_id",
        "cand0",
        "cand1",
        "positive_index",
        "family",
    }
    assert required <= set(manifest.columns)
    assert manifest["pair_id"].is_unique

    # Round-trip one row from each dataset through the exact invariant checker.
    for _, group in manifest.groupby(["background", "split"]):
        row = group.iloc[0].to_dict()
        assert_manifest_row_invariants(row)
        assert_strict_pair_invariants(record_from_manifest_row(row))
