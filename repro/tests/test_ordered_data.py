"""Focused invariants for the shared Ordered Compliance data layer."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.oc_completion.oracle import ALPHABET, N_EVENTS, NON_KEY_LETTERS, SEP, oc_label_tokens
from src.oc_completion.ordered_data import (
    CATEGORY_DECISIVE,
    CATEGORY_FIXED_ONE,
    CATEGORY_FIXED_ZERO,
    NOISE_LEVELS,
    build_hole_oracle_arrays,
    enumerate_hole_oracle,
    flip_column,
    load_split,
    make_nested_noise_tables,
    observed_column,
    observed_label_oracle_auc,
    original_sequence_metadata,
    replacement_candidates,
    select_hole_manifest,
    write_ordered_artifacts,
)


def _known_frame(split: str, n_rows: int) -> pd.DataFrame:
    """Unique balanced examples with labels verified by the canonical oracle."""
    nonkeys = list(NON_KEY_LETTERS)
    rows = []
    for index in range(n_rows):
        tokens = ["A"] * N_EVENTS
        # Identifier positions remain non-key and cannot affect the witness.
        tokens[18] = nonkeys[index % len(nonkeys)]
        tokens[19] = nonkeys[(index // len(nonkeys)) % len(nonkeys)]
        if index % 2:
            tokens[0], tokens[7] = "W", "D"  # ordered positive witness
        else:
            tokens[0], tokens[7] = "D", "W"  # decreasing negative run
        rows.append(
            {
                "sequence_id": f"{split}_{index:09d}",
                "X": SEP.join(tokens),
                "Y_star": oc_label_tokens(tokens),
            }
        )
    frame = pd.DataFrame(rows)
    frame["Y_star"] = frame["Y_star"].astype(np.uint8)
    assert frame["X"].is_unique
    assert set(frame["Y_star"]) == {0, 1}
    return frame


def _base_splits() -> dict[str, pd.DataFrame]:
    return {split: _known_frame(split, 20) for split in ("train", "val", "test")}


def test_nested_noise_uses_one_reproducible_global_uniform_vector():
    base = _base_splits()
    first = make_nested_noise_tables(base, noise_seed=12345)
    second = make_nested_noise_tables(base, noise_seed=12345)
    different = make_nested_noise_tables(base, noise_seed=12346)

    expected_uniforms = np.random.default_rng(12345).random(60)
    actual_uniforms = np.concatenate([first[split]["noise_uniform"] for split in base])
    np.testing.assert_array_equal(actual_uniforms, expected_uniforms)
    assert not np.array_equal(
        first["train"]["noise_uniform"], different["train"]["noise_uniform"]
    )

    for split, base_frame in base.items():
        pd.testing.assert_frame_equal(first[split], second[split])
        np.testing.assert_array_equal(
            first[split][observed_column(0.0)], base_frame["Y_star"]
        )
        previous = np.zeros(len(base_frame), dtype=bool)
        for pi in NOISE_LEVELS:
            mask = first[split][flip_column(pi)].to_numpy()
            assert np.all(~previous | mask)
            expected = np.bitwise_xor(
                base_frame["Y_star"].to_numpy(dtype=np.uint8), mask.astype(np.uint8)
            )
            np.testing.assert_array_equal(first[split][observed_column(pi)], expected)
            previous = mask


def test_observed_label_oracle_auc_uses_pi_and_empirical_prevalence():
    rho = 0.2917
    assert observed_label_oracle_auc(rho, 0.0) == pytest.approx(1.0)
    assert observed_label_oracle_auc(rho, 0.1) == pytest.approx(0.8718851238)
    assert observed_label_oracle_auc(rho, 0.2) == pytest.approx(0.7644565795)
    assert observed_label_oracle_auc(rho, 0.3) == pytest.approx(0.6700098734)
    assert observed_label_oracle_auc(0.0, 0.0) is None


@pytest.mark.parametrize("position", [0, 7, 19])
def test_replacement_candidates_are_complete_and_change_only_one_position(position):
    tokens = tuple("ABCDEFGHIJKLMNOPQRST")
    candidates = list(replacement_candidates(tokens, position))
    assert len(candidates) == 26
    assert [candidate[position] for candidate in candidates] == list(ALPHABET)
    assert tokens in candidates
    for candidate in candidates:
        changed = [i for i, (left, right) in enumerate(zip(tokens, candidate)) if left != right]
        assert changed in ([], [position])


def test_hole_categories_are_exhaustive_and_target_sets_partition_alphabet():
    # Negative sequence with exactly one key: replacing its lag partner can
    # create compliance, while most other positions remain fixed-zero.
    create_tokens = ["A"] * N_EVENTS
    create_tokens[0] = "W"
    create = enumerate_hole_oracle(create_tokens)
    assert create["category"][7] == CATEGORY_DECISIVE
    assert create["can_create_compliance"][7]
    assert np.any(create["category"] == CATEGORY_FIXED_ZERO)

    # An existing ordered witness makes unrelated positions fixed-one; either
    # witness endpoint is decisive because a replacement can destroy it.
    destroy_tokens = ["A"] * N_EVENTS
    destroy_tokens[0], destroy_tokens[7] = "W", "D"
    destroy = enumerate_hole_oracle(destroy_tokens)
    assert destroy["category"][0] == CATEGORY_DECISIVE
    assert destroy["can_destroy_compliance"][0]
    assert destroy["category"][1] == CATEGORY_FIXED_ONE

    for result in (create, destroy):
        categories = result["category"]
        assert set(np.unique(categories)).issubset(
            {CATEGORY_FIXED_ZERO, CATEGORY_DECISIVE, CATEGORY_FIXED_ONE}
        )
        np.testing.assert_array_equal(
            result["fixed_zero"].astype(np.uint8)
            + result["decisive"].astype(np.uint8)
            + result["fixed_one"].astype(np.uint8),
            np.ones(N_EVENTS, dtype=np.uint8),
        )
        np.testing.assert_array_equal(
            result["num_candidate_zero"] + result["num_candidate_one"],
            np.full(N_EVENTS, 26, dtype=np.uint8),
        )


def test_optional_run_and_witness_metadata_uses_canonical_oracle():
    tokens = ["A"] * N_EVENTS
    tokens[0], tokens[7] = "W", "D"
    tokens[1] = "Q"  # singleton key in a different residue chain
    metadata = original_sequence_metadata(tokens)
    assert metadata["ordered_witness_count"] == 1
    assert metadata["original_maximal_key_run_length"][0] == 2
    assert metadata["original_maximal_key_run_length"][7] == 2
    assert metadata["original_maximal_key_run_length"][1] == 1
    assert metadata["position_in_ordered_witness"][0]
    assert metadata["position_in_ordered_witness"][7]
    assert not metadata["position_in_ordered_witness"][1]
    assert metadata["residue_chain_index"].tolist()[:8] == [1, 2, 3, 4, 5, 6, 7, 1]


def test_fixed_hole_manifest_and_npz_contract_is_row_aligned(tmp_path):
    base = _base_splits()
    noise = make_nested_noise_tables(base, noise_seed=222)
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    dataset_manifest, noise_manifest = write_ordered_artifacts(
        base_splits=base,
        noise_tables=noise,
        data_root=data_root,
        results_root=results_root,
        data_seed=111,
        noise_seed=222,
        hole_seed=333,
        hole_sizes={"validation": 6, "test": 8, "diagnostic": 4},
        n_workers=1,
        pool_metadata={"pool_size": 60, "pool_latent_prevalence": 0.5},
        command="pytest",
    )

    holes = pd.read_parquet(results_root / "hole_manifest.parquet")
    assert len(holes) == 14
    assert holes["base_sequence_id"].is_unique
    assert holes["oracle_row"].tolist() == list(range(14))
    assert set(holes["split"]) == {"hole_val", "hole_test"}
    assert int(holes["is_epoch_diagnostic"].sum()) == 4
    np.testing.assert_array_equal(holes["base_Y_star"], holes["Y_star"])

    with np.load(results_root / "hole_oracle_labels.npz", allow_pickle=False) as arrays:
        assert arrays["oracle_labels"].shape == (14, 20, 26)
        np.testing.assert_array_equal(arrays["oracle_labels"], arrays["candidate_y_star"])
        np.testing.assert_array_equal(
            arrays["sequence_id"], holes["base_sequence_id"].to_numpy(dtype=str)
        )
        np.testing.assert_array_equal(arrays["base_y_star"], holes["base_Y_star"])

    loaded = load_split(data_root, "train", 0.3)
    assert list(loaded.columns) == [
        "sequence_id",
        "X",
        "Y_star",
        "noise_uniform",
        "flipped",
        "Y_observed",
    ]
    assert len(loaded) == len(base["train"])
    for pi in NOISE_LEVELS:
        condition = load_split(data_root, "train", pi)
        pd.testing.assert_frame_equal(
            condition[["sequence_id", "X", "Y_star"]],
            loaded[["sequence_id", "X", "Y_star"]],
        )
    assert (data_root / "dataset_manifest.json").read_bytes() == (
        results_root / "dataset_manifest.json"
    ).read_bytes()
    assert (data_root / "noise_manifest.json").read_bytes() == (
        results_root / "noise_manifest.json"
    ).read_bytes()
    assert dataset_manifest["hole_artifacts"]["oracle_label_shape"] == [14, 20, 26]
    assert noise_manifest["noise_levels"] == list(NOISE_LEVELS)
    with (results_root / "dataset_manifest.json").open() as handle:
        assert json.load(handle)["mechanism"]["lag"] == 7


def test_hole_correctness_never_uses_observed_noisy_labels():
    base = _base_splits()
    noise = make_nested_noise_tables(base, noise_seed=1)
    holes = select_hole_manifest(base, 6, 8, 4, hole_seed=2)
    pd.testing.assert_frame_equal(
        holes, select_hole_manifest(base, 6, 8, 4, hole_seed=2)
    )
    arrays = build_hole_oracle_arrays(holes, n_workers=1)
    # At high noise some observed labels differ, but the row-aligned base and
    # every original-letter candidate still equal latent Y_star.
    observed = noise["val"].set_index("sequence_id")[observed_column(0.3)]
    val_holes = holes[holes["source_split"] == "val"]
    assert any(
        int(observed.loc[row.sequence_id]) != int(row.Y_star)
        for row in val_holes.itertuples()
    )
    np.testing.assert_array_equal(arrays["base_y_star"], holes["Y_star"])
    for row_index, sequence in enumerate(holes["X"]):
        tokens = sequence.split(SEP)
        for position, original in enumerate(tokens):
            letter_index = ALPHABET.index(original)
            assert arrays["oracle_labels"][row_index, position, letter_index] == holes.loc[
                row_index, "Y_star"
            ]
