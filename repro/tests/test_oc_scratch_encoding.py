"""Regression tests for the OC-specific scratch-model token contract.

The generic DL experiment historically encoded ``A`` as zero while its
Transformer models also treated zero as padding.  The OC runners deliberately
use a separate encoding path: zero is padding only, letters occupy ids 1..26,
and every OC example contains exactly 20 events.
"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from src.oc_completion import train_dl


N_EVENTS = 20
EXAMPLE_WITH_A = "ABCDEFGHIJKLMNOPQRST"
EXAMPLE_WITH_Z = "BCDEFGHIJKLMNOPQRSTZ"


@pytest.mark.parametrize(
    ("letter", "expected_id"),
    [(chr(ord("A") + offset), offset + 1) for offset in range(26)],
)
def test_oc_encoder_maps_a_through_z_to_one_through_twenty_six(
    letter: str, expected_id: int
):
    encoded = train_dl.encode_oc_sequence(letter * N_EVENTS)

    assert encoded.dtype == torch.long
    assert encoded.shape == (N_EVENTS,)
    assert encoded.unique().tolist() == [expected_id]


@pytest.mark.parametrize(
    "invalid_sequence",
    [
        "",
        "A" * (N_EVENTS - 1),
        "A" * (N_EVENTS + 1),
        "A" * (N_EVENTS - 1) + "!",
        "a" * N_EVENTS,
    ],
)
def test_oc_encoder_rejects_non_twenty_event_or_non_alphabet_inputs(
    invalid_sequence: str,
):
    with pytest.raises(ValueError):
        train_dl.encode_oc_sequence(invalid_sequence)


def test_oc_dataset_and_collator_produce_batch_by_twenty_without_real_zero():
    dataset = train_dl.OCSequenceDataset(
        [EXAMPLE_WITH_A, EXAMPLE_WITH_Z], [0, 1]
    )

    first_sequence, first_label = dataset[0]
    assert first_sequence.shape == (N_EVENTS,)
    assert first_sequence[0].item() == 1
    assert first_label.item() == 0

    inputs, labels = train_dl.collate_oc_batch([dataset[0], dataset[1]])
    assert inputs.shape == (2, N_EVENTS)
    assert labels.shape == (2,)
    assert labels.tolist() == [0, 1]
    assert not torch.any(inputs == 0)
    assert inputs.min().item() == 1
    assert inputs.max().item() == 26


def test_make_loader_uses_the_oc_dataset_and_fixed_width_collator():
    loader = train_dl.make_loader(
        [EXAMPLE_WITH_A, EXAMPLE_WITH_Z],
        [0, 1],
        shuffle=False,
        batch_size=2,
    )

    assert isinstance(loader.dataset, train_dl.OCSequenceDataset)
    inputs, labels = next(iter(loader))
    assert inputs.shape == (2, N_EVENTS)
    assert labels.tolist() == [0, 1]
    assert not inputs.eq(0).any()


def test_oc_collator_rejects_non_twenty_width_input():
    malformed = torch.ones(N_EVENTS - 1, dtype=torch.long)
    with pytest.raises(ValueError, match=r"shape \[batch, 20\]"):
        train_dl.collate_oc_batch(
            [(malformed, torch.tensor(0, dtype=torch.long))]
        )


def test_oc_collator_rejects_a_zero_in_a_real_fixed_width_example():
    malformed = torch.ones(N_EVENTS, dtype=torch.long)
    malformed[7] = 0
    with pytest.raises(ValueError, match="contains padding"):
        train_dl.collate_oc_batch(
            [(malformed, torch.tensor(0, dtype=torch.long))]
        )


@pytest.mark.parametrize("model_name", ["LSTM", "Transformer", "RNNTransformer"])
def test_oc_recipe_reserves_a_twenty_seventh_embedding_slot_for_padding(
    model_name: str,
):
    config = train_dl.RECIPES[model_name]["config"]
    model = train_dl.create_model(model_name, config)

    assert config["vocab_size"] == 27
    assert model.embedding.num_embeddings == 27
    assert model.embedding.padding_idx == 0


def test_oc_encoding_metadata_records_the_fixed_width_contract():
    encoding = train_dl.OC_ENCODING

    assert encoding["name"] == "oc_pad0_a1_z26"
    assert encoding["padding_id"] == 0
    assert encoding["alphabet_ids"]["A"] == 1
    assert encoding["alphabet_ids"]["Z"] == 26
    assert encoding["sequence_length"] == N_EVENTS
    assert encoding["vocab_size"] == 27


class _TransformerMaskRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.masks: list[torch.Tensor] = []

    def forward(self, hidden, *, src_key_padding_mask=None):
        assert src_key_padding_mask is not None
        self.masks.append(src_key_padding_mask.detach().clone())
        return hidden


@pytest.mark.parametrize("model_name", ["Transformer", "RNNTransformer"])
def test_real_a_event_is_not_masked_by_scratch_transformers(model_name: str):
    config = deepcopy(train_dl.RECIPES[model_name]["config"])
    config["dropout"] = 0.0
    model = train_dl.create_model(model_name, config)
    recorder = _TransformerMaskRecorder()
    model.transformer = recorder
    model.eval()

    inputs = torch.stack(
        [
            train_dl.encode_oc_sequence(EXAMPLE_WITH_A),
            train_dl.encode_oc_sequence("A" * N_EVENTS),
        ]
    )
    logits = model(inputs)

    assert logits.shape == (2, 2)
    assert len(recorder.masks) == 1
    padding_mask = recorder.masks[0]
    assert padding_mask.shape == (2, N_EVENTS)
    assert padding_mask.dtype == torch.bool
    assert not padding_mask.any()
    assert not padding_mask[:, 0].any()  # both first positions are real A events


class _InputRecorder(torch.nn.Module):
    def __init__(self, vocab_size: int = 27):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size, 1, padding_idx=0
        )
        self.inputs: list[torch.Tensor] = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().cpu().clone())
        return torch.zeros((inputs.shape[0], 2), device=inputs.device)


def test_pair_model_logits_calls_the_shared_corrected_oc_encoder(monkeypatch):
    original_encoder = train_dl.encode_oc_sequence
    encoded_sequences: list[str] = []

    def recording_encoder(sequence: str, *, legacy_zero_based: bool = False):
        encoded_sequences.append(sequence)
        return original_encoder(
            sequence, legacy_zero_based=legacy_zero_based
        )

    monkeypatch.setattr(train_dl, "encode_oc_sequence", recording_encoder)
    model = _InputRecorder()
    raw_sequences = [
        "\x1f".join(EXAMPLE_WITH_A),
        "\x1f".join(EXAMPLE_WITH_Z),
    ]

    logits = train_dl.model_logits(
        model, raw_sequences, torch.device("cpu"), batch_size=2
    )

    assert logits.shape == (2, 2)
    assert encoded_sequences == raw_sequences
    assert len(model.inputs) == 1
    inputs = model.inputs[0]
    assert inputs.shape == (2, N_EVENTS)
    assert not inputs.eq(0).any()
    assert inputs[0, 0].item() == 1
    assert inputs[1, -1].item() == 26


def test_pair_model_logits_rejects_a_non_twenty_event_candidate():
    model = _InputRecorder()
    with pytest.raises(ValueError):
        train_dl.model_logits(
            model,
            ["\x1f".join("A" * (N_EVENTS - 1))],
            torch.device("cpu"),
        )


@pytest.mark.parametrize("model_name", ["Transformer", "RNNTransformer"])
def test_explicit_legacy_zero_based_encoding_remains_vocab26_compatible(
    model_name: str,
):
    config = deepcopy(train_dl.RECIPES[model_name]["config"])
    config["vocab_size"] = 26
    config["dropout"] = 0.0
    model = train_dl.create_model(model_name, config).eval()
    legacy_input = train_dl.encode_oc_sequence(
        EXAMPLE_WITH_Z, legacy_zero_based=True
    ).unsqueeze(0)

    assert legacy_input.shape == (1, N_EVENTS)
    assert legacy_input.min().item() == 1  # this sequence starts at B
    assert legacy_input.max().item() == 25
    assert model(legacy_input).shape == (1, 2)


def test_legacy_zero_based_mode_reproduces_a_as_zero_only_when_requested():
    corrected = train_dl.encode_oc_sequence("A" * N_EVENTS)
    legacy = train_dl.encode_oc_sequence(
        "A" * N_EVENTS, legacy_zero_based=True
    )

    assert corrected.unique().tolist() == [1]
    assert legacy.unique().tolist() == [0]


def test_pair_model_logits_autodetects_a_legacy_vocab26_checkpoint():
    model = _InputRecorder(vocab_size=26)
    raw_sequences = [
        "\x1f".join("A" * N_EVENTS),
        "\x1f".join(EXAMPLE_WITH_Z),
    ]

    with pytest.warns(RuntimeWarning, match="legacy 26-slot"):
        logits = train_dl.model_logits(
            model, raw_sequences, torch.device("cpu"), batch_size=2
        )

    assert logits.shape == (2, 2)
    inputs = model.inputs[0]
    assert inputs.shape == (2, N_EVENTS)
    assert inputs[0].unique().tolist() == [0]
    assert inputs[1, -1].item() == 25
