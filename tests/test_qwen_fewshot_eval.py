"""CPU checks of the few-shot protocol and label-scoring boundary."""
import json
import numpy as np
import pytest

from src.experiments import qwen_fewshot_eval as q


SEQ = "ABCDEFGHIJKLMNOPQRST"
QUERY = "ZYXWVUTSRQPONMLKJIHG"


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_unit_separator_events_and_raw_prompt_have_no_separator_artifacts():
    sequence = "\x1f".join(SEQ)
    prompt = q.build_prompt([(sequence, 1)], QUERY)
    assert prompt == ("Sequential events: " + " ".join(SEQ) + "\nOutcome (0 or 1): 1\n\n"
                      + "Sequential events: " + " ".join(QUERY) + "\nOutcome (0 or 1): ")
    assert "\x1f" not in prompt
    with pytest.raises(ValueError, match="20 uppercase"):
        q.normalize_sequence(SEQ[:-1])


def test_label_scoring_conditions_on_space_before_digit():
    prompt = q.build_prompt([], QUERY)
    ids, labels = q.tokenize_prompts(CharacterTokenizer(), [prompt], 4096)
    assert ids[0][-1] == ord(" ")
    assert labels == [[ord("0"), ord("1")]]
    assert ids[0] + [labels[0][1]] == CharacterTokenizer().encode(prompt + "1")


@pytest.mark.parametrize("kind", ["retokenizes_prefix", "two_label_tokens", "same_label_token"])
def test_unsupported_label_tokenization_is_rejected(kind):
    class Unsupported(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens)
            if text[-1:] in ("0", "1"):
                if kind == "retokenizes_prefix":
                    return ids[:-2] + [1000]
                if kind == "two_label_tokens":
                    return ids + [1000]
                return ids[:-1] + [1000]
            return ids
    with pytest.raises(ValueError, match="tokenization|distinct"):
        q.tokenize_prompts(Unsupported(), [q.build_prompt([], QUERY)], 4096)


def test_overlength_prompts_are_rejected_without_truncation():
    prompt = q.build_prompt([(SEQ, 0)] * 20, QUERY)
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="truncation forbidden"):
        q.tokenize_prompts(tokenizer, [prompt], len(prompt) - 1)
    ids, _ = q.tokenize_prompts(tokenizer, [prompt], len(prompt))
    assert len(ids[0]) == len(prompt)


def test_test_sample_is_fixed_stratified_unique_and_independent_of_demo_seed():
    labels = np.array([0] * 70 + [1] * 30)
    expected = q.stratified_test_indices(labels, 20, 9950)
    assert len(set(expected)) == 20 and labels[expected].sum() == 6
    for seed in (0, 1, 2):
        q.sample_demo_indices(labels, 5, seed)
        assert np.array_equal(expected, q.stratified_test_indices(labels, 20, 9950))
    assert not np.array_equal(expected, q.stratified_test_indices(labels, 20, 9951))
    for size in (2, 99, 100):
        selected = q.stratified_test_indices(labels, size, 0)
        assert len(selected) == size and set(labels[selected]) == {0, 1}
    with pytest.raises(ValueError, match="available rows"):
        q.stratified_test_indices(labels, 101, 0)


@pytest.mark.parametrize("shots", [0, 5, 20])
def test_demonstrations_are_exactly_balanced_and_respect_eligible_training_rows(shots):
    labels = np.array([0] * 40 + [1] * 40)
    eligible = np.ones(80, dtype=bool)
    eligible[::3] = False
    chosen = q.sample_demo_indices(labels, shots, 2, eligible)
    assert len(chosen) == len(set(chosen)) == shots
    assert int(labels[chosen].sum()) == shots // 2
    assert eligible[chosen].all()
    assert np.array_equal(chosen, q.sample_demo_indices(labels, shots, 2, eligible))
    with pytest.raises(ValueError, match="insufficient"):
        q.sample_demo_indices(labels, 100, 0, eligible)


def test_overlap_check_compares_normalized_sequences_not_row_numbers():
    with pytest.raises(ValueError, match="overlaps"):
        q.assert_no_overlap([(SEQ, 1)], ["\x1f".join(SEQ)])
    q.assert_no_overlap([(SEQ, 1)], [QUERY])


def test_grid_zero_shot_is_evaluated_once_per_task():
    cells = q.build_cells(["kip4", "kip6", "kip4"], [0, 5, 20, 0], [0, 1, 2, 0])
    assert len(cells) == 14
    zero = [c for c in cells if c["shots"] == 0]
    assert len(zero) == 2 and all(c["demo_seed"] is None for c in zero)
    assert len({c["key"] for c in cells}) == len(cells)


def test_completed_skip_requires_identical_configuration_and_all_artifacts(tmp_path):
    cell_id = q.identity({"shots": 5, "seed": 0})
    assert q.completed_cell(tmp_path, cell_id) is None
    sample = tmp_path / "test_sample.json"
    sample.write_text('{"indices": [1, 2]}')
    for name in ("predictions.csv", "demonstrations.json"):
        (tmp_path / name).write_text("retained artifact\n")
    q.atomic_json(tmp_path / "manifest.json", {"test_sample": {"path": str(sample), "sha256": q.sha256(sample)}})
    record = {"status": "completed", "cell_identity": cell_id,
              "artifact_sha256": {name: q.sha256(tmp_path / name) for name in
                                  ("predictions.csv", "manifest.json", "demonstrations.json")}}
    q.atomic_json(tmp_path / "metrics.json", record)
    assert q.completed_cell(tmp_path, cell_id) == record
    with pytest.raises(ValueError, match="configuration differs"):
        q.completed_cell(tmp_path, q.identity({"shots": 20, "seed": 0}))
    (tmp_path / "predictions.csv").write_text("changed\n")
    with pytest.raises(ValueError, match="missing or changed"):
        q.completed_cell(tmp_path, cell_id)


def test_configuration_is_not_silently_overwritten(tmp_path):
    path = tmp_path / "config.json"
    q.ensure_json(path, {"shots": [0, 5]})
    q.ensure_json(path, {"shots": [0, 5]})
    with pytest.raises(ValueError, match="configuration differs"):
        q.ensure_json(path, {"shots": [0, 20]})
    assert json.loads(path.read_text()) == {"shots": [0, 5]}


def test_bfloat16_logits_are_upcast_before_subtraction_and_padding_is_left():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    class Model:
        def __call__(self, **kwargs):
            assert kwargs["logits_to_keep"] == 1 and kwargs["use_cache"] is False
            assert kwargs["input_ids"].tolist() == [[7, 8], [0, 9]]
            assert kwargs["attention_mask"].tolist() == [[1, 1], [0, 1]]
            assert kwargs["position_ids"].tolist() == [[0, 1], [0, 0]]
            assert not torch.is_grad_enabled()
            # Subtraction in BF16 would round 1 - BF16(.001) to exactly 1.
            logits = torch.tensor([[[.001, 1.]], [[1., 0.]]], dtype=torch.bfloat16)
            return SimpleNamespace(logits=logits)
    logits, odds, p = q.score_batch(Model(), [[7, 8], [9]], [[0, 1], [0, 1]], 0, "cpu")
    assert logits.dtype == np.float32 and odds.dtype == np.float32
    assert odds[0] < 1 and odds[0] == 1 - float(torch.tensor(.001, dtype=torch.bfloat16))
    assert odds[1] == -1
    assert np.allclose(p, 1 / (1 + np.exp(-odds)))


def test_metrics_expose_all_positive_f1_baseline():
    result = q.metrics([0, 0, 1, 1], [.6] * 4)
    assert result["auc"] == .5 and result["accuracy"] == .5
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["positive_predictions"] == 4
    assert result["confusion_matrix_labels_0_1"] == [[0, 2], [0, 2]]


def test_auc_uses_raw_log_odds_when_float32_sigmoid_saturates():
    odds = np.array([20., 30., 40., 50.], dtype=np.float32)
    probabilities = 1 / (1 + np.exp(-odds))
    assert np.array_equal(probabilities, np.ones(4, dtype=np.float32))
    result = q.metrics([0, 0, 1, 1], probabilities, ranking_scores=odds)
    assert result["auc"] == 1.
    assert result["auc_from_probabilities"] == .5
    assert result["accuracy"] == .5 and result["positive_predictions"] == 4
    assert q.metrics([0, 0, 1, 1], probabilities)["auc"] == .5
    with pytest.raises(ValueError, match="finite and aligned"):
        q.metrics([0, 0, 1, 1], probabilities, ranking_scores=[0., 1.])
    with pytest.raises(ValueError, match="finite and aligned"):
        q.metrics([0, 0, 1, 1], probabilities, ranking_scores=[0., 1., 2., np.inf])


def test_dry_run_accepts_missing_weights_and_float_labels(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    sequences = ["\x1f".join(chr(65 + i) + SEQ[1:]) for i in range(6)]
    for split in ("train", "test"):
        # Test examples have a different last letter; no identical demo/query sequence.
        values = sequences if split == "train" else [s[:-1] + "Z" for s in sequences]
        (directory / f"X_{split}_kip_m4.csv").write_text("Sequences\n" + "\n".join(values) + "\n")
        (directory / f"y_{split}_kip_m4.csv").write_text("Outcome\n0.0\n1.0\n0.0\n1.0\n0.0\n1.0\n")
    args = q.parse_args(["--model-path", str(tmp_path / "not-downloaded"), "--output-dir", str(tmp_path / "output"),
                        "--kip-data-dir", str(directory), "--tasks", "kip4", "--shots", "0,5",
                        "--n-test", "4", "--dry-run"])
    result = q.run(args)
    assert result["status"] == "data_validated" and not result["weights_loaded"]
    assert len(result["config"]["cells"]) == 4
    assert result["config"]["data"]["kip4"]["sampled_test_positives"] == 2


def test_oc_source_override_keeps_legacy_tasks_explicit(tmp_path):
    args = q.parse_args(["--model-path", str(tmp_path / "model"), "--output-dir", str(tmp_path / "out"),
                        "--data-dir", str(tmp_path / "legacy"), "--oc-data-dir", str(tmp_path / "canonical"),
                        "--oc-det-tag", "ocdet", "--oc-noise-tag", "ocnoise"])
    assert q.task_source(args, "oc_det") == (tmp_path / "canonical", "ocdet")
    assert q.task_source(args, "oc_noise") == (tmp_path / "canonical", "ocnoise")
    assert q.task_source(args, "oc_det_legacy") == (tmp_path / "legacy", "6")
    assert q.task_source(args, "gpc") == (tmp_path / "legacy", "test_just_pair")


def test_grid_loads_model_once_and_completed_restart_checks_saved_samples(tmp_path, monkeypatch):
    directory = tmp_path / "data"
    directory.mkdir()
    for split in ("train", "test"):
        rows = [chr(65 + i) + SEQ[1:-1] + ("T" if split == "train" else "Z") for i in range(6)]
        (directory / f"X_{split}_kip_m4.csv").write_text("Sequences\n" + "\n".join(rows) + "\n")
        (directory / f"y_{split}_kip_m4.csv").write_text("Outcome\n0\n1\n0\n1\n0\n1\n")
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").write_bytes(b"mock-weight-file")
    (model_path / "config.json").write_text('{"model_type":"qwen2"}')
    args = q.parse_args(["--model-path", str(model_path), "--output-dir", str(tmp_path / "output"),
                        "--kip-data-dir", str(directory), "--tasks", "kip4", "--shots", "0,5",
                        "--demo-seeds", "0,1", "--n-test", "4", "--batch-size", "2"])
    calls = {"load": 0, "score": 0}
    def load(_):
        calls["load"] += 1
        tokenizer = CharacterTokenizer()
        tokenizer.pad_token_id = 0
        return object(), tokenizer, {"test_runtime": True}
    def score(model, token_ids, label_ids, pad_token_id, device):
        calls["score"] += 1
        n = len(token_ids)
        return np.zeros((n, 2)), np.zeros(n), np.full(n, .5)
    monkeypatch.setattr(q, "load_model", load)
    monkeypatch.setattr(q, "score_batch", score)
    result = q.run(args)
    assert result["status"] == "completed" and len(result["cells"]) == 3
    assert calls == {"load": 1, "score": 6}
    assert q.run(args) == result
    assert calls == {"load": 1, "score": 6}
    sample_path = args.output_dir / "samples" / "kip4.json"
    sample_path.write_text('{"indices": [99]}')
    with pytest.raises(ValueError, match="test sample missing or changed"):
        q.run(args)
