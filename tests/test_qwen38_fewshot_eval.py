import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments import qwen38_fewshot_eval as q

SEQ = "ABCDEFGHIJKLMNOPQRST"
QUERY = "ZYXWVUTSRQPONMLKJIHG"


class Tokenizer:
    chat_template = "official-template-fixture"
    pad_token_id, eos_token_id = 0, 1

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return "<user>" + messages[0]["content"].strip() + "</user><assistant><think>\n\n</think>\n\n"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]


def test_official_chat_non_thinking_preserves_examples_and_verifies_newline_boundary():
    tokenizer = Tokenizer()
    prompt = q.build_chat_prompt(tokenizer, [(SEQ, 1)], QUERY)
    assert q.INSTRUCTION in prompt
    assert q.core.render_example(SEQ, 1) in prompt
    assert prompt.endswith("</think>\n\n")
    ids, labels = q.tokenize_prompts(tokenizer, [prompt], 4096)
    assert labels == [[ord("0"), ord("1")]]
    assert ids[0] + [labels[0][1]] == tokenizer.encode(prompt + "1")


@pytest.mark.parametrize("kind", ["prefix", "multiple", "identical"])
def test_chat_label_boundary_rejects_unsupported_tokenizers(kind):
    class Unsupported(Tokenizer):
        def encode(self, text, add_special_tokens=False):
            ids = super().encode(text, add_special_tokens)
            if text[-1:] in ("0", "1"):
                if kind == "prefix":
                    return ids[:-2] + [99]
                if kind == "multiple":
                    return ids + [99]
                return ids[:-1] + [99]
            return ids
    with pytest.raises(ValueError, match="prefix|distinct"):
        q.tokenize_prompts(Unsupported(), ["assistant\n\n"], 50)


def test_chat_overflow_is_rejected_instead_of_truncated():
    with pytest.raises(ValueError, match="truncation forbidden"):
        q.tokenize_prompts(Tokenizer(), ["assistant\n\n"], 3)


def make_model(path):
    path.mkdir(parents=True)
    blob = path.parent / ("a" * 64)
    blob.write_bytes(b"mock-weights")
    (path / "model-00001-of-00001.safetensors").symlink_to(blob)
    q.core.atomic_json(path / "model.safetensors.index.json", {
        "weight_map": {"lm_head.weight": "model-00001-of-00001.safetensors"}})
    (path / "config.json").write_text('{"model_type":"qwen3_5"}')
    return blob


def test_content_identity_ignores_cache_location_and_mtime_without_claiming_verification(tmp_path):
    first, second = tmp_path / "cache1" / ("b" * 40), tmp_path / "cache2" / ("b" * 40)
    blob = make_model(first)
    make_model(second)
    before = q.model_source(first)
    os.utime(blob, (100, 100))
    assert q.model_source(first) == before == q.model_source(second)
    assert "payload_not_hashed_by_evaluator" in next(iter(before["weight_verification"].values()))
    manifest = tmp_path / "weight_hashes.json"
    q.core.atomic_json(manifest, {"files": {"model-00001-of-00001.safetensors": {
        "sha256": "a" * 64, "bytes": len(b"mock-weights")}}})
    audited = q.model_source(first, manifest)
    assert "external_sha256_audit" in next(iter(audited["weight_verification"].values()))
    record = json.loads(manifest.read_text())
    record["files"]["model-00001-of-00001.safetensors"]["sha256"] = "c" * 64
    q.core.atomic_json(manifest, record)
    with pytest.raises(ValueError, match="disagrees"):
        q.model_source(first, manifest)


@pytest.mark.parametrize("field", ["missing_keys", "mismatched_keys", "error_msgs", "conversion_errors"])
def test_loader_refuses_incomplete_or_mismatched_weights(field):
    with pytest.raises(ValueError, match="loading failed"):
        q.validate_loading_info({field: ["bad"]})
    assert q.validate_loading_info({"unexpected_keys": {"mtp.norm.weight"}})["unexpected_keys"] == ["mtp.norm.weight"]
    with pytest.raises(ValueError, match="non-MTP"):
        q.validate_loading_info({"unexpected_keys": ["model.unexpected.weight"]})


def test_kernel_versions_record_missing_and_installed_distributions(monkeypatch):
    monkeypatch.setattr(q.core, "versions", lambda: {"torch": "test"})
    def version(name):
        if name == "causal-conv1d":
            raise q.importlib.metadata.PackageNotFoundError(name)
        return {"fla-core": "0.4.0", "einops": "0.8.2", "triton": "3.5.0"}[name]
    monkeypatch.setattr(q.importlib.metadata, "version", version)
    assert q.versions() == {"torch": "test", "fla-core": "0.4.0", "einops": "0.8.2",
                            "triton": "3.5.0", "causal-conv1d": None}


def test_partial_fast_path_records_actual_fused_norm_and_torch_convolution_bindings():
    class FusedNorm:
        pass
    class TorchNorm:
        pass
    def fla_chunk():
        pass
    def fla_recurrent():
        pass
    def torch_conv_update():
        pass
    class DeltaNet:
        def __init__(self, norm):
            self.norm = norm
            self.causal_conv1d_fn = None
            self.causal_conv1d_update = torch_conv_update
            self.chunk_gated_delta_rule = fla_chunk
            self.recurrent_gated_delta_rule = fla_recurrent
    architecture = SimpleNamespace(Qwen3_5GatedDeltaNet=DeltaNet, FusedRMSNormGated=FusedNorm,
                                   is_fast_path_available=False, causal_conv1d_fn=None, chunk_gated_delta_rule=fla_chunk)
    model = SimpleNamespace(modules=lambda: iter([object(), DeltaNet(FusedNorm()), DeltaNet(TorchNorm())]))
    info = q.kernel_metadata(model, architecture)
    assert info["all_optional_fast_path_available"] is False
    assert info["causal_conv1d_available"] is False
    assert info["flash_linear_attention_available"] is True
    assert info["fused_gated_norm_available"] is True
    assert info["linear_attention_layers"] == 2 and info["fused_gated_norm_layers"] == 1
    assert info["torch_causal_conv1d_prefill_layers"] == 2
    assert info["causal_conv1d_fn_bindings"] == ["torch.nn.Conv1d (prefill)"]
    assert info["chunk_gated_delta_rule_bindings"][0].endswith("fla_chunk")
    assert len(info["gated_norm_bindings"]) == 2


def input_args(tmp_path, name, reference=None, dry=False):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    for split in ("train", "test"):
        rows = [chr(65 + i) + SEQ[1:-1] + ("T" if split == "train" else "Z") for i in range(6)]
        (data / f"X_{split}_kip_m4.csv").write_text("Sequences\n" + "\n".join(rows) + "\n")
        (data / f"y_{split}_kip_m4.csv").write_text("Outcome\n0\n1\n0\n1\n0\n1\n")
    argv = ["--model-path", str(tmp_path / "model"), "--output-dir", str(tmp_path / name),
            "--kip-data-dir", str(data), "--tasks", "kip4", "--shots", "0,5", "--demo-seeds", "0,1",
            "--n-test", "4", "--batch-size", "2"]
    if reference:
        argv += ["--reference-run", str(reference)]
    if dry:
        argv += ["--dry-run"]
    return q.parse_args(argv)


def reference_run(args, reference):
    data, provenance = q.core.prepare_data(args)
    cells = q.core.build_cells(args.tasks, args.shots, args.demo_seeds)
    demos = {c["key"]: q.core.demonstration_payload(data[c["task"]], c) for c in cells}
    config = {"data": provenance}
    q.core.atomic_json(reference / "config.json", config)
    for task, current in data.items():
        q.core.atomic_json(reference / "samples" / f"{task}.json", {
            "indices": current["test_indices"], "labels": current["labels"].tolist(), "sequences": current["queries"]})
    for cell in cells:
        directory = reference / "cells" / cell["key"]
        sample = reference / "samples" / f"{cell['task']}.json"
        q.core.atomic_json(directory / "demonstrations.json", demos[cell["key"]])
        q.core.atomic_json(directory / "manifest.json", {"test_sample": {"path": str(sample), "sha256": q.core.sha256(sample)}})
        (directory / "predictions.csv").write_text("retained\n")
        q.core.atomic_json(directory / "metrics.json", {"status": "completed", "cell_identity": cell["key"],
            "run_identity": q.core.identity(config), "cell": cell,
            "artifact_sha256": {name: q.core.sha256(directory / name) for name in
                                ("predictions.csv", "manifest.json", "demonstrations.json")}})
    return data, provenance, cells, demos


def test_reference_requires_exact_indices_labels_sequences_and_demo_order(tmp_path):
    reference = tmp_path / "reference"
    args = input_args(tmp_path, "new", reference)
    data, provenance, cells, demos = reference_run(args, reference)
    assert len(q.check_reference(reference, data, provenance, cells, demos)["matched_completed_cells"]) == 3
    data["kip4"]["test_indices"] = list(reversed(data["kip4"]["test_indices"]))
    with pytest.raises(ValueError, match="test indices"):
        q.check_reference(reference, data, provenance, cells, demos)
    data["kip4"]["test_indices"].reverse()
    demos[cells[1]["key"]]["examples"].reverse()
    with pytest.raises(ValueError, match="demonstration"):
        q.check_reference(reference, data, provenance, cells, demos)


def test_dry_run_matches_reference_without_model_snapshot(tmp_path):
    args = input_args(tmp_path, "new", tmp_path / "reference", dry=True)
    reference_run(args, args.reference_run)
    result = q.run(args)
    assert result["status"] == "data_validated" and not result["weights_loaded"]
    assert len(result["config"]["reference_check"]["matched_completed_cells"]) == 3
    assert args.model_name == q.MODEL_NAME


def test_grid_loads_once_exports_chat_protocol_and_skips_identical_completed_run(tmp_path, monkeypatch):
    args = input_args(tmp_path, "new", tmp_path / "reference")
    reference_run(args, args.reference_run)
    make_model(args.model_path)
    calls = {"load": 0, "score": 0}
    monkeypatch.setattr(q, "load_tokenizer", lambda args: Tokenizer())
    def load(args, tokenizer):
        calls["load"] += 1
        return object(), {"mock": True}
    def score(model, ids, labels, pad, device):
        calls["score"] += 1
        return np.zeros((len(ids), 2)), np.zeros(len(ids)), np.full(len(ids), .5)
    monkeypatch.setattr(q, "load_model", load)
    monkeypatch.setattr(q.core, "score_batch", score)
    result = q.run(args)
    assert result["status"] == "completed" and len(result["cells"]) == 3
    assert calls == {"load": 1, "score": 6}
    config = json.loads((args.output_dir / "config.json").read_text())
    assert config["protocol"]["enable_thinking"] is False
    assert config["protocol"]["chat_template"] == Tokenizer.chat_template
    assert config["protocol"]["sample_rendered_prompt"].endswith("</think>\n\n")
    assert set(config["protocol"]["source_code_sha256"]) == {"qwen38_fewshot_eval.py", "qwen_fewshot_eval.py"}
    os.utime((args.model_path / "model-00001-of-00001.safetensors").resolve(), (100, 100))
    assert q.run(args) == result and calls == {"load": 1, "score": 6}
