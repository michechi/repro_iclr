import csv
import json

import pytest

from scripts import summarize_qwen_thinking as report


def trace(reason="thinking_completed"):
    generated = [80, 248069] if reason == "thinking_completed" else [80, 81] if reason == "token_cap" else [80, 248046]
    reasoning = generated if reason == "token_cap" else generated[:-1]
    return {"prompt_ids": [1, 2], "generated_ids": generated, "reasoning_ids": reasoning,
            "scoring_prefix_ids": [1, 2] + reasoning + [248069, 271], "label_ids": [15, 16],
            "max_new_tokens": 2, "batch_seed": 17, "generated_token_count": 2, "reasoning_token_count": len(reasoning),
            "stop_reason": reason, "stop_token_id": None if reason == "token_cap" else generated[-1],
            "thinking_completed": reason == "thinking_completed", "forced_close": reason != "thinking_completed",
            "forced_close_reason": None if reason == "thinking_completed" else reason}


@pytest.mark.parametrize("reason", ["thinking_completed", "token_cap", "early_eos"])
def test_trace_accepts_natural_and_explicit_forced_closure(reason):
    report.verify_trace(trace(reason), {"max_length": 16, "protocol": {"max_thinking_tokens": 2}}, 17)


@pytest.mark.parametrize("mutation", ["prefix", "label", "natural_flag", "hidden_stop", "short_cap"])
def test_trace_rejects_structurally_wrong_but_hashable_archive(mutation):
    record = trace("token_cap")
    if mutation == "prefix":
        record["scoring_prefix_ids"][2] = 999
    elif mutation == "label":
        record["label_ids"].reverse()
    elif mutation == "natural_flag":
        record["thinking_completed"] = True
    elif mutation == "hidden_stop":
        record["generated_ids"][0] = 248046
    else:
        record["generated_ids"] = [80]
        record["reasoning_ids"] = [80]
        record["generated_token_count"] = record["reasoning_token_count"] = 1
    with pytest.raises(ValueError):
        report.verify_trace(record, {"max_length": 16, "protocol": {"max_thinking_tokens": 2}}, 17)


def test_baseline_metrics_are_recomputed_on_exact_subset(tmp_path):
    path = tmp_path / "predictions.csv"
    # Full-sample AUROC is 0.75; selected positions [0, 1] have reversed ranking (AUROC 0).
    rows = [{"label": label, "log_odds_1_minus_0": odds, "probability_1": prob}
            for label, odds, prob in [(0, 0, .5), (1, -1, .268941421), (0, -2, .119202922), (1, 2, .880797078)]]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source = {"predictions_path": str(path), "auc": .75, "task": "kip6", "shots": 20, "demo_seed": 0}
    baseline = {"run": tmp_path, "rows": {"kip6/shots20/demo_seed0": source}}
    config = {"tasks": ["kip6"], "demo_seeds": [0], "reference_check": {"tasks": {"kip6": {"reference_positions": [0, 1]}}}}
    result = report.baseline_rows(baseline, config)
    assert len(result) == 1 and result[0]["rows"] == 2
    assert result[0]["auc"] == 0.0 and result[0]["accuracy"] == 0.0
    assert result[0]["metric_scope"] == "recomputed_on_new_fixed_test_subset"


def test_partial_comparison_uses_matched_seeds_and_counts_all_forced_queries():
    def row(shots, seed, auc):
        value = {"task": "kip6", "shots": shots, "demo_seed": seed, **{m: auc for m in report.audit.SCORES}}
        if shots == 40:
            value.update(natural_completion=1, token_cap=2, early_eos=1, forced_close=3, reasoning_tokens_total=7)
        return value
    rows = [row(20, 0, .2), row(20, 1, .9), row(40, 0, .4)]
    groups, delta = report.aggregate(rows, {"tasks": ["kip6"], "demo_seeds": [0, 1], "n_test": 4})
    thinking = next(r for r in groups if r["shots"] == 40)
    assert thinking["status"] == "partial" and thinking["completed_cells"] == 1
    assert thinking["scored_queries"] == 4 and thinking["forced_close"] == 3
    assert thinking["auc_sample_sd"] == ""
    assert delta[0]["nonthinking20_auc_mean"] == .2
    assert delta[0]["auc_delta_thinking40_minus_nonthinking20"] == pytest.approx(.2)


def test_reference_validation_rejects_reordered_full_sample_even_when_plans_agree(tmp_path):
    protocol = {"prompt_mode": "official_chat_nonthinking", "enable_thinking": False,
                "chat_template": "template", "chat_template_sha256": "hash", "instruction": "instruction",
                "source_code_sha256": {"qwen_fewshot_eval.py": "core", "qwen38_fewshot_eval.py": "native"}}
    sample = {"indices": [10, 20, 30, 40], "labels": [0, 0, 1, 1], "sequences": ["A", "B", "C", "D"], "test_seed": 9950}
    old = {"model_name": "Qwen/Qwen3.8-27B", "model_source": {}, "protocol": protocol, "test_seed": 9950,
           "demo_seeds": [0], "n_test": 4}
    baseline = {"run": tmp_path, "evidence": {"config_sha256": "config"}, "config": old, "samples": {"kip6": sample}}
    config = {**old, "tasks": ["kip6"], "data": {"kip6": {"test_indices": [20, 10, 30, 40], "sampled_test_positives": 2}},
              "reference_check": {"path": str(tmp_path), "config_sha256": "config", "tasks": {
                  "kip6": {"reference_positions": [1, 0, 2, 3]}}}}
    with pytest.raises(ValueError, match="fixed stratified sampling"):
        report.verify_reference(config, baseline)


def test_batch_audit_checks_csv_against_authoritative_token_archive(tmp_path):
    directory = tmp_path / "run/cells/kip6/shots40/demo_seed0"
    (directory / "batches").mkdir(parents=True)
    cell = {"task": "kip6", "shots": 40, "demo_seed": 0, "key": "kip6/shots40/demo_seed0"}
    config = {"n_test": 2, "batch_size": 2, "max_length": 16, "protocol": {"max_thinking_tokens": 2, "generation_seed": 9951}}
    seed = report.expected_seed(config, cell["key"], 0)
    rows = []
    for i, reason in enumerate(("thinking_completed", "token_cap")):
        record = trace(reason)
        record["batch_seed"] = seed
        rows.append({"test_row_index": [3, 7][i], "label": i, "probability_1": [.2, .8][i], "prediction_at_0_5": i,
                     "logit_0": 0., "logit_1": [-1., 1.][i], "log_odds_1_minus_0": [-1., 1.][i],
                     "prompt_tokens": 2, "scoring_tokens": len(record["scoring_prefix_ids"]), "thinking": record})
    payload = {"cell_identity": "cell", "batch_start": 0, "generation_seed": seed, "rows": rows,
               "rows_sha256": report.audit.identity(rows), "input_ids_sha256": report.audit.identity([[1, 2], [1, 2]]),
               "generation_seconds": 1., "scoring_seconds": .1}
    batch = directory / "batches/batch_000000.json"
    batch.write_text(json.dumps(payload))
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"indices": [3, 7], "labels": [0, 1]}))
    manifest = {"protocol": config["protocol"], "test_sample": {"path": str(sample)},
                "reasoning_batch_sha256": {"batches/batch_000000.json": report.audit.digest_file(batch)},
                "input_ids_sha256": payload["input_ids_sha256"], "token_lengths": {"min": 2, "max": 2, "all": [2, 2]}}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "metrics.json").write_text(json.dumps({"cell_identity": "cell", "thinking": {
        "stop_counts": {"thinking_completed": 1, "token_cap": 1}, "forced_close": 1, "natural_completion": 1, "reasoning_tokens_mean": 1.5}}))
    flat = [{k: row[k] if k in row else row["thinking"][k] for k in report.PREDICTION_COLUMNS} for row in rows]
    csv_path = directory / "predictions.csv"
    def write_csv():
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=report.PREDICTION_COLUMNS)
            writer.writeheader()
            writer.writerows(flat)
    write_csv()
    checked, evidence = report.verify_batches(directory, cell, config, {})
    assert checked["natural_completion"] == 1 and checked["forced_close"] == 1
    assert len(evidence) == 1
    flat[1]["thinking_completed"] = True
    write_csv()
    with pytest.raises(ValueError, match="CSV value differs"):
        report.verify_batches(directory, cell, config, {})
