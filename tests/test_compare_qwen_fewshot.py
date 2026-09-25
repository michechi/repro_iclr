import copy
import csv

import pytest

from scripts import compare_qwen_fewshot as comparison


def analyses():
    data = {task: {"tag": task, "test_indices": [2, 7], "sampled_test_positives": 1,
                   **{split: {kind: {"sha256": split + kind} for kind in ("features", "labels")}
                      for split in ("train", "test")}} for task in comparison.audit.TASKS}
    config = {"model_name": "Qwen/Qwen2.5-14B", "test_seed": 9950, "n_test": 2,
              "shots": [0, 5, 20], "demo_seeds": [0, 1, 2], "threshold": .5, "data": data,
              "prompt": "Sequential events: <20 space-separated letters>\nOutcome (0 or 1): "}
    old = {"config": config, "rows": {}, "samples": {"oc_det": {
        "test_seed": 9950, "indices": [2, 7], "labels": [0, 1], "sequences": ["AB", "BA"]}},
        "demonstrations": {}}
    for seed, auc in enumerate((.2, .4, .9)):
        key = f"oc_det/shots5/demo_seed{seed}"
        old["rows"][key] = {"auc": auc, "accuracy": .5, "f1": .5}
        old["demonstrations"][key] = {"examples": [{"index": 10, "sequence": "CD", "label": 0},
                                                      {"index": 20, "sequence": "DC", "label": 1}]}
    new = copy.deepcopy(old)
    new["config"].update(model_name="Qwen/Qwen3.8-27B", protocol={
        "prompt_mode": "official_chat_nonthinking", "enable_thinking": False,
        "model_training_stage": "posttrained"})
    new["rows"] = {"oc_det/shots5/demo_seed0": {"auc": .8, "accuracy": .5, "f1": .5}}
    return old, new


def test_partial_means_use_identical_seed_subset_and_keep_pending_cells():
    old, new = analyses()
    paired, groups = comparison.paired_tables(old, new, comparison.check_pairing(old, new))
    row = next(r for r in groups if r["task"] == "oc_det" and r["shots"] == 5)
    assert row["old_completed_cells"] == 3 and row["paired_cells"] == 1
    assert row["old_paired_auc_mean"] == .2
    assert row["new_paired_auc_mean"] == .8
    assert row["paired_auc_mean_delta_new_minus_old"] == pytest.approx(.6)
    assert row["paired_demo_seeds"] == "0" and row["status"] == "partial"
    assert len(paired) == 35 and len(groups) == 15
    unmatched = next(r for r in paired if r["cell_key"] == "oc_det/shots5/demo_seed1")
    assert unmatched["status"] == "old_only" and unmatched["delta_auc_new_minus_old"] == ""
    assert sum(r["shots"] == 0 for r in paired) == 5


@pytest.mark.parametrize("mismatch", ["labels", "demo_order", "source_hash", "planned_indices"])
def test_pairing_rejects_protocol_correspondence_failures(mismatch):
    old, new = analyses()
    if mismatch == "labels":
        new["samples"]["oc_det"]["labels"].reverse()
    elif mismatch == "demo_order":
        new["demonstrations"]["oc_det/shots5/demo_seed0"]["examples"].reverse()
    elif mismatch == "source_hash":
        new["config"]["data"]["kip6"]["train"]["labels"]["sha256"] = "changed"
    else:
        new["config"]["data"]["kip6"]["test_indices"] = [7, 2]
    with pytest.raises(ValueError, match="differ"):
        comparison.check_pairing(old, new)


def test_csv_rejects_changed_metric_and_duplicate_cells(tmp_path):
    path = tmp_path / "per_cell.csv"
    expected = [{"cell_key": "oc_det/shots0", "auc": .7}]

    def write(rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(expected[0]))
            writer.writeheader()
            writer.writerows(rows)

    write(expected)
    comparison.check_csv(path, expected, ("cell_key",))
    write([{"cell_key": "oc_det/shots0", "auc": .9}])
    with pytest.raises(ValueError, match="differs from source"):
        comparison.check_csv(path, expected, ("cell_key",))
    write(expected * 2)
    with pytest.raises(ValueError, match="duplicate CSV keys"):
        comparison.check_csv(path, expected * 2, ("cell_key",))
