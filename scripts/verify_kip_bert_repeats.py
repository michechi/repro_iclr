#!/usr/bin/env python3
"""Independently verify four new BERT fits and six-seed table aggregation."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_kip_bert_repeats import OUT, DATA, JOBS, sha, write, ROOT


def verify(output=OUT):
    checks = []
    for tag, seed in JOBS:
        directory = output / "runs" / tag / f"seed{seed}"
        manifest = json.loads((directory / "manifest.json").read_text())
        result = json.loads((directory / "metrics.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        predictions = np.load(directory / "predictions.npz")
        assert result["tag"] == tag and result["seed"] == seed and not result["smoke"]
        assert [result[f"n_{split}"] for split in ("train", "val", "test")] == [400000, 50000, 50000]
        for split in ("test", "val"):
            labels = pd.read_csv(DATA / f"y_{split}_{tag}.csv").Outcome.to_numpy()
            prefix = "validation_" if split == "val" else ""
            np.testing.assert_array_equal(labels, predictions[prefix + "labels"])
            probabilities = predictions[prefix + "probabilities"]
            assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
            if split == "test":
                assert result["test_auc"] == float(roc_auc_score(labels, probabilities))
        for split in ("train", "val", "test"):
            assert manifest["data"][split]["features_sha256"] == sha(DATA / f"X_{split}_{tag}.csv")
            assert manifest["data"][split]["labels_sha256"] == sha(DATA / f"y_{split}_{tag}.csv")
        args = manifest["arguments"]
        assert args["max_length"] == (512 if tag == "kip_m4" else 64)
        assert args["batch_size"] == 16 and args["epochs"] == 20 and args["patience"] == 3
        assert args["lr"] == 2e-5 and args["early"] == "loss" and args["peft"]
        assert manifest["dtype"] == "float32" and not manifest["autocast"]
        assert manifest["trainable_parameters"] == 443906
        assert manifest["schedule"] == {"total_steps": 500000, "warmup_steps": 30000}
        losses = [row["val_loss"] for row in history]
        assert result["best_epoch"] == int(np.argmin(losses)) + 1
        assert result["best_validation_loss"] == min(losses)
        assert len(history) == result["epochs_completed"] <= 20
        assert len(history) == 20 or len(history) - result["best_epoch"] == 3
        checks.append(dict(tag=tag, seed=seed, test_auc=result["test_auc"], predictions_sha256=sha(directory / "predictions.npz")))
    frame = pd.read_csv(output / "per_seed.csv")
    summary = pd.read_csv(output / "summary.csv")
    table = (output / "table.tex").read_text()
    line = next(line for line in table.splitlines() if line.startswith("BERT &"))
    cells = [v.strip() for v in line.removesuffix(r"\\").split("&")][-2:]
    for tag, cell in zip(("kip_m4", "kip_m6"), cells):
        rows = frame[frame.tag == tag]
        assert sorted(rows.seed.tolist()) == [9550, 9551, 9552]
        old = rows[rows.seed == 9550].iloc[0]
        log = Path(old.source).read_text()
        assert f"test_auc={old.test_auc:.4f}" in log
        for seed in (9551, 9552):
            row = rows[rows.seed == seed].iloc[0]
            source = json.loads(Path(row.source).read_text())
            assert abs(row.test_auc - source["test_auc"]) < 1e-15
        mean, sd = rows.test_auc.mean(), rows.test_auc.std(ddof=1)
        recorded = summary[summary.tag == tag].iloc[0]
        assert recorded.n == 3
        assert abs(mean - recorded.auc_mean) < 1e-15 and abs(sd - recorded.auc_sample_sd) < 1e-15
        assert cell == f"{mean:.3f} ({sd:.3f})".replace("0.", ".")
    report = dict(all_checks_passed=True, verified_unix=time.time(), new_fits=4,
        saved_test_prediction_rows=200000, saved_validation_prediction_rows=200000,
        historical_seed_note="Seed9550 AUCs are four-decimal surviving log values; historical predictions are unavailable. Means/SD inherit this rounding and historical runtime differences.",
        checks=checks, summary=summary.to_dict(orient="records"))
    write(output / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
