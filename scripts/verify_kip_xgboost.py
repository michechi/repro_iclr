#!/usr/bin/env python3
"""Verify the completed 12-fit KIP XGBoost table from saved evidence (no GPU)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from run_kip_xgboost import DATA, OUT, SEEDS, sha, write


def main():
    results = []
    for encoding in ("basic", "llm"):
        for tag in ("kip_m4", "kip_m6"):
            expected_labels = pd.read_csv(DATA / f"y_test_{tag}.csv").Outcome.to_numpy(dtype=np.int8)
            for seed in SEEDS:
                directory = OUT / "runs" / encoding / tag / f"seed{seed}"
                result = json.loads((directory / "metrics.json").read_text())
                manifest = json.loads((directory / "manifest.json").read_text())
                saved = np.load(directory / "test_predictions.npz")
                np.testing.assert_array_equal(saved["labels"], expected_labels)
                assert np.isfinite(saved["probabilities"]).all()
                assert ((saved["probabilities"] >= 0) & (saved["probabilities"] <= 1)).all()
                auc = float(roc_auc_score(expected_labels, saved["probabilities"]))
                assert auc == result["test_auc"]
                assert not result["smoke"] and manifest["full_data"]
                assert (result["n_train"], result["n_val"], result["n_test"]) == (400000, 50000, 50000)
                assert result["seed"] == seed and result["tag"] == tag and result["encoding"] == encoding
                assert manifest["features"]["val"]["sha256_raw_array"] != manifest["features"]["test"]["sha256_raw_array"]
                dimensions = 20 if encoding == "basic" else 2048
                for split, rows in (("train", 400000), ("val", 50000), ("test", 50000)):
                    assert manifest["features"][split]["shape"] == [rows, dimensions]
                    assert sha(DATA / f"X_{split}_{tag}.csv") == manifest["data"][split]["features_sha256"]
                    assert sha(DATA / f"y_{split}_{tag}.csv") == manifest["data"][split]["labels_sha256"]
                history = json.loads((directory / "validation_history.json").read_text())["validation_0"]["logloss"]
                assert result["best_iteration"] == int(np.argmin(history))
                assert abs(result["best_validation_logloss"] - min(history)) < 1e-12
                assert len(history) == result["fitted_rounds"] <= 200
                assert result["model_reload_check"] == "passed"
                booster = xgb.Booster()
                booster.load_model(directory / "model.ubj")
                assert booster.num_boosted_rounds() == len(history)
                if encoding == "llm":
                    assert manifest["llama"]["frozen"]
                    assert manifest["llama"]["weights_and_outputs_dtype"] == "float32"
                    assert manifest["llama"]["float32_matmul_precision"] == "high"
                    assert all(check["token_length"] == 32 and check["batch_vs_single_check"].startswith("passed") for check in manifest["embedding_checks"].values())
                results.append(result)
    frame = pd.DataFrame(results)
    summary = frame.groupby(["encoding", "tag"]).test_auc.agg(n="count", auc_mean="mean", auc_sample_sd="std").reset_index()
    stored = pd.read_csv(OUT / "summary.csv")
    for row in summary.itertuples():
        assert row.n == 3
        recorded = stored[(stored.encoding == row.encoding) & (stored.tag == row.tag)].iloc[0]
        assert abs(row.auc_mean - recorded.auc_mean) < 1e-15
        assert abs(row.auc_sample_sd - recorded.auc_sample_sd) < 1e-15
    table = (OUT / "table.tex").read_text()
    for encoding, label in (("basic", "XGBoost (basic)"), ("llm", "XGBoost (LLM)")):
        line = next(line for line in table.splitlines() if line.startswith(label + " &"))
        fields = [field.strip() for field in line.removesuffix(r"\\").split("&")]
        assert len(fields) == 7
        for value, tag in zip(fields[-2:], ("kip_m4", "kip_m6")):
            row = summary[(summary.encoding == encoding) & (summary.tag == tag)].iloc[0]
            assert value == f"{row.auc_mean:.3f} ({row.auc_sample_sd:.3f})".replace("0.", ".")
    report = {
        "verified_utc": datetime.now(timezone.utc).isoformat(), "production_fits": len(results),
        "prediction_rows_verified": len(results) * 50000, "all_checks_passed": True,
        "checks": ["test labels match source", "AUC recomputed from saved predictions", "full split sizes", "input hashes", "distinct validation/test features", "validation-minimum checkpoint", "saved model tree counts", "three seeds and sample SD", "four LaTeX cells", "frozen model and pooling metadata"],
        "summary": summary.to_dict(orient="records"),
    }
    write(OUT / "verification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
