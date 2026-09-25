"""Synthetic, dependency-free integration tests for ordered_report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.oc_completion import ordered_report as report
from src.oc_completion.oracle import ALPHABET


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_time_constrained_inventory_uses_one_hf_seed_only():
    expected = pd.DataFrame(report._expected_configurations())
    training = expected[expected.stage == "training"]
    hf = training[training.kind == "hf"]
    scratch = training[training.kind == "scratch"]

    assert set(hf.model_seed) == {9950}
    assert len(hf[hf.model == "bert_lora"]) == 2
    assert len(hf[hf.model == "llama_lora"]) == 2
    assert set(hf.noise_level) == {0.0, 0.3}
    assert set(scratch.model_seed) == {9550, 9551, 9552}


def test_optional_qlora_model_names_are_normalised_for_reporting():
    assert report._normalise_model("meta-llama/Llama-3.1-8B") == "llama8b_qlora"
    assert report._normalise_model("Qwen/Qwen2.5-14B") == "qwen14b_qlora"
    assert report._normalise_model("meta-llama/Llama-3.1-70B") == "llama70b_qlora"


def _training_run(
    checkpoint_root: Path,
    *,
    model: str,
    pi: float,
    seed: int,
    observed_auc: float,
    latent_auc: float,
) -> Path:
    tag = f"{pi:.1f}".replace(".", "p")
    run = checkpoint_root / model.lower() / f"pi_{tag}" / f"seed_{seed}"
    run.mkdir(parents=True)
    _write_json(run / "config.json", {
        "model_name": model, "noise_pi": pi, "model_seed": seed,
        "split_rows": {"train": 100, "val": 20, "test": 20},
        "command": f"train --model {model} --noise-pi {pi} --seed {seed}",
    })
    _write_json(run / "done.json", {
        "status": "complete", "model_name": model, "noise_pi": pi,
        "model_seed": seed, "best_epoch": 2, "epochs_completed": 3,
        "validation_selected_threshold": 0.42,
        "best_validation_f1": 0.8,
        "test_observed_auc": observed_auc, "test_latent_auc": latent_auc,
        "test_observed_f1": 0.75, "test_observed_precision": 0.8,
        "test_observed_recall": 0.7, "training_time_seconds": 12.5,
    })
    pd.DataFrame([
        {"epoch": 1, "train_loss": 0.7, "validation_observed_auc": 0.7,
         "validation_latent_auc": 0.71, "validation_observed_f1": 0.6,
         "validation_selected_threshold": 0.4},
        {"epoch": 2, "train_loss": 0.5, "validation_observed_auc": 0.8,
         "validation_latent_auc": 0.82, "validation_observed_f1": 0.75,
         "validation_selected_threshold": 0.42},
    ]).to_csv(run / "history.csv", index=False)
    pd.DataFrame([
        {"epoch": 1, "standard_validation_f1": 0.6,
         "strict_matched_pair_accuracy": 0.55,
         "hole_position_localization_auroc": 0.6},
        {"epoch": 2, "standard_validation_f1": 0.75,
         "strict_matched_pair_accuracy": 0.7,
         "hole_position_localization_auroc": 0.75},
    ]).to_csv(run / "epoch_diagnostics.csv", index=False)
    return run


def _baseline_run(checkpoint_root: Path) -> None:
    run = checkpoint_root / "baselines" / "pi_0p0" / "letter_count_logreg"
    _write_json(run / "result.json", {
        "model": "letter_count_logreg", "noise_pi": 0.0, "seed": 9550,
        "training_mode": "baseline", "train_rows": 100,
        "validation_threshold": 0.5, "validation_f1": 0.6,
        "standard_observed_auc": 0.65, "standard_latent_auc": 0.65,
        "standard_observed_f1": 0.6, "training_time_s": 1.0,
        "checkpoint": str(run / "artifact.joblib"),
    })
    _write_json(run / "done.json", {
        "status": "complete", "family": "letter_count_logreg",
        "noise_pi": 0.0, "seed": 9550,
    })


def _hole_rows(model: str, pi: float, seed: int) -> pd.DataFrame:
    definitions = (
        ("create", 0, np.r_[np.zeros(13), np.ones(13)]),
        ("destroy", 1, np.r_[np.zeros(13), np.ones(13)]),
        ("fixed0", 0, np.zeros(26)),
        ("fixed1", 1, np.ones(26)),
    )
    rows = []
    for position, (base_id, base_y, labels) in enumerate(definitions):
        for index, (letter, label) in enumerate(zip(ALPHABET, labels)):
            logit = (index - 12.5) / 10
            rows.append({
                "base_sequence_id": base_id, "base_Y_star": base_y,
                "position": position, "original_letter": "A",
                "candidate_letter": letter, "candidate_Y_star": int(label),
                "positive_logit": logit,
                "positive_probability": 1 / (1 + np.exp(-logit)),
                "model": model, "noise_level": pi, "model_seed": seed,
                "checkpoint_epoch": 2,
            })
    return pd.DataFrame(rows)


def _evaluation_run(
    results_root: Path,
    *,
    artifact: str,
    model: str,
    pi: float,
    seed: int,
    prediction_rows: bool = False,
) -> None:
    run = results_root / "runs" / "evaluation" / artifact
    run.mkdir(parents=True)
    common = {"model": model, "noise_level": pi, "model_seed": seed,
              "checkpoint_epoch": 2, "split": "test"}
    pd.DataFrame([{**common, "segment": "all", "position_auroc": 0.8,
                   "position_auprc": 0.7}]).to_csv(
        run / "hole_position_metrics_test.csv", index=False)
    pd.DataFrame([
        {**common, "segment": "target_0", "macro_candidate_auc": 0.75,
         "pooled_candidate_auc": 0.74, "top1_valid_filling_accuracy": 0.7,
         "tie_aware_top1_accuracy": 0.7, "random_choice_reference": 0.5},
        {**common, "segment": "target_1", "macro_candidate_auc": 0.85,
         "pooled_candidate_auc": 0.84, "top1_valid_filling_accuracy": 0.8,
         "tie_aware_top1_accuracy": 0.8, "random_choice_reference": 0.5},
    ]).to_csv(run / "hole_repair_metrics_test.csv", index=False)
    pd.DataFrame([
        {**common, "category": "fixed_zero", "threshold_name": "validation_selected",
         "n_positions": 20, "prediction_flip_rate": 0.1,
         "mean_logit_range": 0.2, "mean_probability_range": 0.1},
        {**common, "category": "fixed_one", "threshold_name": "validation_selected",
         "n_positions": 10, "prediction_flip_rate": 0.4,
         "mean_logit_range": 0.5, "mean_probability_range": 0.3},
    ]).to_csv(run / "hole_stability_metrics_test.csv", index=False)
    pd.DataFrame([{**common, "background": "heldout", "pair_accuracy": 0.9,
                   "pair_accuracy_ci_lo": 0.87, "pair_accuracy_ci_hi": 0.93,
                   "strict_win_rate": 0.89, "tie_rate": 0.02,
                   "mean_margin": 1.2, "median_margin": 1.1,
                   "flattened_candidate_auc": 0.92, "tie_tolerance": 1e-6}]).to_csv(
        run / "strict_pair_metrics_heldout_test.csv", index=False)
    if prediction_rows:
        shard = run / "hole_prediction_shards"
        shard.mkdir()
        _hole_rows(model, pi, seed).to_parquet(shard / "bases_000000_000004.parquet", index=False)
        pd.DataFrame([{"pair_id": "pair0", "base_sequence_id": "base0",
                       "background": "heldout", "split": "test", "positive_index": 1,
                       "positive_logit": 1.0, "negative_logit": 0.0,
                       "margin": 1.0, "strict_win": True, "tie": False,
                       "loss": False}]).to_parquet(
            run / "strict_pair_predictions_heldout_test.parquet", index=False)
    _write_json(run / "done.json", {
        "status": "complete", "artifact_id": artifact,
        "metadata": {"model": model, "noise_pi": pi, "seed": seed,
                     "checkpoint_epoch": 2},
        "outputs": {},
    })


def test_report_discovers_only_completed_runs_and_is_idempotent(tmp_path):
    results = tmp_path / "results" / "oc_hole_audit"
    checkpoints = results / "checkpoints"
    data = tmp_path / "data"
    _write_json(data / "dataset_manifest.json", {
        "command": "generate --synthetic", "sizes": {"train": 100}
    })
    _write_json(data / "noise_manifest.json", {"noise_levels": [0.0, 0.1, 0.2, 0.3]})

    _training_run(checkpoints, model="LSTM", pi=0.0, seed=9550,
                  observed_auc=0.97, latent_auc=0.97)
    _training_run(checkpoints, model="LSTM", pi=0.0, seed=9551,
                  observed_auc=0.99, latent_auc=0.99)
    _training_run(checkpoints, model="LSTM", pi=0.1, seed=9550,
                  observed_auc=0.90, latent_auc=0.96)
    # A present expected directory without a completion marker is inventoried
    # as failed/incomplete and must not become a result row.
    _write_json(checkpoints / "transformer" / "pi_0p0" / "seed_9550" / "config.json",
                {"model_name": "Transformer", "noise_pi": 0.0, "model_seed": 9550})
    _baseline_run(checkpoints)

    _evaluation_run(results, artifact="lstm0s0", model="LSTM", pi=0.0,
                    seed=9550, prediction_rows=True)
    _evaluation_run(results, artifact="lstm0s1", model="LSTM", pi=0.0, seed=9551)
    _evaluation_run(results, artifact="lstm1s0", model="LSTM", pi=0.1, seed=9550)
    _evaluation_run(results, artifact="count0", model="letter_count_logreg",
                    pi=0.0, seed=9550)

    first = report.build_report(
        results_root=results, checkpoint_root=checkpoints, data_root=data,
        reporter_command="ordered-report --synthetic",
    )
    second = report.build_report(
        results_root=results, checkpoint_root=checkpoints, data_root=data,
        reporter_command="ordered-report --synthetic",
    )

    assert first["completed_training_rows"] == 3
    assert second["completed_training_rows"] == 3
    assert first["hole_prediction_rows"] == 104
    assert second["hole_prediction_rows"] == 104
    assert first["strict_prediction_rows"] == 1
    assert len(pd.read_parquet(results / "hole_predictions.parquet")) == 104
    strict_predictions = pd.read_parquet(results / "strict_pair_predictions.parquet")
    assert len(strict_predictions) == 1
    assert strict_predictions.model.iloc[0] == "LSTM"
    assert strict_predictions.noise_level.iloc[0] == 0.0
    assert strict_predictions.model_seed.iloc[0] == 9550

    training = pd.read_csv(results / "training_results.csv")
    assert len(training) == 3
    assert set(training.model) == {"LSTM"}
    flagged = training[(training.noise_level == 0.0) & (training.model_seed == 9550)]
    assert flagged.paper_auc_deviation_gt_0p02.iloc[0]

    summary = pd.read_csv(results / "model_summary.csv")
    pi_zero = summary[summary.noise_level == 0.0]
    assert len(pi_zero) == 2
    np.testing.assert_allclose(pi_zero.standard_observed_auc_seed_mean, 0.98)
    assert set(pi_zero.standard_observed_auc_seed_n) == {2}
    assert "strict_pair_accuracy_ci_lo" in summary

    adjacent = pd.read_csv(results / "adjacent_noise_differences.csv")
    row = adjacent[(adjacent.model_seed == 9550) & (adjacent.from_noise == 0.0)].iloc[0]
    assert np.isclose(row.delta_standard_observed_auc, -0.07)

    baselines = pd.read_csv(results / "baseline_results.csv")
    assert set(baselines.model) == {"letter_count_logreg"}
    assert baselines.strict_pair_accuracy.iloc[0] == 0.9

    inventory = pd.read_csv(results / "configuration_inventory.csv")
    transformer = inventory[(inventory.stage == "training")
                            & (inventory.model == "Transformer")
                            & (inventory.noise_level == 0.0)
                            & (inventory.model_seed == 9550)]
    assert transformer.status.iloc[0] == "failed"

    required = {
        "training_results.csv", "training_history.csv", "epoch_diagnostics.csv",
        "hole_position_metrics.csv", "hole_repair_metrics.csv",
        "hole_stability_metrics.csv", "strict_pair_metrics.csv",
        "baseline_results.csv", "model_summary.csv", "report.md",
        "table_main.tex", "table_baselines.tex",
    }
    assert required.issubset({path.name for path in results.iterdir()})
    figures = sorted((results / "figures").glob("figure_*.svg"))
    assert len(figures) == 9
    assert all("<svg" in path.read_text(encoding="utf-8") for path in figures)
    report_text = (results / "report.md").read_text(encoding="utf-8")
    assert "generate --synthetic" in report_text
    assert "ordered-report --synthetic" in report_text
    assert "no result is inferred" in report_text
