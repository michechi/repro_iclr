"""Efficient four-noise evaluation for one Ordered Compliance baseline.

The generic evaluator is intentionally model-agnostic, so invoking it once per
noise level rebuilds the same 5.2-million-row hole feature matrix four times.
This baseline-specific runner reverses those loops: each resumable hole or
strict-pair chunk is enumerated and featurized once, then all requested
noise-trained estimators score that shared matrix before it is released.

Outputs use the same per-artifact run directories, shard schemas, metric file
names, and ``done.json`` contract as :mod:`src.oc_completion.ordered_eval`.
Consequently the report builder treats these runs exactly like generic
evaluator runs, and a partially completed generic run can be resumed here.

The default command evaluates the full fixed test manifests for all four noise
levels::

    python -m src.oc_completion.ordered_eval_baseline_grid \
        --family letter_count_logreg --pis 0.0,0.1,0.2,0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd

from src.oc_completion.ordered_baselines import (
    ALL_FAMILIES,
    TRAINED_FAMILIES,
)
from src.oc_completion.ordered_eval import (
    _artifact_id,
    _load_oracle_array,
    _manifest_sequence_column,
    _oracle_rows_for_manifest,
    _select_holes,
    _select_pairs,
    _tokens,
    enumerate_position_candidates,
)
from src.oc_completion.ordered_features import (
    DETERMINISTIC_OCCUPANCY_FAMILIES,
    build_features,
    deterministic_occupancy_score,
)
from src.oc_completion.ordered_io import (
    CHECKPOINT_ROOT,
    DATA_ROOT,
    PI_LEVELS,
    RESULT_ROOT,
    atomic_json_dump,
    pi_slug,
)
from src.oc_completion.ordered_metrics import (
    calibrate_tie_tolerance,
    position_localization_metrics,
    stability_metrics,
    strict_pair_metrics,
    valid_filling_metrics,
)
from src.oc_completion.oracle import ALPHABET, N_EVENTS, oc_label


@dataclass
class ArtifactContext:
    pi: float
    path: Path
    artifact: dict[str, Any]
    metadata: dict[str, Any]
    artifact_id: str
    run_dir: Path
    outputs: dict[str, str] = field(default_factory=dict)
    complete: bool = False


def _artifact_path(
    checkpoint_root: Path, family: str, pi: float, smoke: bool
) -> Path:
    suffix = "_smoke" if smoke else ""
    return (
        checkpoint_root
        / "baselines"
        / f"pi_{pi_slug(pi)}"
        / f"{family}{suffix}"
        / "artifact.joblib"
    )


def _artifact_metadata(
    artifact: dict[str, Any], path: Path, family: str, pi: float
) -> dict[str, Any]:
    return {
        "model": family,
        "family": family,
        "seed": int(artifact.get("seed", 9550)),
        "noise_pi": float(pi),
        "noise_level": float(pi),
        "checkpoint_epoch": 1,
        "validation_threshold": float(
            artifact.get("validation_threshold", 0.5)
        ),
        "training_mode": artifact.get("training_mode", "baseline"),
        "checkpoint_path": str(path),
    }


def _done_is_usable(path: Path, metadata: dict[str, Any]) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    if payload.get("status") != "complete":
        return False
    saved = payload.get("metadata", {})
    if str(saved.get("model")) != str(metadata["model"]):
        return False
    if float(saved.get("noise_pi", saved.get("noise_level", -1))) != metadata["noise_pi"]:
        return False
    if int(saved.get("seed", -1)) != metadata["seed"]:
        return False
    outputs = payload.get("outputs", {})
    if not outputs:
        return False
    return all(Path(value).is_file() for value in outputs.values())


def load_grid_artifacts(
    family: str,
    pis: Sequence[float],
    *,
    checkpoint_root: Path,
    result_root: Path,
    smoke: bool = False,
) -> list[ArtifactContext]:
    if family not in ALL_FAMILIES:
        raise KeyError(f"unknown baseline family {family!r}")
    contexts = []
    feature_family = None
    for pi in pis:
        path = _artifact_path(checkpoint_root, family, pi, smoke)
        if not path.is_file():
            raise FileNotFoundError(f"baseline artifact is missing: {path}")
        artifact = joblib.load(path)
        if artifact.get("family") != family:
            raise ValueError(
                f"artifact {path} has family={artifact.get('family')!r}, "
                f"expected {family!r}"
            )
        artifact_pi = float(artifact.get("noise_pi", pi))
        if artifact_pi != float(pi):
            raise ValueError(f"artifact {path} has pi={artifact_pi}, expected {pi}")
        if family in TRAINED_FAMILIES:
            current_feature = artifact.get(
                "feature_family", TRAINED_FAMILIES[family][0]
            )
            if feature_family is None:
                feature_family = current_feature
            elif current_feature != feature_family:
                raise ValueError("noise artifacts do not share one feature family")
        metadata = _artifact_metadata(artifact, path, family, pi)
        artifact_id = _artifact_id(metadata, pi, path)
        run_dir = result_root / "runs" / "evaluation" / artifact_id
        run_dir.mkdir(parents=True, exist_ok=True)
        complete = _done_is_usable(run_dir / "done.json", metadata)
        contexts.append(ArtifactContext(
            pi=float(pi),
            path=path,
            artifact=artifact,
            metadata=metadata,
            artifact_id=artifact_id,
            run_dir=run_dir,
            complete=complete,
        ))
    return contexts


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _sigmoid(decision: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(decision, dtype=np.float64), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _score_estimator(
    artifact: dict[str, Any], feature_matrix: Any
) -> tuple[np.ndarray, np.ndarray]:
    estimator = artifact.get("estimator", artifact.get("model"))
    if estimator is None:
        raise ValueError("trained baseline artifact has no estimator")
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(feature_matrix), dtype=np.float64)
        probability = _sigmoid(decision)
    else:
        probability64 = np.asarray(
            estimator.predict_proba(feature_matrix)[:, 1], dtype=np.float64
        )
        clipped = np.clip(probability64, 1e-12, 1.0 - 1e-12)
        decision = np.log(clipped / (1.0 - clipped))
        probability = probability64.astype(np.float32)
    return np.asarray(decision, dtype=np.float32), probability


def _score_shared_batch(
    family: str,
    contexts: Sequence[ArtifactContext],
    sequences: Sequence[str],
    *,
    oracle_decision: np.ndarray | None = None,
) -> tuple[dict[float, tuple[np.ndarray, np.ndarray]], int]:
    """Build one feature view, then score every requested noise artifact."""
    if not contexts:
        return {}, 0
    if family in TRAINED_FAMILIES:
        feature_family = contexts[0].artifact.get(
            "feature_family", TRAINED_FAMILIES[family][0]
        )
        feature_matrix = build_features(sequences, feature_family)
        return {
            context.pi: _score_estimator(context.artifact, feature_matrix)
            for context in contexts
        }, 1
    if family in DETERMINISTIC_OCCUPANCY_FAMILIES:
        decision = np.asarray(
            deterministic_occupancy_score(sequences, family), dtype=np.float32
        )
        probability = _sigmoid(decision)
        return {
            context.pi: (decision, probability) for context in contexts
        }, 1
    if family == "oracle":
        if oracle_decision is None:
            decision = np.asarray([oc_label(sequence) for sequence in sequences],
                                  dtype=np.float32)
        else:
            decision = np.asarray(oracle_decision, dtype=np.float32)
        if decision.shape != (len(sequences),):
            raise ValueError("oracle_decision is not aligned to candidate sequences")
        # The oracle's class probability is exact; this keeps fixed-zero and
        # fixed-one stability perfect at both allowed thresholds.
        probability = decision.copy()
        return {
            context.pi: (decision, probability) for context in contexts
        }, 1
    raise KeyError(family)


def _hole_candidate_batch(
    rows: pd.DataFrame,
    oracle: np.ndarray,
    sequence_column: str,
) -> tuple[list[str], pd.DataFrame]:
    candidates: list[str] = []
    frames = []
    for local_index, row in enumerate(rows.itertuples(index=False)):
        sequence = str(getattr(row, sequence_column))
        original = _tokens(sequence)
        for position in range(N_EVENTS):
            candidates.extend(enumerate_position_candidates(sequence, position))
        frames.append(pd.DataFrame({
            "base_sequence_id": np.repeat(
                getattr(row, "base_sequence_id"), N_EVENTS * len(ALPHABET)
            ),
            "base_Y_star": np.repeat(
                int(getattr(row, "base_Y_star")), N_EVENTS * len(ALPHABET)
            ),
            "position": np.repeat(
                np.arange(N_EVENTS, dtype=np.int8), len(ALPHABET)
            ),
            "original_letter": np.repeat(np.asarray(original), len(ALPHABET)),
            "candidate_letter": np.tile(np.asarray(ALPHABET), N_EVENTS),
            "candidate_Y_star": oracle[local_index].ravel().astype(np.int8),
        }))
    return candidates, pd.concat(frames, ignore_index=True)


def _valid_hole_shard(path: Path, expected_rows: int, context: ArtifactContext) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(
            path,
            columns=("positive_logit", "positive_probability", "model", "noise_level"),
        )
    except (OSError, ValueError):
        return False
    return (
        len(frame) == expected_rows
        and {"positive_logit", "positive_probability", "model", "noise_level"}
        <= set(frame.columns)
        and frame["model"].astype(str).eq(context.metadata["model"]).all()
        and np.allclose(frame["noise_level"].astype(float), context.pi)
    )


def _add_common_columns(frame: pd.DataFrame, context: ArtifactContext, split: str) -> None:
    common = {
        "model": context.metadata["model"],
        "noise_level": context.pi,
        "model_seed": context.metadata["seed"],
        "checkpoint_epoch": context.metadata["checkpoint_epoch"],
        "split": split,
    }
    for key, value in reversed(list(common.items())):
        frame.insert(0, key, value)


def evaluate_hole_grid(
    family: str,
    contexts: Sequence[ArtifactContext],
    *,
    manifest_path: Path,
    oracle_path: Path,
    chunk_bases: int,
    bootstrap_iters: int,
) -> int:
    """Evaluate fixed test holes, returning the number of feature builds."""
    if chunk_bases <= 0:
        raise ValueError("chunk_bases must be positive")
    all_manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
    if "manifest_index" not in all_manifest:
        all_manifest["manifest_index"] = np.arange(len(all_manifest))
    selected = _select_holes(all_manifest, "test").reset_index(drop=True)
    all_oracle = _load_oracle_array(oracle_path)
    oracle = _oracle_rows_for_manifest(all_manifest, selected, all_oracle)
    sequence_column = _manifest_sequence_column(selected)
    score_arrays = {
        context.pi: {
            "logits": np.empty_like(oracle, dtype=np.float32),
            "probabilities": np.empty_like(oracle, dtype=np.float32),
        }
        for context in contexts
    }
    feature_builds = 0

    for start in range(0, len(selected), chunk_bases):
        stop = min(start + chunk_bases, len(selected))
        expected_rows = (stop - start) * N_EVENTS * len(ALPHABET)
        missing: list[ArtifactContext] = []
        for context in contexts:
            shard = (
                context.run_dir / "hole_prediction_shards"
                / f"bases_{start:06d}_{stop:06d}.parquet"
            )
            if _valid_hole_shard(shard, expected_rows, context):
                old = pd.read_parquet(
                    shard, columns=("positive_logit", "positive_probability")
                )
                score_arrays[context.pi]["logits"][start:stop] = (
                    old.positive_logit.to_numpy(dtype=np.float32).reshape(
                        stop - start, N_EVENTS, len(ALPHABET)
                    )
                )
                score_arrays[context.pi]["probabilities"][start:stop] = (
                    old.positive_probability.to_numpy(dtype=np.float32).reshape(
                        stop - start, N_EVENTS, len(ALPHABET)
                    )
                )
            else:
                missing.append(context)
        if not missing:
            continue

        rows = selected.iloc[start:stop]
        candidates, common = _hole_candidate_batch(
            rows, oracle[start:stop], sequence_column
        )
        shared_scores, builds = _score_shared_batch(
            family,
            missing,
            candidates,
            oracle_decision=oracle[start:stop].ravel() if family == "oracle" else None,
        )
        feature_builds += builds
        for context in missing:
            decision, probability = shared_scores[context.pi]
            score_arrays[context.pi]["logits"][start:stop] = decision.reshape(
                stop - start, N_EVENTS, len(ALPHABET)
            )
            score_arrays[context.pi]["probabilities"][start:stop] = probability.reshape(
                stop - start, N_EVENTS, len(ALPHABET)
            )
            prediction = common.copy()
            prediction["positive_logit"] = decision
            prediction["positive_probability"] = probability
            prediction["model"] = context.metadata["model"]
            prediction["noise_level"] = context.pi
            prediction["model_seed"] = context.metadata["seed"]
            prediction["checkpoint_epoch"] = context.metadata["checkpoint_epoch"]
            shard = (
                context.run_dir / "hole_prediction_shards"
                / f"bases_{start:06d}_{stop:06d}.parquet"
            )
            _atomic_parquet(prediction, shard)

    for context in contexts:
        logits = score_arrays[context.pi]["logits"]
        probabilities = score_arrays[context.pi]["probabilities"]
        score_path = context.run_dir / "hole_scores_test.npz"
        np.savez_compressed(
            score_path,
            positive_logits=logits,
            probabilities=probabilities,
            base_sequence_ids=selected.base_sequence_id.to_numpy(),
        )
        position = position_localization_metrics(
            selected, oracle, logits, probabilities,
            bootstrap_iters=bootstrap_iters,
        )
        repair = valid_filling_metrics(
            selected, oracle, logits,
            bootstrap_iters=bootstrap_iters,
        )
        stability = stability_metrics(
            selected,
            oracle,
            logits,
            probabilities,
            {
                "fixed_0p5": 0.5,
                "validation_selected": context.metadata["validation_threshold"],
            },
            bootstrap_iters=bootstrap_iters,
        )
        for frame in (position, repair, stability):
            _add_common_columns(frame, context, "test")
        for name, frame in (
            ("hole_position_metrics", position),
            ("hole_repair_metrics", repair),
            ("hole_stability_metrics", stability),
        ):
            path = context.run_dir / f"{name}_test.csv"
            frame.to_csv(path, index=False)
            context.outputs[name] = str(path)
        context.outputs["hole_scores"] = str(score_path)
    return feature_builds


def _valid_strict_shard(path: Path, expected_rows: int, context: ArtifactContext) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_parquet(
            path,
            columns=("pair_id", "positive_logit", "negative_logit", "model",
                     "noise_level"),
        )
    except (OSError, ValueError):
        return False
    return (
        len(frame) == expected_rows
        and frame["model"].astype(str).eq(context.metadata["model"]).all()
        and np.allclose(frame["noise_level"].astype(float), context.pi)
    )


def _strict_scored_frame(
    pairs: pd.DataFrame,
    decision: np.ndarray,
    probability: np.ndarray,
    context: ArtifactContext,
) -> pd.DataFrame:
    n = len(pairs)
    scores = np.asarray(decision, dtype=np.float32).reshape(n, 2)
    probs = np.asarray(probability, dtype=np.float32).reshape(n, 2)
    positive_index = pairs.positive_index.to_numpy(dtype=np.int8)
    negative_index = 1 - positive_index
    row_index = np.arange(n)
    out = pairs.copy()
    out["cand0_positive_logit"] = scores[:, 0]
    out["cand1_positive_logit"] = scores[:, 1]
    out["cand0_positive_probability"] = probs[:, 0]
    out["cand1_positive_probability"] = probs[:, 1]
    out["positive_logit"] = scores[row_index, positive_index]
    out["negative_logit"] = scores[row_index, negative_index]
    out["margin"] = out.positive_logit - out.negative_logit
    out["strict_win"] = out.margin > 0.0
    out["tie"] = out.margin == 0.0
    out["loss"] = out.margin < 0.0
    out["model"] = context.metadata["model"]
    out["noise_level"] = context.pi
    out["model_seed"] = context.metadata["seed"]
    out["checkpoint_epoch"] = context.metadata["checkpoint_epoch"]
    return out


def evaluate_strict_grid(
    family: str,
    contexts: Sequence[ArtifactContext],
    *,
    manifest_path: Path,
    backgrounds: Sequence[str],
    pair_chunk: int,
    bootstrap_iters: int,
) -> int:
    """Evaluate strict test pairs, returning the number of feature builds."""
    if pair_chunk <= 0:
        raise ValueError("pair_chunk must be positive")
    manifest = pd.read_parquet(manifest_path)
    feature_builds = 0
    for background in backgrounds:
        pairs = _select_pairs(manifest, "test", background)
        pending = []
        for context in contexts:
            final_path = (
                context.run_dir
                / f"strict_pair_predictions_{background}_test.parquet"
            )
            if final_path.is_file():
                try:
                    final = pd.read_parquet(
                        final_path,
                        columns=("pair_id", "model", "noise_level"),
                    )
                except (OSError, ValueError):
                    final = pd.DataFrame()
                if (len(final) == len(pairs)
                        and final["model"].astype(str).eq(
                            context.metadata["model"]
                        ).all()
                        and np.allclose(final["noise_level"].astype(float), context.pi)):
                    continue
            pending.append(context)

        for start in range(0, len(pairs), pair_chunk):
            stop = min(start + pair_chunk, len(pairs))
            chunk = pairs.iloc[start:stop]
            missing = []
            for context in pending:
                shard = (
                    context.run_dir / "strict_prediction_shards"
                    / f"{background}_pairs_{start:06d}_{stop:06d}.parquet"
                )
                if not _valid_strict_shard(shard, stop - start, context):
                    missing.append(context)
            if not missing:
                continue
            candidates = np.column_stack([chunk.cand0, chunk.cand1]).ravel().tolist()
            oracle_decision = None
            if family == "oracle":
                positive_index = chunk.positive_index.to_numpy(dtype=np.int8)
                oracle_decision = np.column_stack(
                    [1 - positive_index, positive_index]
                ).ravel().astype(np.float32)
            shared_scores, builds = _score_shared_batch(
                family,
                missing,
                candidates,
                oracle_decision=oracle_decision,
            )
            feature_builds += builds
            for context in missing:
                decision, probability = shared_scores[context.pi]
                scored = _strict_scored_frame(
                    chunk, decision, probability, context
                )
                shard = (
                    context.run_dir / "strict_prediction_shards"
                    / f"{background}_pairs_{start:06d}_{stop:06d}.parquet"
                )
                _atomic_parquet(scored, shard)

        for context in contexts:
            prediction_path = (
                context.run_dir
                / f"strict_pair_predictions_{background}_test.parquet"
            )
            valid_final = False
            if prediction_path.is_file():
                try:
                    existing = pd.read_parquet(prediction_path)
                    valid_final = (
                        len(existing) == len(pairs)
                        and existing["pair_id"].is_unique
                        and existing["model"].astype(str).eq(
                            context.metadata["model"]
                        ).all()
                        and np.allclose(
                            existing["noise_level"].astype(float), context.pi
                        )
                    )
                except (OSError, ValueError, KeyError):
                    valid_final = False
            if valid_final:
                scored = existing
            else:
                shard_dir = context.run_dir / "strict_prediction_shards"
                shards = sorted(shard_dir.glob(f"{background}_pairs_*.parquet"))
                if not shards:
                    raise RuntimeError(
                        f"no strict prediction shards for {context.artifact_id}/"
                        f"{background}"
                    )
                scored = pd.concat(
                    [pd.read_parquet(path) for path in shards], ignore_index=True
                )
                if len(scored) != len(pairs) or not scored["pair_id"].is_unique:
                    raise RuntimeError(
                        f"strict shards are incomplete for {context.artifact_id}/"
                        f"{background}: {len(scored)}/{len(pairs)}"
                    )
                _atomic_parquet(scored, prediction_path)

            jitter_scores = np.repeat(
                scored[["positive_logit", "negative_logit"]]
                .to_numpy(dtype=np.float32)
                .ravel()[:128, None],
                4,
                axis=1,
            )
            jitter = calibrate_tie_tolerance(jitter_scores)
            tolerance = float(jitter["tie_tolerance"])
            # Baseline evaluation is deterministic; duplicated feature rows
            # are expected to be bit-identical.  Recompute classifications in
            # case a compatible generic-evaluator prediction file was reused.
            scored["strict_win"] = scored.margin > tolerance
            scored["tie"] = np.abs(scored.margin) <= tolerance
            scored["loss"] = scored.margin < -tolerance
            cluster = (
                "base_sequence_id"
                if background == "heldout" and "base_sequence_id" in scored
                else "pair_id"
            )
            metrics = strict_pair_metrics(
                scored,
                tolerance=tolerance,
                cluster_col=cluster,
                n_boot=bootstrap_iters,
            )
            metrics.update(jitter)
            metrics.update({
                "model": context.metadata["model"],
                "noise_level": context.pi,
                "model_seed": context.metadata["seed"],
                "checkpoint_epoch": context.metadata["checkpoint_epoch"],
                "split": "test",
                "background": background,
                "tie_tolerance_source": "repeated deterministic baseline scores",
            })
            metric_path = (
                context.run_dir
                / f"strict_pair_metrics_{background}_test.csv"
            )
            pd.DataFrame([metrics]).to_csv(metric_path, index=False)
            context.outputs[f"strict_predictions_{background}"] = str(
                prediction_path
            )
            context.outputs[f"strict_metrics_{background}"] = str(metric_path)
    return feature_builds


def evaluate_baseline_grid(
    family: str,
    pis: Sequence[float],
    *,
    data_root: Path = DATA_ROOT,
    checkpoint_root: Path = CHECKPOINT_ROOT,
    result_root: Path = RESULT_ROOT,
    hole_manifest: Path | None = None,
    hole_oracle: Path | None = None,
    strict_manifest: Path | None = None,
    backgrounds: Sequence[str] = ("clean", "heldout"),
    chunk_bases: int = 100,
    pair_chunk: int = 1_000,
    bootstrap_iters: int = 1_000,
    smoke: bool = False,
    command: str | None = None,
) -> dict[str, Any]:
    """Evaluate one family over multiple pi artifacts with shared features."""
    requested_pis = tuple(float(pi) for pi in pis)
    if not requested_pis or len(set(requested_pis)) != len(requested_pis):
        raise ValueError("pis must be a non-empty sequence of unique levels")
    if any(pi not in PI_LEVELS for pi in requested_pis):
        raise ValueError(f"pis must be drawn from {PI_LEVELS}")
    if bootstrap_iters <= 0:
        raise ValueError("bootstrap_iters must be positive")
    hole_manifest = Path(hole_manifest or result_root / "hole_manifest.parquet")
    hole_oracle = Path(hole_oracle or result_root / "hole_oracle_labels.npz")
    strict_manifest = Path(
        strict_manifest or result_root / "strict_pair_manifest.parquet"
    )
    for required in (hole_manifest, hole_oracle, strict_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"fixed evaluation manifest is missing: {required}")

    contexts = load_grid_artifacts(
        family,
        requested_pis,
        checkpoint_root=checkpoint_root,
        result_root=result_root,
        smoke=smoke,
    )
    active = [context for context in contexts if not context.complete]
    if not active:
        return {
            "family": family,
            "pis": list(requested_pis),
            "completed": len(contexts),
            "skipped_complete": len(contexts),
            "hole_feature_builds": 0,
            "strict_feature_builds": 0,
            "run_directories": [str(context.run_dir) for context in contexts],
        }

    hole_builds = evaluate_hole_grid(
        family,
        active,
        manifest_path=hole_manifest,
        oracle_path=hole_oracle,
        chunk_bases=chunk_bases,
        bootstrap_iters=bootstrap_iters,
    )
    strict_builds = evaluate_strict_grid(
        family,
        active,
        manifest_path=strict_manifest,
        backgrounds=tuple(backgrounds),
        pair_chunk=pair_chunk,
        bootstrap_iters=bootstrap_iters,
    )

    invocation = command or " ".join(sys.argv)
    for context in active:
        atomic_json_dump({
            "status": "complete",
            "artifact_id": context.artifact_id,
            "metadata": context.metadata,
            "outputs": context.outputs,
            "evaluation_mode": "baseline_shared_feature_grid",
            "shared_grid": {
                "family": family,
                "pis": list(requested_pis),
                "hole_split": "test",
                "strict_split": "test",
                "backgrounds": list(backgrounds),
                "hole_feature_builds": hole_builds,
                "strict_feature_builds": strict_builds,
                "data_root": str(data_root),
                "hole_manifest": str(hole_manifest),
                "hole_oracle": str(hole_oracle),
                "strict_manifest": str(strict_manifest),
            },
            "command": invocation,
        }, context.run_dir / "done.json")

    summary = {
        "family": family,
        "pis": list(requested_pis),
        "completed": len(active),
        "skipped_complete": len(contexts) - len(active),
        "hole_feature_builds": hole_builds,
        "strict_feature_builds": strict_builds,
        "run_directories": [str(context.run_dir) for context in contexts],
        "command": invocation,
    }
    summary_path = (
        result_root / "runs" / "baseline_grid"
        / f"{family}{'_smoke' if smoke else ''}.json"
    )
    atomic_json_dump(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary


def _parse_pis(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--pis must contain unique comma-separated levels")
    if any(value not in PI_LEVELS for value in values):
        raise argparse.ArgumentTypeError(f"--pis must be drawn from {PI_LEVELS}")
    return values


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=ALL_FAMILIES, required=True)
    parser.add_argument("--pis", type=_parse_pis, default=PI_LEVELS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--hole-manifest", type=Path)
    parser.add_argument("--hole-oracle", type=Path)
    parser.add_argument("--strict-manifest", type=Path)
    parser.add_argument("--backgrounds", default="clean,heldout")
    parser.add_argument("--chunk-bases", type=int, default=100)
    parser.add_argument("--pair-chunk", type=int, default=1_000)
    parser.add_argument("--bootstrap-iters", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    backgrounds = tuple(
        value.strip() for value in args.backgrounds.split(",") if value.strip()
    )
    if not backgrounds or any(value not in ("clean", "heldout")
                              for value in backgrounds):
        parser.error("--backgrounds must contain clean and/or heldout")
    summary = evaluate_baseline_grid(
        args.family,
        args.pis,
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        result_root=args.result_root,
        hole_manifest=args.hole_manifest,
        hole_oracle=args.hole_oracle,
        strict_manifest=args.strict_manifest,
        backgrounds=backgrounds,
        chunk_bases=args.chunk_bases,
        pair_chunk=args.pair_chunk,
        bootstrap_iters=args.bootstrap_iters,
        smoke=args.smoke,
        command=" ".join(sys.argv),
    )
    print(
        f"[ordered_eval_baseline_grid] {args.family}: "
        f"completed={summary['completed']} skipped={summary['skipped_complete']} "
        f"hole_feature_builds={summary['hole_feature_builds']} "
        f"strict_feature_builds={summary['strict_feature_builds']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
