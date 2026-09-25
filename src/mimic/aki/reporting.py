"""Audit and provenance reporting helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pandas as pd


def config_to_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "as_dict"):
        value = config.as_dict()
        if isinstance(value, Mapping):
            return dict(value)
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "to_dict"):
        value = config.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    raise TypeError("config must be a dataclass, mapping, or expose to_dict()")


def config_digest(config: Any) -> str:
    configured = getattr(config, "config_hash", None)
    if isinstance(configured, str) and configured:
        return configured
    payload = json.dumps(config_to_dict(config), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def build_episode_flow_report(episodes: pd.DataFrame) -> pd.DataFrame:
    """Headline phenotype/censor/exclusion counts and percentages."""

    required = {"status", "include_in_analysis", "reason_code"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"episode table missing audit columns: {sorted(missing)}")
    phenotype_column = "phenotype" if "phenotype" in episodes else "label"
    if phenotype_column not in episodes:
        raise ValueError("episode table must contain phenotype or label")
    total = len(episodes)
    report = (
        episodes.assign(
            phenotype=episodes[phenotype_column].fillna("<none>"),
            reason_code=episodes["reason_code"].fillna("<none>"),
        )
        .groupby(
            ["status", "include_in_analysis", "phenotype", "reason_code"],
            dropna=False,
        )
        .size()
        .rename("n_episodes")
        .reset_index()
    )
    report["fraction_of_all_episodes"] = report["n_episodes"] / total if total else float("nan")
    return report.sort_values(
        ["include_in_analysis", "status", "phenotype", "reason_code"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def build_class_distribution(
    episodes: pd.DataFrame,
    *,
    split_column: str = "split",
) -> pd.DataFrame:
    phenotype_column = "phenotype" if "phenotype" in episodes else "label"
    required = {phenotype_column, "subject_id"}
    missing = required.difference(episodes.columns)
    if missing:
        raise ValueError(f"episode table missing columns: {sorted(missing)}")
    group_columns = [phenotype_column]
    if split_column in episodes:
        group_columns.insert(0, split_column)
    result = (
        episodes.groupby(group_columns, dropna=False)
        .agg(n_episodes=("subject_id", "size"), n_subjects=("subject_id", "nunique"))
        .reset_index()
    )
    if len(group_columns) > 1:
        denominator = result.groupby(group_columns[:-1])["n_episodes"].transform("sum")
    else:
        denominator = result["n_episodes"].sum()
    result["episode_fraction"] = result["n_episodes"] / denominator
    return result


def write_run_manifest(
    path: Path,
    *,
    config: Any,
    repo_root: Path,
    artifacts: Mapping[str, str | Path],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(repo_root),
        "config_sha256": config_digest(config),
        "config": config_to_dict(config),
        "artifacts": {name: str(value) for name, value in artifacts.items()},
    }
    if extra:
        manifest.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    return manifest
