from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
        return out.stdout.strip()
    except Exception:
        return None


def _count_parquet(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_parquet(path, columns=[pd.read_parquet(path).columns[0]]))
    except Exception:
        return len(pd.read_parquet(path))


def assert_binary(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns:
        return f"missing column {col}"
    uniq = set(pd.unique(df[col].dropna()))
    if not uniq.issubset({0, 1}):
        return f"{col} has non-binary values: {sorted(uniq)}"
    return None


def assert_no_duplicates(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns:
        return f"missing column {col}"
    if df[col].duplicated().any():
        n = int(df[col].duplicated().sum())
        return f"{col} has {n} duplicates"
    return None


def assert_monotonic_within(df: pd.DataFrame, group_col: str, time_col: str) -> str | None:
    if group_col not in df.columns or time_col not in df.columns:
        return f"missing {group_col} or {time_col}"
    if df.empty:
        return None
    bad = (
        df.sort_values([group_col, time_col])
        .groupby(group_col)[time_col]
        .apply(lambda s: (s.diff().dt.total_seconds() < 0).any())
    )
    if bad.any():
        return f"{time_col} not monotonic within {group_col} for {int(bad.sum())} patients"
    return None


def assert_no_null(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns:
        return f"missing column {col}"
    n = int(df[col].isna().sum())
    if n:
        return f"{col} has {n} nulls"
    return None


def build_report(
    counts: dict[str, Any],
    sanity: list[str | None],
    leakage: list[str | None],
    addenda: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "counts": counts,
        "sanity": [msg for msg in sanity if msg],
        "leakage": [msg for msg in leakage if msg],
        "addenda": addenda,
        "config": {**config, "git_sha": _git_sha()},
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(report, f, indent=2, default=str)


def has_failures(report: dict[str, Any]) -> bool:
    return bool(report.get("sanity")) or bool(report.get("leakage"))
