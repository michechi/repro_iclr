"""Step 9: bag vs order baselines for both datasets.

The whole point of the pipeline is to quantify the gap between
semantic/bag-like and order-aware representations. This script trains
matched baselines with identical hyperparameters and prints AUROC /
AUPRC / F1 per dataset per representation.

Dataset A (strict order-only task):
  - bag representation:    patient-level counts / mean / max probs over
                           the context window (n_features ~ 31)
  - order representation:  concatenation of the first-progression and
                           first-response note flag vectors (20 binary
                           features). Because the strict subset has
                           exactly one of each, this trivially encodes
                           order: "which of the two comes first".

Dataset B (1-year mortality):
  - bag representation:    patient-level counts / mean / max probs +
                           n_notes + n_admissions + observation_window_days
  - order representation:  first-note flags + last-note flags + bag
                           (first / last capture "where the trajectory
                           started and where it ended")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mimic_cancer.dfci_imaging_model import LABEL_ORDER  # noqa: E402
from mimic_cancer.logging_utils import setup_logging  # noqa: E402
from mimic_cancer.paths import ensure_directories, load_paths  # noqa: E402

logger = setup_logging("09_run_baselines")

FLAG_COLS = [f"flag_{label}" for label in LABEL_ORDER]
PROB_COLS = [f"prob_{label}" for label in LABEL_ORDER]
CNT_COLS = [f"cnt_{label}" for label in LABEL_ORDER]
MEAN_COLS = [f"mean_{label}" for label in LABEL_ORDER]
MAX_COLS = [f"max_{label}" for label in LABEL_ORDER]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--order-dir",
        type=Path,
        default=None,
        help="Override Dataset A directory (strict if unset).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix appended to baseline names (used to distinguish strict vs. relaxed).",
    )
    parser.add_argument(
        "--skip-dataset-b",
        action="store_true",
        help="Only run Dataset A baselines (useful when running once per A cohort).",
    )
    return parser.parse_args()


def _ordered_notes_by_seq(seq: pd.DataFrame) -> pd.DataFrame:
    return seq.sort_values(["subject_id", "seq_idx"])


def _first_last_flags(seq: pd.DataFrame) -> pd.DataFrame:
    """Per subject, return a wide row with `first_flag_*` and `last_flag_*` columns."""
    s = _ordered_notes_by_seq(seq)
    first = s.groupby("subject_id").head(1).set_index("subject_id")[FLAG_COLS]
    last = s.groupby("subject_id").tail(1).set_index("subject_id")[FLAG_COLS]
    first = first.rename(columns={c: f"first_{c}" for c in first.columns})
    last = last.rename(columns={c: f"last_{c}" for c in last.columns})
    return first.join(last, how="outer").reset_index()


def _ds_a_order_features(core: pd.DataFrame) -> pd.DataFrame:
    """For Dataset A, `X_seq_core` has exactly two rows per subject
    (first progression + first response). Concatenate them in chronological
    order into a single 20-dim binary feature row per subject.
    """
    s = core.sort_values(["subject_id", "seq_idx"])
    rows = []
    for subj, g in s.groupby("subject_id"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            # fall back to zeros if something is off
            v = np.zeros(2 * len(FLAG_COLS), dtype=np.int8)
        else:
            v = np.concatenate([g.iloc[0][FLAG_COLS].to_numpy(dtype=np.int8),
                                g.iloc[1][FLAG_COLS].to_numpy(dtype=np.int8)])
        rows.append((subj, *v))
    cols = ["subject_id"] + [f"first_{c}" for c in FLAG_COLS] + [f"second_{c}" for c in FLAG_COLS]
    return pd.DataFrame(rows, columns=cols)


def _bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_boot: int = 1000,
    seed: int = 2026,
) -> tuple[float, float, float, float]:
    """Return (auroc_lo, auroc_hi, auprc_lo, auprc_hi) — 95% percentile CIs."""
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return (float("nan"),) * 4
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        p_b = proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aurocs.append(roc_auc_score(y_b, p_b))
        auprcs.append(average_precision_score(y_b, p_b))
    if not aurocs:
        return (float("nan"),) * 4
    return (
        float(np.percentile(aurocs, 2.5)),
        float(np.percentile(aurocs, 97.5)),
        float(np.percentile(auprcs, 2.5)),
        float(np.percentile(auprcs, 97.5)),
    )


def _fit_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    seed: int,
    n_boot: int = 1000,
) -> dict:
    if len(np.unique(y_train)) < 2:
        return {"name": name, "error": "train has one class"}
    scaler = StandardScaler(with_mean=False)  # safe for sparse / binary
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=2000, C=1.0, random_state=seed, solver="liblinear"
    )
    clf.fit(X_train_s, y_train)

    def metrics(y_true, X_used, split_name, with_ci: bool = False):
        if len(np.unique(y_true)) < 2:
            return {f"{split_name}_auroc": float("nan"), f"{split_name}_auprc": float("nan")}
        proba = clf.predict_proba(X_used)[:, 1]
        pred = (proba >= 0.5).astype(int)
        row = {
            f"{split_name}_auroc": float(roc_auc_score(y_true, proba)),
            f"{split_name}_auprc": float(average_precision_score(y_true, proba)),
            f"{split_name}_f1": float(f1_score(y_true, pred)),
            f"{split_name}_acc": float((pred == y_true).mean()),
            f"{split_name}_n": int(len(y_true)),
            f"{split_name}_pos_rate": float(y_true.mean()),
        }
        if with_ci:
            lo, hi, plo, phi = _bootstrap_ci(y_true, proba, n_boot=n_boot, seed=seed)
            row[f"{split_name}_auroc_ci_lo"] = lo
            row[f"{split_name}_auroc_ci_hi"] = hi
            row[f"{split_name}_auprc_ci_lo"] = plo
            row[f"{split_name}_auprc_ci_hi"] = phi
        return row

    out = {"name": name, "n_train_features": int(X_train.shape[1])}
    out.update(metrics(y_train, X_train_s, "train"))
    out.update(metrics(y_val, X_val_s, "val"))
    out.update(metrics(y_test, X_test_s, "test", with_ci=True))
    return out


def _split_xy(df: pd.DataFrame, split_col: str, label_col: str, feature_cols: list[str]):
    def pick(name: str):
        sub = df[df[split_col] == name]
        X = sub[feature_cols].to_numpy(dtype=np.float32)
        y = sub[label_col].to_numpy(dtype=np.int8)
        return X, y

    return pick("train"), pick("val"), pick("test")


def run_dataset_a(paths, seed: int, order_dir: Path | None = None, tag: str = "") -> list[dict]:
    a_dir = order_dir or paths.dataset_order_only_dir
    labels = pd.read_parquet(a_dir / "labels.parquet")
    bag = pd.read_parquet(a_dir / "X_bag_context.parquet")
    core = pd.read_parquet(a_dir / "X_seq_core.parquet")
    suffix = f"_{tag}" if tag else ""

    bag_features = ["n_notes"] + CNT_COLS + MEAN_COLS + MAX_COLS
    bag_frame = labels[["subject_id", "y_order", "split_order_task"]].merge(
        bag, on="subject_id", how="left"
    )
    order_wide = _ds_a_order_features(core)
    order_feature_cols = [c for c in order_wide.columns if c != "subject_id"]
    order_frame = labels[["subject_id", "y_order", "split_order_task"]].merge(
        order_wide, on="subject_id", how="left"
    )

    results: list[dict] = []

    (Xtr, ytr), (Xv, yv), (Xte, yte) = _split_xy(
        bag_frame, "split_order_task", "y_order", bag_features
    )
    results.append(
        _fit_eval(Xtr, ytr, Xv, yv, Xte, yte, f"dataset_a_bag{suffix}", seed)
    )

    (Xtr, ytr), (Xv, yv), (Xte, yte) = _split_xy(
        order_frame, "split_order_task", "y_order", order_feature_cols
    )
    results.append(
        _fit_eval(Xtr, ytr, Xv, yv, Xte, yte, f"dataset_a_order{suffix}", seed)
    )
    return results


def run_dataset_b(paths, seed: int) -> list[dict]:
    labels = pd.read_parquet(paths.dataset_mortality365_dir / "labels.parquet")
    bag = pd.read_parquet(paths.dataset_mortality365_dir / "X_bag.parquet")
    seq = pd.read_parquet(paths.dataset_mortality365_dir / "X_seq.parquet")

    bag_features = ["n_notes"] + CNT_COLS + MEAN_COLS + MAX_COLS
    extra_bag = ["n_notes_used", "n_admissions", "observation_window_days"]
    bag_frame = labels[
        ["subject_id", "y_death_365", "split_mortality_task"] + extra_bag
    ].merge(bag, on="subject_id", how="left")
    all_bag_features = bag_features + extra_bag

    first_last = _first_last_flags(seq)
    order_frame = bag_frame.merge(first_last, on="subject_id", how="left")
    order_feature_cols = all_bag_features + [
        c for c in first_last.columns if c != "subject_id"
    ]

    results: list[dict] = []

    (Xtr, ytr), (Xv, yv), (Xte, yte) = _split_xy(
        bag_frame, "split_mortality_task", "y_death_365", all_bag_features
    )
    results.append(
        _fit_eval(Xtr, ytr, Xv, yv, Xte, yte, "dataset_b_bag", seed)
    )

    (Xtr, ytr), (Xv, yv), (Xte, yte) = _split_xy(
        order_frame, "split_mortality_task", "y_death_365", order_feature_cols
    )
    results.append(
        _fit_eval(Xtr, ytr, Xv, yv, Xte, yte, "dataset_b_bag_plus_first_last", seed)
    )

    return results


def _print_table(results: list[dict]) -> None:
    rows = []
    for r in results:
        if "error" in r:
            logger.warning("%s: %s", r["name"], r["error"])
            continue
        auroc = r.get("test_auroc", float("nan"))
        lo = r.get("test_auroc_ci_lo", float("nan"))
        hi = r.get("test_auroc_ci_hi", float("nan"))
        auroc_str = f"{auroc:.3f} [{lo:.3f}, {hi:.3f}]"
        auprc = r.get("test_auprc", float("nan"))
        plo = r.get("test_auprc_ci_lo", float("nan"))
        phi = r.get("test_auprc_ci_hi", float("nan"))
        auprc_str = f"{auprc:.3f} [{plo:.3f}, {phi:.3f}]"
        rows.append(
            (
                r["name"],
                r["n_train_features"],
                r.get("test_n", 0),
                auroc_str,
                auprc_str,
                f"{r.get('test_f1', float('nan')):.3f}",
                f"{r.get('test_acc', float('nan')):.3f}",
            )
        )
    df = pd.DataFrame(
        rows,
        columns=["model", "n_feats", "n_test", "test_auroc_95ci", "test_auprc_95ci", "test_f1", "test_acc"],
    )
    print("\n" + df.to_string(index=False))


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    ensure_directories(paths)

    logger.info("running Dataset A baselines...")
    a_results = run_dataset_a(paths, args.seed, order_dir=args.order_dir, tag=args.tag)
    if args.skip_dataset_b:
        b_results: list[dict] = []
    else:
        logger.info("running Dataset B baselines...")
        b_results = run_dataset_b(paths, args.seed)

    all_results = a_results + b_results
    _print_table(all_results)

    out_path = paths.qc_dir / "baselines_report.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
