"""Step 12: two-panel summary figure.

Left panel: Dataset A, strict + relaxed cohorts, bag vs order logistic regression.
Right panel: Dataset B, bag LR / bag+first-last LR / transformer ordered / transformer shuffled.

All AUROCs are recomputed fresh from the processed parquet files so the
figure stays in sync with the current data. Error bars are percentile
bootstrap 95% CIs on the test set (1000 resamples).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from mimic_cancer.dfci_imaging_model import LABEL_ORDER  # noqa: E402
from mimic_cancer.logging_utils import setup_logging  # noqa: E402
from mimic_cancer.paths import ensure_directories, load_paths  # noqa: E402

logger = setup_logging("12_make_summary_figure")

CNT_COLS = [f"cnt_{label}" for label in LABEL_ORDER]
MEAN_COLS = [f"mean_{label}" for label in LABEL_ORDER]
MAX_COLS = [f"max_{label}" for label in LABEL_ORDER]
FLAG_COLS = [f"flag_{label}" for label in LABEL_ORDER]
BAG_FEATURES_A = ["n_notes"] + CNT_COLS + MEAN_COLS + MAX_COLS
BAG_FEATURES_B = (
    ["n_notes"] + CNT_COLS + MEAN_COLS + MAX_COLS
    + ["n_notes_used", "n_admissions", "observation_window_days"]
)


def _bootstrap_ci(
    y_true: np.ndarray, proba: np.ndarray, n_boot: int = 1000, seed: int = 2026
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y_true[idx], proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        aurocs.append(roc_auc_score(yb, pb))
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def _fit_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 2026,
) -> tuple[float, float, float]:
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(
        max_iter=2000, C=1.0, random_state=seed, solver="liblinear"
    )
    clf.fit(X_train_s, y_train)
    proba = clf.predict_proba(X_test_s)[:, 1]
    auroc = float(roc_auc_score(y_test, proba))
    lo, hi = _bootstrap_ci(y_test, proba, seed=seed)
    return auroc, lo, hi


def _dataset_a_split(labels: pd.DataFrame, X: pd.DataFrame, feat_cols: list[str]):
    train_mask = labels["split_order_task"].to_numpy() == "train"
    test_mask = labels["split_order_task"].to_numpy() == "test"
    y = labels["y_order"].to_numpy()
    Xm = X.to_numpy(dtype=np.float32)
    return Xm[train_mask], y[train_mask], Xm[test_mask], y[test_mask]


def _build_a_order_features(core: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subj, g in core.sort_values(["subject_id", "seq_idx"]).groupby("subject_id"):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            v = np.zeros(2 * len(FLAG_COLS), dtype=np.int8)
        else:
            v = np.concatenate(
                [
                    g.iloc[0][FLAG_COLS].to_numpy(dtype=np.int8),
                    g.iloc[1][FLAG_COLS].to_numpy(dtype=np.int8),
                ]
            )
        rows.append((subj, *v))
    cols = (
        ["subject_id"]
        + [f"first_{c}" for c in FLAG_COLS]
        + [f"second_{c}" for c in FLAG_COLS]
    )
    return pd.DataFrame(rows, columns=cols)


def dataset_a_results(paths, which: str) -> dict:
    directory = (
        paths.dataset_order_only_dir if which == "strict" else paths.dataset_order_only_relaxed_dir
    )
    labels = pd.read_parquet(directory / "labels.parquet")
    bag = pd.read_parquet(directory / "X_bag_context.parquet")
    core = pd.read_parquet(directory / "X_seq_core.parquet")
    n_total = len(labels)

    # BAG
    merged = labels[["subject_id", "y_order", "split_order_task"]].merge(
        bag, on="subject_id", how="left"
    )
    X = merged[BAG_FEATURES_A]
    X_tr, y_tr, X_te, y_te = _dataset_a_split(merged, X, BAG_FEATURES_A)
    bag_auroc, bag_lo, bag_hi = _fit_lr(X_tr, y_tr, X_te, y_te)

    # ORDER
    order_wide = _build_a_order_features(core)
    order_merged = labels[["subject_id", "y_order", "split_order_task"]].merge(
        order_wide, on="subject_id", how="left"
    )
    order_cols = [c for c in order_wide.columns if c != "subject_id"]
    X_tr, y_tr, X_te, y_te = _dataset_a_split(order_merged, order_merged[order_cols], order_cols)
    ord_auroc, ord_lo, ord_hi = _fit_lr(X_tr, y_tr, X_te, y_te)

    return {
        "cohort": which,
        "n_patients": n_total,
        "bag_auroc": bag_auroc, "bag_lo": bag_lo, "bag_hi": bag_hi,
        "order_auroc": ord_auroc, "order_lo": ord_lo, "order_hi": ord_hi,
    }


def _first_last_flags(seq: pd.DataFrame) -> pd.DataFrame:
    s = seq.sort_values(["subject_id", "seq_idx"])
    first = s.groupby("subject_id").head(1).set_index("subject_id")[FLAG_COLS]
    last = s.groupby("subject_id").tail(1).set_index("subject_id")[FLAG_COLS]
    first = first.rename(columns={c: f"first_{c}" for c in first.columns})
    last = last.rename(columns={c: f"last_{c}" for c in last.columns})
    return first.join(last, how="outer").reset_index()


def dataset_b_bag_results(paths) -> dict:
    labels = pd.read_parquet(paths.dataset_mortality365_dir / "labels.parquet")
    bag = pd.read_parquet(paths.dataset_mortality365_dir / "X_bag.parquet")
    seq = pd.read_parquet(paths.dataset_mortality365_dir / "X_seq.parquet")

    extra = ["n_notes_used", "n_admissions", "observation_window_days"]
    bag_frame = labels[["subject_id", "y_death_365", "split_mortality_task"] + extra].merge(
        bag, on="subject_id", how="left"
    )
    all_bag = BAG_FEATURES_B

    def _split(frame, cols):
        tr = frame["split_mortality_task"].to_numpy() == "train"
        te = frame["split_mortality_task"].to_numpy() == "test"
        y = frame["y_death_365"].to_numpy()
        X = frame[cols].to_numpy(dtype=np.float32)
        return X[tr], y[tr], X[te], y[te]

    # bag LR
    X_tr, y_tr, X_te, y_te = _split(bag_frame, all_bag)
    bag_auroc, bag_lo, bag_hi = _fit_lr(X_tr, y_tr, X_te, y_te)

    # bag + first/last LR
    fl = _first_last_flags(seq)
    order_frame = bag_frame.merge(fl, on="subject_id", how="left")
    order_cols = all_bag + [c for c in fl.columns if c != "subject_id"]
    X_tr, y_tr, X_te, y_te = _split(order_frame, order_cols)
    fl_auroc, fl_lo, fl_hi = _fit_lr(X_tr, y_tr, X_te, y_te)

    return {
        "bag_auroc": bag_auroc, "bag_lo": bag_lo, "bag_hi": bag_hi,
        "fl_auroc": fl_auroc, "fl_lo": fl_lo, "fl_hi": fl_hi,
    }


def dataset_b_transformer_results(paths) -> dict:
    rep = json.loads((paths.qc_dir / "sequence_model_b_report.json").read_text())
    out: dict[str, float] = {}
    for entry in rep:
        tag = "ordered" if entry["name"].endswith("ordered") else "shuffled"
        out[f"{tag}_auroc"] = entry["test_auroc"]
        out[f"{tag}_lo"] = entry["test_auroc_ci_lo"]
        out[f"{tag}_hi"] = entry["test_auroc_ci_hi"]
    return out


def _errbar(lo: float, hi: float, value: float) -> tuple[float, float]:
    """Return (lower_length, upper_length) for matplotlib yerr (asymmetric)."""
    return max(0.0, value - lo), max(0.0, hi - value)


def plot_figure(a_strict, a_relaxed, b_bag, b_tx, out_png: Path, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.0, 1.1]})

    # --- LEFT: Dataset A ---
    ax = axes[0]
    groups = [
        (f"Strict\n(n={a_strict['n_patients']})", a_strict["bag_auroc"], a_strict["order_auroc"],
         (a_strict["bag_lo"], a_strict["bag_hi"]), (a_strict["order_lo"], a_strict["order_hi"])),
        (f"Relaxed\n(n={a_relaxed['n_patients']})", a_relaxed["bag_auroc"], a_relaxed["order_auroc"],
         (a_relaxed["bag_lo"], a_relaxed["bag_hi"]), (a_relaxed["order_lo"], a_relaxed["order_hi"])),
    ]
    xpos = np.arange(len(groups))
    width = 0.35
    bag_color = "#4C78A8"
    order_color = "#F58518"

    bag_vals = [g[1] for g in groups]
    ord_vals = [g[2] for g in groups]
    bag_err = np.array([_errbar(g[3][0], g[3][1], g[1]) for g in groups]).T
    ord_err = np.array([_errbar(g[4][0], g[4][1], g[2]) for g in groups]).T

    ax.bar(xpos - width / 2, bag_vals, width, yerr=bag_err, capsize=4,
           color=bag_color, label="Bag LR", edgecolor="black", linewidth=0.5)
    ax.bar(xpos + width / 2, ord_vals, width, yerr=ord_err, capsize=4,
           color=order_color, label="Order LR", edgecolor="black", linewidth=0.5)

    for i, g in enumerate(groups):
        ax.text(xpos[i] - width / 2, g[1] + 0.015, f"{g[1]:.3f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(xpos[i] + width / 2, g[2] + 0.015, f"{g[2]:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Random")
    ax.set_xticks(xpos)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylim(0.45, 1.08)
    ax.set_ylabel("Test AUROC")
    ax.set_title("Dataset A — order-only cancer trajectory task", fontsize=11)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # --- RIGHT: Dataset B ---
    ax = axes[1]
    bars = [
        ("Bag LR",                  b_bag["bag_auroc"],         (b_bag["bag_lo"], b_bag["bag_hi"]),         "#4C78A8"),
        ("Bag + first/last LR",     b_bag["fl_auroc"],          (b_bag["fl_lo"], b_bag["fl_hi"]),           "#6BA4D1"),
        ("Transformer (ordered)",   b_tx["ordered_auroc"],      (b_tx["ordered_lo"], b_tx["ordered_hi"]),   "#F58518"),
        ("Transformer (shuffled)",  b_tx["shuffled_auroc"],     (b_tx["shuffled_lo"], b_tx["shuffled_hi"]), "#FFB77E"),
    ]
    xpos = np.arange(len(bars))
    vals = [b[1] for b in bars]
    errs = np.array([_errbar(b[2][0], b[2][1], b[1]) for b in bars]).T
    colors = [b[3] for b in bars]
    labels_x = [b[0] for b in bars]

    ax.bar(xpos, vals, 0.6, yerr=errs, capsize=4, color=colors,
           edgecolor="black", linewidth=0.5)
    for i, (_, v, _, _) in enumerate(bars):
        ax.text(xpos[i], v + 0.010, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels_x, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.45, 0.85)
    ax.set_ylabel("Test AUROC")
    ax.set_title("Dataset B — 1-year mortality (n=9,601)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Order vs. semantics in real MIMIC-IV cancer trajectories",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the figure (default: data/intermediate/qc).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    ensure_directories(paths)
    out_dir = args.out_dir or paths.qc_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("computing Dataset A strict...")
    a_strict = dataset_a_results(paths, "strict")
    logger.info("  strict: n=%d bag=%.3f order=%.3f", a_strict["n_patients"], a_strict["bag_auroc"], a_strict["order_auroc"])

    logger.info("computing Dataset A relaxed...")
    a_relaxed = dataset_a_results(paths, "relaxed")
    logger.info("  relaxed: n=%d bag=%.3f order=%.3f", a_relaxed["n_patients"], a_relaxed["bag_auroc"], a_relaxed["order_auroc"])

    logger.info("computing Dataset B bag baselines...")
    b_bag = dataset_b_bag_results(paths)
    logger.info("  bag=%.3f bag+first/last=%.3f", b_bag["bag_auroc"], b_bag["fl_auroc"])

    logger.info("loading Dataset B transformer results from sequence_model_b_report.json...")
    b_tx = dataset_b_transformer_results(paths)
    logger.info("  ordered=%.3f shuffled=%.3f", b_tx["ordered_auroc"], b_tx["shuffled_auroc"])

    out_png = out_dir / "summary_figure.png"
    out_pdf = out_dir / "summary_figure.pdf"
    plot_figure(a_strict, a_relaxed, b_bag, b_tx, out_png, out_pdf)
    logger.info("wrote %s and %s", out_png, out_pdf)

    # Also drop a json alongside so the figure numbers are auditable
    report = {
        "dataset_a_strict": a_strict,
        "dataset_a_relaxed": a_relaxed,
        "dataset_b_bag": b_bag,
        "dataset_b_transformer": b_tx,
    }
    (out_dir / "summary_figure_values.json").write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
