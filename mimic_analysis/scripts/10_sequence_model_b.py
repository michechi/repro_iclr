"""Step 10: train a small transformer on Dataset B sequences.

The bag baseline + first/last baseline in step 09 showed zero gap, suggesting
order does not help mortality prediction once the semantic counts are present.
This script gives the order-aware story a proper chance: a small transformer
encoder reads the full ordered note sequence (10 binary flags per note + a
log-scaled delta-days feature) and produces a patient-level mortality logit.

For a complementary control, we also train the same transformer on the
*shuffled* version of each patient's sequence — if the ordered model beats
the shuffled model, that gap is a lower bound on how much order helps.
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
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from mimic_cancer.dfci_imaging_model import LABEL_ORDER  # noqa: E402
from mimic_cancer.logging_utils import setup_logging  # noqa: E402
from mimic_cancer.paths import ensure_directories, load_paths  # noqa: E402

logger = setup_logging("10_sequence_model_b")

FLAG_COLS = [f"flag_{label}" for label in LABEL_ORDER]
N_LABELS = len(LABEL_ORDER)
N_INPUT = N_LABELS + 1  # flags + log1p(delta_days)


class SeqDataset(Dataset):
    def __init__(self, patients: list[tuple[int, torch.Tensor, float]]):
        self.patients = patients

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, i: int):
        return self.patients[i]


def collate_fn(batch):
    lengths = [p[1].shape[0] for p in batch]
    max_len = max(lengths)
    B = len(batch)
    x = torch.zeros(B, max_len, N_INPUT, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    y = torch.zeros(B, dtype=torch.float32)
    for i, (_subj, feats, label) in enumerate(batch):
        L = feats.shape[0]
        x[i, :L] = feats
        mask[i, :L] = True
        y[i] = label
    return x, mask, y


class SeqModel(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.input_proj = nn.Linear(N_INPUT, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        L = min(L, self.max_len)
        x = x[:, :L]
        mask = mask[:, :L]
        h = self.input_proj(x)
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        padding_mask = ~mask  # True where padding
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.norm(h)
        mask_f = mask.float().unsqueeze(-1)
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


def build_patient_data(
    seq_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split_col: str,
    split_name: str,
    max_len: int,
    shuffle_sequences: bool = False,
    seed: int = 2026,
) -> list[tuple[int, torch.Tensor, float]]:
    labels_map = dict(zip(labels_df["subject_id"], labels_df["y_death_365"]))
    split_map = dict(zip(labels_df["subject_id"], labels_df[split_col]))
    rng = np.random.default_rng(seed)

    data: list[tuple[int, torch.Tensor, float]] = []
    for subj, g in seq_df.groupby("subject_id"):
        if subj not in labels_map or split_map.get(subj) != split_name:
            continue
        g = g.sort_values("seq_idx")
        if len(g) > max_len:
            g = g.tail(max_len)

        flags = g[FLAG_COLS].to_numpy(dtype=np.float32)
        deltas = g["delta_days_from_prev_note"].to_numpy(dtype=np.float32)
        deltas_log = np.log1p(np.clip(deltas, 0.0, None)).reshape(-1, 1)
        feats = np.concatenate([flags, deltas_log], axis=1)

        if shuffle_sequences and len(feats) > 1:
            perm = rng.permutation(len(feats))
            feats = feats[perm]

        data.append((int(subj), torch.from_numpy(feats), float(labels_map[subj])))
    return data


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    losses: list[float] = []
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        logit = model(x, mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logit: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for x, mask, y in loader:
        x, mask = x.to(device), mask.to(device)
        logit = model(x, mask)
        all_logit.append(logit.cpu().numpy())
        all_y.append(y.numpy())
    logits = np.concatenate(all_logit)
    proba = 1.0 / (1.0 + np.exp(-logits))
    y = np.concatenate(all_y)
    return proba, y


def bootstrap_ci(y: np.ndarray, proba: np.ndarray, n_boot: int = 1000, seed: int = 2026):
    rng = np.random.default_rng(seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb, pb = y[idx], proba[idx]
        if len(np.unique(yb)) < 2:
            continue
        aurocs.append(roc_auc_score(yb, pb))
        auprcs.append(average_precision_score(yb, pb))
    if not aurocs:
        return (float("nan"),) * 4
    return (
        float(np.percentile(aurocs, 2.5)),
        float(np.percentile(aurocs, 97.5)),
        float(np.percentile(auprcs, 2.5)),
        float(np.percentile(auprcs, 97.5)),
    )


def train_and_eval(
    seq_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    args: argparse.Namespace,
    shuffle_sequences: bool,
    tag: str,
) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = build_patient_data(
        seq_df, labels_df, "split_mortality_task", "train", args.max_len,
        shuffle_sequences=shuffle_sequences, seed=args.seed,
    )
    val_data = build_patient_data(
        seq_df, labels_df, "split_mortality_task", "val", args.max_len,
        shuffle_sequences=shuffle_sequences, seed=args.seed,
    )
    test_data = build_patient_data(
        seq_df, labels_df, "split_mortality_task", "test", args.max_len,
        shuffle_sequences=shuffle_sequences, seed=args.seed,
    )
    logger.info("%s train=%d val=%d test=%d", tag, len(train_data), len(val_data), len(test_data))

    train_loader = DataLoader(
        SeqDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        SeqDataset(val_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        SeqDataset(test_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    device = args.device
    model = SeqModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_len,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_auroc = -1.0
    best_state: dict | None = None
    patience_counter = 0
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        val_proba, val_y = evaluate(model, val_loader, device)
        val_auroc = roc_auc_score(val_y, val_proba)
        logger.info("%s epoch %3d  loss=%.4f  val_auroc=%.4f", tag, epoch, loss, val_auroc)
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("%s early stop at epoch %d", tag, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_proba, test_y = evaluate(model, test_loader, device)
    test_auroc = float(roc_auc_score(test_y, test_proba))
    test_auprc = float(average_precision_score(test_y, test_proba))
    pred = (test_proba >= 0.5).astype(int)
    test_f1 = float(f1_score(test_y, pred))
    test_acc = float((pred == test_y).mean())
    auroc_lo, auroc_hi, auprc_lo, auprc_hi = bootstrap_ci(test_y, test_proba, seed=args.seed)

    result = {
        "name": f"dataset_b_transformer_{tag}",
        "n_train": len(train_data),
        "n_val": len(val_data),
        "n_test": len(test_data),
        "best_val_auroc": best_val_auroc,
        "test_auroc": test_auroc,
        "test_auroc_ci_lo": auroc_lo,
        "test_auroc_ci_hi": auroc_hi,
        "test_auprc": test_auprc,
        "test_auprc_ci_lo": auprc_lo,
        "test_auprc_ci_hi": auprc_hi,
        "test_f1": test_f1,
        "test_acc": test_acc,
    }
    logger.info(
        "%s TEST: auroc=%.4f [%.4f,%.4f]  auprc=%.4f [%.4f,%.4f]  f1=%.4f  acc=%.4f",
        tag, test_auroc, auroc_lo, auroc_hi, test_auprc, auprc_lo, auprc_hi, test_f1, test_acc,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-config", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-shuffled", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths(args.paths_config)
    ensure_directories(paths)

    logger.info("device=%s", args.device)
    labels = pd.read_parquet(paths.dataset_mortality365_dir / "labels.parquet")
    seq = pd.read_parquet(paths.dataset_mortality365_dir / "X_seq.parquet")

    results = []
    results.append(train_and_eval(seq, labels, args, shuffle_sequences=False, tag="ordered"))
    if not args.skip_shuffled:
        results.append(train_and_eval(seq, labels, args, shuffle_sequences=True, tag="shuffled"))

    out_path = paths.qc_dir / "sequence_model_b_report.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("wrote %s", out_path)

    print()
    print("-- summary --")
    for r in results:
        print(
            f"{r['name']:40s}  auroc={r['test_auroc']:.4f} "
            f"[{r['test_auroc_ci_lo']:.4f},{r['test_auroc_ci_hi']:.4f}]  "
            f"auprc={r['test_auprc']:.4f}  f1={r['test_f1']:.4f}  acc={r['test_acc']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
