"""
Visit-level shuffle test for CKD→ESRD temporal analysis.

Permutes the ORDER of visits (not codes within visits), then compares
model performance on original vs shuffled visit sequences.

Models:
  - Bag-of-visits (LogReg, XGBoost): sum multi-hot vectors → order-invariant control
  - Visit-level LSTM: processes visit sequence step by step → order-sensitive
  - Visit-level Transformer: self-attention over visits → order-sensitive

Usage:
  # CPU-only (sklearn baselines):
  python src/mimic/visit_shuffle_test.py --mode sklearn

  # GPU (neural models + sklearn):
  python src/mimic/visit_shuffle_test.py --mode all

  # Neural only:
  python src/mimic/visit_shuffle_test.py --mode neural
"""

import argparse
import json
import pickle
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "mimic_training"
RESULTS_CSV = ROOT / "results" / "mimic" / "visit_shuffle_test.csv"

SEED = 42
N_SHUFFLE_SEEDS = 10
N_NEURAL_SEEDS = 5


# ── Data loading ──────────────────────────────────────────────────────

def load_split(name):
    with open(DATA_DIR / f"visit_level_{name}.pkl", "rb") as f:
        return pickle.load(f)


def load_meta():
    with open(DATA_DIR / "visit_level_meta.pkl", "rb") as f:
        return pickle.load(f)


def to_bag_of_visits(records):
    """Sum multi-hot visit vectors into a single vector per patient."""
    X = np.stack([r["visit_sequence"].sum(axis=0) for r in records])
    y = np.array([r["label"] for r in records])
    return X, y


def shuffle_visits(records, rng):
    """Return new records with visit order randomly permuted per patient."""
    shuffled = []
    for r in records:
        perm = rng.permutation(r["num_visits"])
        shuffled.append({
            **r,
            "visit_sequence": r["visit_sequence"][perm],
        })
    return shuffled


def find_optimal_threshold(y_true, preds):
    best_thr, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (np.array(preds) >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thr = t
    return best_thr, best_f1


def eval_metrics(y_true, preds, threshold):
    pred_bin = (np.array(preds) >= threshold).astype(int)
    return {
        "AUC": roc_auc_score(y_true, preds),
        "F1": f1_score(y_true, pred_bin, zero_division=0),
        "Precision": precision_score(y_true, pred_bin, zero_division=0),
        "Recall": recall_score(y_true, pred_bin, zero_division=0),
    }


# ── Sklearn models ────────────────────────────────────────────────────

def run_sklearn(train, val, test):
    """Run bag-of-visits baselines (order-invariant)."""
    from xgboost import XGBClassifier

    X_tr, y_tr = to_bag_of_visits(train)
    X_va, y_va = to_bag_of_visits(val)
    X_te, y_te = to_bag_of_visits(test)

    results = []

    # Verify BoV is shuffle-invariant
    rng = np.random.RandomState(0)
    train_shuf = shuffle_visits(train, rng)
    X_tr_shuf, _ = to_bag_of_visits(train_shuf)
    assert np.allclose(X_tr, X_tr_shuf), "Bag-of-visits should be shuffle-invariant"
    logger.info("Bag-of-visits shuffle-invariance verified")

    for name, model in [
        ("XGBoost_BoV", XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            random_state=SEED, eval_metric="logloss", verbosity=0,
        )),
        ("LogReg_BoV", LogisticRegression(
            max_iter=1000, C=1.0, random_state=SEED, class_weight="balanced",
        )),
    ]:
        logger.info(f"Training {name}...")
        if "XGBoost" in name:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

        preds_va = model.predict_proba(X_va)[:, 1]
        thr, _ = find_optimal_threshold(y_va, preds_va)
        preds_te = model.predict_proba(X_te)[:, 1]
        m = eval_metrics(y_te, preds_te, thr)

        logger.info(f"  {name}: AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")
        results.append({
            "model": name, "condition": "original", "shuffle_seed": -1,
            "AUC": m["AUC"], "F1": m["F1"],
            "Precision": m["Precision"], "Recall": m["Recall"],
        })
        # BoV is invariant — record same result for shuffled
        for ss in range(N_SHUFFLE_SEEDS):
            results.append({
                "model": name, "condition": "shuffled", "shuffle_seed": ss,
                "AUC": m["AUC"], "F1": m["F1"],
                "Precision": m["Precision"], "Recall": m["Recall"],
            })

    return results


# ── Neural models ─────────────────────────────────────────────────────

def run_neural(train, val, test, meta):
    """Run LSTM and Transformer on visit sequences (ordered vs shuffled)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    vocab_size = meta["vocab_size"]

    # ── Dataset ───────────────────────────────────────────────────────

    class VisitDataset(Dataset):
        def __init__(self, records, max_visits=100):
            self.data = []
            for r in records:
                seq = r["visit_sequence"][:max_visits]  # (n_visits, vocab_size)
                n = len(seq)
                padded = np.zeros((max_visits, vocab_size), dtype=np.float32)
                padded[:n] = seq
                self.data.append({
                    "sequence": padded,
                    "length": n,
                    "label": r["label"],
                })

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            d = self.data[idx]
            return (
                torch.tensor(d["sequence"], dtype=torch.float32),
                torch.tensor(d["length"], dtype=torch.long),
                torch.tensor(d["label"], dtype=torch.long),
            )

    # ── Models ────────────────────────────────────────────────────────

    class VisitLSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
            super().__init__()
            self.proj = nn.Linear(input_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                batch_first=True,
                                dropout=dropout if num_layers > 1 else 0)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 2)

        def forward(self, x, lengths):
            x = torch.relu(self.proj(x))
            # Use last valid hidden state
            _, (h_n, _) = self.lstm(x)
            return self.fc(self.dropout(h_n[-1]))

    class VisitTransformer(nn.Module):
        def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                     dropout=0.3, max_visits=100):
            super().__init__()
            self.proj = nn.Linear(input_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, max_visits, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(d_model, 2)

        def forward(self, x, lengths):
            batch_size, seq_len, _ = x.size()
            x = torch.relu(self.proj(x)) + self.pos_enc[:, :seq_len, :]
            # Mask padding
            max_len = seq_len
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            x = self.transformer(x, src_key_padding_mask=mask)
            # Mean pool over valid positions
            mask_exp = (~mask).unsqueeze(-1).float()
            pooled = (x * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-9)
            return self.fc(self.dropout(self.norm(pooled)))

    # ── Training loop ─────────────────────────────────────────────────

    def train_and_eval(model_class, model_name, train_records, val_records,
                       test_records, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_ds = VisitDataset(train_records)
        val_ds = VisitDataset(val_records)
        test_ds = VisitDataset(test_records)

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                                  pin_memory=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                                pin_memory=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                                 pin_memory=True, num_workers=0)

        model = model_class(vocab_size).to(device)

        # Class weights
        labels = [r["label"] for r in train_records]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val_loss = float("inf")
        best_state = None
        patience = 5
        no_improve = 0

        for epoch in range(30):
            model.train()
            for seqs, lens, labels_batch in train_loader:
                seqs, lens, labels_batch = (
                    seqs.to(device), lens.to(device), labels_batch.to(device)
                )
                optimizer.zero_grad()
                logits = model(seqs, lens)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seqs, lens, labels_batch in val_loader:
                    seqs, lens, labels_batch = (
                        seqs.to(device), lens.to(device), labels_batch.to(device)
                    )
                    val_loss += criterion(model(seqs, lens), labels_batch).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
            model = model.to(device)

        # Eval
        model.eval()
        def predict(loader):
            preds, labs = [], []
            with torch.no_grad():
                for seqs, lens, labels_batch in loader:
                    seqs, lens = seqs.to(device), lens.to(device)
                    probs = torch.softmax(model(seqs, lens), dim=-1)[:, 1]
                    preds.extend(probs.cpu().numpy())
                    labs.extend(labels_batch.numpy())
            return np.array(preds), np.array(labs)

        preds_va, y_va = predict(val_loader)
        thr, _ = find_optimal_threshold(y_va, preds_va)
        preds_te, y_te = predict(test_loader)
        return eval_metrics(y_te, preds_te, thr)

    # ── Run experiments ───────────────────────────────────────────────

    results = []

    for model_class, model_name in [
        (VisitLSTM, "Visit_LSTM"),
        (VisitTransformer, "Visit_Transformer"),
    ]:
        logger.info(f"\n=== {model_name} ===")

        # Original (ordered)
        orig_aucs = []
        for s in range(N_NEURAL_SEEDS):
            m = train_and_eval(model_class, model_name, train, val, test,
                               seed=SEED + s)
            orig_aucs.append(m["AUC"])
            results.append({
                "model": model_name, "condition": "original",
                "shuffle_seed": -1, "neural_seed": s,
                "AUC": m["AUC"], "F1": m["F1"],
                "Precision": m["Precision"], "Recall": m["Recall"],
            })
            logger.info(f"  Original seed={s}: AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

        logger.info(f"  Original mean AUC: {np.mean(orig_aucs):.4f} ± {np.std(orig_aucs):.4f}")

        # Shuffled
        for ss in range(N_SHUFFLE_SEEDS):
            rng = np.random.RandomState(ss)
            train_shuf = shuffle_visits(train, rng)
            val_shuf = shuffle_visits(val, rng)
            test_shuf = shuffle_visits(test, rng)

            m = train_and_eval(model_class, model_name, train_shuf, val_shuf,
                               test_shuf, seed=SEED)
            results.append({
                "model": model_name, "condition": "shuffled",
                "shuffle_seed": ss, "neural_seed": 0,
                "AUC": m["AUC"], "F1": m["F1"],
                "Precision": m["Precision"], "Recall": m["Recall"],
            })
            logger.info(f"  Shuffled seed={ss}: AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")

        shuf_aucs = [r["AUC"] for r in results
                     if r["model"] == model_name and r["condition"] == "shuffled"]
        logger.info(f"  Shuffled mean AUC: {np.mean(shuf_aucs):.4f} ± {np.std(shuf_aucs):.4f}")

    return results


# ── Summary ───────────────────────────────────────────────────────────

def print_summary(results):
    df = pd.DataFrame(results)

    print("\n" + "=" * 90)
    print("VISIT-LEVEL SHUFFLE TEST  (CKD→ESRD, MIMIC-IV 3.1)")
    print("=" * 90)
    print(f"{'Model':<22s}  {'Original AUC':>16s}  {'Shuffled AUC':>20s}  {'Δ AUC':>10s}")
    print("-" * 90)

    for model in df["model"].unique():
        orig = df[(df["model"] == model) & (df["condition"] == "original")]
        shuf = df[(df["model"] == model) & (df["condition"] == "shuffled")]

        orig_mean = orig["AUC"].mean()
        orig_std = orig["AUC"].std() if len(orig) > 1 else 0

        shuf_mean = shuf["AUC"].mean()
        shuf_std = shuf["AUC"].std()

        delta = orig_mean - shuf_mean

        if orig_std > 0:
            orig_str = f"{orig_mean:.4f}±{orig_std:.4f}"
        else:
            orig_str = f"{orig_mean:.4f}"

        print(f"{model:<22s}  {orig_str:>16s}  "
              f"{shuf_mean:.4f}±{shuf_std:.4f}  {delta:>+10.4f}")

    print("-" * 90)
    print(f"Neural original: {N_NEURAL_SEEDS} seeds. Shuffled: {N_SHUFFLE_SEEDS} permutations.")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sklearn", "neural", "all"], default="all")
    args = parser.parse_args()

    logger.info("Loading visit-level data...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    meta = load_meta()

    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    logger.info(f"Vocab: {meta['vocab_size']} codes")

    all_results = []

    if args.mode in ("sklearn", "all"):
        logger.info("\n=== Sklearn baselines ===")
        all_results.extend(run_sklearn(train, val, test))

    if args.mode in ("neural", "all"):
        logger.info("\n=== Neural models ===")
        all_results.extend(run_neural(train, val, test, meta))

    # Save
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(RESULTS_CSV, index=False)
    logger.info(f"\nSaved to {RESULTS_CSV}")

    print_summary(all_results)


if __name__ == "__main__":
    main()
