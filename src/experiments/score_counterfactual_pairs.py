"""
Score counterfactual pairs with trained models.

Each pair consists of (seq_pos, seq_neg) with identical letter counts but
opposite latent labels.  A model that uses only letter frequencies (BoE)
must score them identically; any separation = genuine order sensitivity.

Usage (DL baselines — trains then scores):
    python -m src.experiments.score_counterfactual_pairs \
        --model_type DL --models LSTM,Transformer,RNNTransformer \
        --number_to_use 9

Usage (LLM — trains then scores):
    python -m src.experiments.score_counterfactual_pairs \
        --model_type LLM --model_name google-bert/bert-base-uncased \
        --number_to_use 9

The script:
  1. Trains the model on the full training set (fraction=1.0)
  2. Loads counterfactual pairs from paper_tables/counterfactual_pairs_tricky_rnd.csv
  3. Scores seq_pos and seq_neg with the trained model
  4. Reports pairwise accuracy, mean margin, and paired AUC
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

PAIRS_PATH = Path("paper_tables/counterfactual_pairs_tricky_rnd.csv")
DATA_DIR = Path("data/simulation/tested")
OUTPUT_DIR = Path("paper_tables")
SEP = "\x1f"


# ──────────────────────────────────────────────────────────────────────
# K-GRAM BASELINE (no GPU needed)
# ──────────────────────────────────────────────────────────────────────
def kgram_score_pairs(pairs_df, k=2):
    """Score pairs with k-gram baseline. Returns pairwise accuracy."""
    X_train = pd.read_csv(DATA_DIR / "X_train_9.csv").fillna("")
    y_train = pd.read_csv(DATA_DIR / "y_train_9.csv").fillna("")

    # Learn P(Y=1 | k-gram) from training set
    stats = defaultdict(lambda: [0, 0])
    for seq, label in zip(X_train.Sequences, y_train.Outcome):
        letters = seq.split(SEP)
        for i in range(len(letters) - k + 1):
            ng = "-".join(letters[i:i + k])
            stats[ng][int(label)] += 1
    probs = {ng: c[1] / sum(c) for ng, c in stats.items() if sum(c) > 0}

    def _score(seq):
        letters = seq.split(SEP)
        ngs = ["-".join(letters[i:i + k]) for i in range(len(letters) - k + 1)]
        scores = [probs.get(ng, 0.5) for ng in ngs]
        return np.mean(scores) if scores else 0.5

    correct = 0
    tied = 0
    margins = []
    for _, row in pairs_df.iterrows():
        s_pos = _score(row.seq_pos)
        s_neg = _score(row.seq_neg)
        if s_pos > s_neg:
            correct += 1
        elif s_pos == s_neg:
            tied += 1
        margins.append(s_pos - s_neg)

    n = len(pairs_df)
    pair_acc = (correct + 0.5 * tied) / n
    return {
        "pairwise_accuracy": pair_acc,
        "mean_margin": np.mean(margins),
        "median_margin": np.median(margins),
        "pct_tied": 100 * tied / n,
    }


# ──────────────────────────────────────────────────────────────────────
# DL BASELINE SCORING
# ──────────────────────────────────────────────────────────────────────
def score_dl_model(model, sequences, device, batch_size=512):
    """Score preprocessed sequences with a DL model. Returns array of P(Y=1)."""
    # Reuse DLDataset with dummy labels
    sys.path.insert(0, str(Path(__file__).parent))
    from DL_TR_baselines_experiment import LetterSequenceDataset as DLDataset, collate_fn

    dummy_labels = [0] * len(sequences)
    dataset = DLDataset(sequences, dummy_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_seqs, _ in loader:
            batch_seqs = batch_seqs.to(device)
            logits = model(batch_seqs)
            # logits shape: (batch, num_classes=2)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def compute_pair_metrics(scores_pos, scores_neg):
    """Compute pairwise accuracy and margins."""
    correct = (scores_pos > scores_neg).sum()
    tied = (scores_pos == scores_neg).sum()
    n = len(scores_pos)
    margins = scores_pos - scores_neg

    return {
        "pairwise_accuracy": float(correct + 0.5 * tied) / n,
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "std_margin": float(np.std(margins)),
        "pct_positive_margin": float(100 * (margins > 0).sum() / n),
        "pct_tied": float(100 * tied / n),
    }


# ──────────────────────────────────────────────────────────────────────
# MAIN: DL models
# ──────────────────────────────────────────────────────────────────────
def run_dl(args):
    """Train DL baselines and score counterfactual pairs."""
    # Import the DL experiment module
    sys.path.insert(0, str(Path(__file__).parent))
    from DL_TR_baselines_experiment import (
        LetterSequenceDataset as DLDataset,
        create_model, train_model_with_early_stopping, set_seed,
        DEFAULT_CONFIGS, collate_fn, preprocess_sequence
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data
    data_dir = Path(args.path_csv)
    X_train = pd.read_csv(data_dir / f"X_train_{args.number_to_use}.csv").fillna("")
    y_train = pd.read_csv(data_dir / f"y_train_{args.number_to_use}.csv").fillna("")
    X_val = pd.read_csv(data_dir / f"X_val_{args.number_to_use}.csv").fillna("")
    y_val = pd.read_csv(data_dir / f"y_val_{args.number_to_use}.csv").fillna("")

    # Load pairs
    pairs_df = pd.read_csv(Path(args.pairs_path))
    logger.info(f"Loaded {len(pairs_df):,} counterfactual pairs")

    # Preprocess: strip separator
    train_seqs = [preprocess_sequence(s) for s in X_train.Sequences]
    val_seqs = [preprocess_sequence(s) for s in X_val.Sequences]

    models_to_run = [m.strip() for m in args.models.split(",")]
    results = []

    for model_name in models_to_run:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")

        # Build and train using default config
        config = DEFAULT_CONFIGS[model_name].copy()
        model = create_model(model_name, config)

        train_dataset = DLDataset(train_seqs, y_train.Outcome.values.astype(int))
        val_dataset = DLDataset(val_seqs, y_val.Outcome.values.astype(int))

        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,
                                collate_fn=collate_fn)

        model, best_f1, best_auc, _, _, best_epoch, _ = train_model_with_early_stopping(
            model, train_loader, val_loader,
            num_epochs=50, lr=1e-3, patience=5, device=str(device)
        )
        logger.info(f"  Best Val AUC: {best_auc:.4f}, F1: {best_f1:.4f} (epoch {best_epoch})")

        # Score pairs
        logger.info(f"Scoring {len(pairs_df):,} pairs...")
        pair_seqs_pos = [preprocess_sequence(s) for s in pairs_df.seq_pos]
        pair_seqs_neg = [preprocess_sequence(s) for s in pairs_df.seq_neg]
        scores_pos = score_dl_model(model, pair_seqs_pos, device)
        scores_neg = score_dl_model(model, pair_seqs_neg, device)

        metrics = compute_pair_metrics(scores_pos, scores_neg)
        metrics["model"] = model_name
        results.append(metrics)
        logger.info(f"  Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
        logger.info(f"  Mean margin:       {metrics['mean_margin']:.4f}")

        del model
        torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────
# MAIN: LLM models
# ──────────────────────────────────────────────────────────────────────
def _is_encoder_model(model_name):
    """Check if model is an encoder (BERT/RoBERTa) vs decoder (Llama/Qwen)."""
    name_lower = model_name.lower()
    return any(k in name_lower for k in ["bert", "roberta", "electra", "deberta"])


def run_llm_encoder(args):
    """Train BERT/RoBERTa via AutoModelForSequenceClassification and score pairs."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    set_seed = __import__("random").seed
    set_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    # Load data
    data_dir = Path(args.path_csv)
    X_train = pd.read_csv(data_dir / f"X_train_{args.number_to_use}.csv").fillna("")
    y_train = pd.read_csv(data_dir / f"y_train_{args.number_to_use}.csv").fillna("")
    X_val = pd.read_csv(data_dir / f"X_val_{args.number_to_use}.csv").fillna("")
    y_val = pd.read_csv(data_dir / f"y_val_{args.number_to_use}.csv").fillna("")

    SEP = "\x1f"
    def make_text(seq):
        return "Sequential events: " + " ".join(seq.split(SEP)) + "\nOutcome (0 or 1):"

    train_texts = [make_text(s) for s in X_train.Sequences]
    val_texts = [make_text(s) for s in X_val.Sequences]
    train_labels = y_train.Outcome.values.astype(int)
    val_labels = y_val.Outcome.values.astype(int)

    # Simple training loop
    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    batch_size = 64
    epochs = 10
    patience = 3
    best_auc = 0
    no_improve = 0
    best_state = None

    logger.info(f"Training {args.model_name} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_texts))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_texts = [train_texts[j] for j in batch_idx]
            batch_labels = torch.tensor(train_labels[batch_idx], dtype=torch.long).to(device)
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                               truncation=True, max_length=128).to(device)
            outputs = model(**inputs, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                batch = val_texts[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=128).to(device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        logger.info(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/n_batches:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info(f"Training done — Best Val AUC: {best_auc:.4f}")

    # Score pairs
    pairs_df = pd.read_csv(Path(args.pairs_path))
    logger.info(f"Scoring {len(pairs_df):,} counterfactual pairs...")

    def score_sequences(seqs, batch_size=128):
        texts = [make_text(s) for s in seqs]
        all_probs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True,
                                   truncation=True, max_length=128).to(device)
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
        return np.array(all_probs)

    scores_pos = score_sequences(list(pairs_df.seq_pos))
    scores_neg = score_sequences(list(pairs_df.seq_neg))

    metrics = compute_pair_metrics(scores_pos, scores_neg)
    metrics["model"] = args.model_name
    logger.info(f"  Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
    logger.info(f"  Mean margin:       {metrics['mean_margin']:.4f}")
    del model
    torch.cuda.empty_cache()
    return [metrics]


def run_llm(args):
    """Train decoder-only LLM from scratch and score counterfactual pairs."""
    sys.path.insert(0, str(Path(__file__).parent))
    from LLM_fraction_experiment import (
        CausalLMWithClassificationHead, set_seed, standard_narrative_prompt,
        find_optimal_threshold, load_tokenizer, load_model_causal,
        TemporalCausalDataset, train_and_evaluate_causal, parse_args
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build args compatible with LLM_fraction_experiment
    llm_extra = []
    if args.peft:
        llm_extra.append("--peft")
    if args.use_quantization:
        llm_extra.append("--use_quantization")
    llm_args = parse_args([
        "--model_name", args.model_name,
        "--number_to_use", args.number_to_use,
        "--seed", str(args.seed),
        "--fractions", "1.0",
        "--path_csv", args.path_csv,
    ] + llm_extra)

    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = load_tokenizer(llm_args.model_name, llm_args.model_type,
                               hf_token, llm_args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load data
    data_dir = Path(args.path_csv)
    X_train = pd.read_csv(data_dir / f"X_train_{args.number_to_use}.csv").fillna("")
    y_train = pd.read_csv(data_dir / f"y_train_{args.number_to_use}.csv").fillna("")
    X_val = pd.read_csv(data_dir / f"X_val_{args.number_to_use}.csv").fillna("")
    y_val = pd.read_csv(data_dir / f"y_val_{args.number_to_use}.csv").fillna("")

    # Prepare dataloaders
    train_texts = X_train.apply(standard_narrative_prompt, axis=1).tolist()
    val_texts = X_val.apply(standard_narrative_prompt, axis=1).tolist()
    train_labels = y_train["Outcome"].tolist()
    val_labels = y_val["Outcome"].tolist()

    train_dataset = TemporalCausalDataset(train_texts, train_labels, tokenizer,
                                          max_length=llm_args.max_length)
    val_dataset = TemporalCausalDataset(val_texts, val_labels, tokenizer,
                                        max_length=llm_args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=llm_args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=llm_args.batch_size,
                            shuffle=False, pin_memory=True, num_workers=0)

    # Load and train model
    logger.info(f"Training {args.model_name} from scratch on dataset {args.number_to_use}...")
    model = load_model_causal(llm_args, tokenizer, hf_token)
    if tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))
    if not args.use_quantization:
        model = model.to(device, dtype=torch.bfloat16)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    best_auc, best_f1, _, _, _, _ = train_and_evaluate_causal(
        model, train_loader, val_loader, llm_args, device,
        use_quantization=args.use_quantization
    )
    logger.info(f"Training done — Val AUC: {best_auc:.4f}, Val F1: {best_f1:.4f}")

    # Load pairs and score
    pairs_df = pd.read_csv(Path(args.pairs_path))
    logger.info(f"Scoring {len(pairs_df):,} counterfactual pairs...")

    def make_prompt(seq):
        row = pd.Series({"Sequences": seq})
        return standard_narrative_prompt(row)

    def score_sequences(seqs, batch_size=64):
        prompts = [make_prompt(s) for s in seqs]
        all_probs = []
        model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True,
                               max_length=llm_args.max_length).to(device)
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
                all_probs.extend(probs.cpu().float().numpy())
        return np.array(all_probs)

    scores_pos = score_sequences(list(pairs_df.seq_pos))
    scores_neg = score_sequences(list(pairs_df.seq_neg))

    metrics = compute_pair_metrics(scores_pos, scores_neg)
    metrics["model"] = args.model_name
    logger.info(f"  Pairwise accuracy: {metrics['pairwise_accuracy']:.4f}")
    logger.info(f"  Mean margin:       {metrics['mean_margin']:.4f}")

    del model
    torch.cuda.empty_cache()
    return [metrics]


# ──────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Score counterfactual pairs with trained models")
    parser.add_argument("--model_type", choices=["DL", "LLM", "kgram"],
                        default="kgram")
    parser.add_argument("--models", type=str, default="LSTM,Transformer,RNNTransformer",
                        help="Comma-separated DL model names")
    parser.add_argument("--model_name", type=str, default="google-bert/bert-base-uncased",
                        help="HuggingFace model name (for LLM mode)")
    parser.add_argument("--number_to_use", type=str, default="9")
    parser.add_argument("--seed", type=int, default=9950)
    parser.add_argument("--peft", action="store_true", help="Use LoRA")
    parser.add_argument("--use_quantization", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--path_csv", type=str, default=str(DATA_DIR) + "/",
                        help="Path to data CSVs")
    parser.add_argument("--pairs_path", type=str, default=str(PAIRS_PATH),
                        help="Path to counterfactual pairs CSV")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    # Override paths if provided
    pairs_path = Path(args.pairs_path)
    data_dir = Path(args.path_csv)

    # Always compute k-gram baseline first (no GPU needed)
    pairs_df = pd.read_csv(pairs_path)
    logger.info(f"Loaded {len(pairs_df):,} counterfactual pairs")

    print("\n" + "=" * 60)
    print("  K-GRAM BASELINES ON COUNTERFACTUAL PAIRS")
    print("=" * 60)
    all_results = []
    for k in [1, 2, 3]:
        res = kgram_score_pairs(pairs_df, k=k)
        res["model"] = f"{k}-gram"
        all_results.append(res)
        print(f"  {k}-gram: pair_acc={res['pairwise_accuracy']:.4f}  "
              f"margin={res['mean_margin']:+.4f}  "
              f"tied={res['pct_tied']:.1f}%")

    # Run model scoring
    if args.model_type == "DL":
        model_results = run_dl(args)
        all_results.extend(model_results)
    elif args.model_type == "LLM":
        if _is_encoder_model(args.model_name):
            model_results = run_llm_encoder(args)
        else:
            model_results = run_llm(args)
        all_results.extend(model_results)

    # Print summary table
    print("\n" + "=" * 60)
    print("  COUNTERFACTUAL PAIR RESULTS")
    print("=" * 60)
    print(f"  {'Model':<20s} | {'Pair Acc':>8s} | {'Margin':>8s} | {'% Tied':>6s}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    for r in all_results:
        print(f"  {r['model']:<20s} | {r['pairwise_accuracy']:>8.4f} | "
              f"{r['mean_margin']:>+8.4f} | {r.get('pct_tied', 0):>6.1f}")

    # Save results
    out_path = OUTPUT_DIR / "counterfactual_pair_results.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
