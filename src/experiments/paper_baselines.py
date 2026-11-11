"""Randomly initialized sequence models, tree baselines, k-gram classifiers, and feature ladders (Appendix B and D)."""
from __future__ import annotations
from collections import Counter
import json, math, random, time
from pathlib import Path

SCRATCH_CONFIGS = {
    "transformer": {"D": dict(embedding_dim=64, hidden_dim=256, layers=2, dropout=.3, lr=5e-4),
                    "O": dict(embedding_dim=64, hidden_dim=128, layers=3, dropout=.3, lr=5e-4)},
    "lstm": {"D": dict(embedding_dim=32, hidden_dim=64, layers=2, dropout=.3, lr=1e-3),
             "O": dict(embedding_dim=32, hidden_dim=128, layers=2, dropout=.2, lr=2e-3)},
    "rnntransformer": {"D": dict(embedding_dim=64, hidden_dim=64, ffn=256, layers=2, dropout=.3, lr=5e-4),
                       "O": dict(embedding_dim=128, hidden_dim=32, ffn=256, layers=3, dropout=.3, lr=1e-4)},
    "llama1m": {"D": dict(hidden_dim=128, layers=4, ffn=512, heads=4, kv_heads=2, lr=5e-4)}
}
FEATURES = ("count26", "count_keys", "residue", "lag_pairs", "key_lag_pairs", "position_pairs", "key_trigrams", "all_lag_pairs", "key_order_pairs")


def configuration(model, variant="D", *, learning_rate=None, sensitivity=False):
    if model in SCRATCH_CONFIGS:
        config = dict(SCRATCH_CONFIGS[model][variant])
        if model == "llama1m" and learning_rate is not None:
            config["lr"] = learning_rate
        elif learning_rate is not None and learning_rate != config["lr"]:
            raise ValueError("learning rate does not match Table 2 configuration")
        if sensitivity:
            if variant != "D" or model not in ("lstm", "transformer"):
                raise ValueError("D.1 sensitivity uses default LSTM or Transformer")
            if model == "lstm":
                config["layers"] = 1
        return {"model": model, "variant": variant, **config, "optimizer": "AdamW", "batch_size": 64, "epochs": 30,
                "stopping": "validation_f1", "patience": 5, "loss": "cross_entropy", "schedule": None,
                "padding_index": 0, "letter_indices": "A=1 through Z=26", "num_classes": 2,
                "weight_decay": .01, "threshold_grid": [i / 100 for i in range(10, 90)]}
    return {"model": model, "threshold_grid": [i / 100 for i in range(10, 90)], "stopping": "validation_logloss" if model.startswith("xgboost") else None}


def xgboost_parameters(labels, seed):
    positive = sum(int(y) == 1 for y in labels)
    negative = len(labels) - positive
    if not positive or not negative:
        raise ValueError("both classes required")
    return dict(n_estimators=200, max_depth=6, learning_rate=.05, subsample=.8, colsample_bytree=.8,
                reg_alpha=.5, reg_lambda=1.5, min_child_weight=2, gamma=.1, tree_method="hist",
                scale_pos_weight=negative / positive, early_stopping_rounds=10, eval_metric="logloss", random_state=seed)


def encode_sequences(sequences):
    from src.experiments.paper_data import tokens
    import numpy as np
    rows = [tokens(s) for s in sequences]
    if not rows or len(rows[0]) < 2 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("nonempty sequences of one fixed length n>=2 required")
    if any(not isinstance(c, str) or len(c) != 1 or not "A" <= c <= "Z"
           for row in rows for c in row):
        raise ValueError("indexed baselines require uppercase A-Z events")
    return np.asarray([[ord(c) - 64 for c in row] for row in rows], dtype=np.int64)


def select_threshold(y, probabilities):
    from sklearn.metrics import f1_score
    grid = [i / 100 for i in range(10, 90)]
    return max(grid, key=lambda t: f1_score(y, probabilities >= t, zero_division=0))


def metrics(y, p, threshold):
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    return dict(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, p >= threshold, zero_division=0)),
                precision=float(precision_score(y, p >= threshold, zero_division=0)),
                recall=float(recall_score(y, p >= threshold, zero_division=0)),
                accuracy=float(accuracy_score(y, p >= threshold)), threshold=float(threshold))


def build_scratch_model(config, n=20):
    import torch
    from torch import nn
    if config["model"] == "llama1m":
        from transformers import LlamaConfig, LlamaForSequenceClassification
        return LlamaForSequenceClassification(LlamaConfig(vocab_size=27, hidden_size=128, intermediate_size=512,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, pad_token_id=0, num_labels=2,
            max_position_embeddings=n, use_cache=False))

    class Scratch(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = config["model"]
            width = config["embedding_dim"]
            self.embedding = nn.Embedding(27, width, padding_idx=0)
            self.dropout = nn.Dropout(config["dropout"])
            if self.kind in ("lstm", "rnntransformer"):
                self.rnn = nn.LSTM(width, config["hidden_dim"], num_layers=config["layers"] if self.kind == "lstm" else 1,
                    dropout=config["dropout"] if self.kind == "lstm" and config["layers"] > 1 else 0,
                    batch_first=True, bidirectional=self.kind == "rnntransformer")
            if self.kind != "lstm":
                self.position = nn.Parameter(torch.randn(1, n, width))
                if self.kind == "rnntransformer":
                    self.projection = nn.Linear(2 * config["hidden_dim"], width)
                layer = nn.TransformerEncoderLayer(width, 4, dim_feedforward=config.get("ffn", config["hidden_dim"]),
                    dropout=config["dropout"], activation="gelu", batch_first=True)
                self.encoder = nn.TransformerEncoder(layer, config["layers"], enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(config["hidden_dim"] if self.kind == "lstm" else width, 2)

        def forward(self, x):
            mask = x != 0
            lengths = mask.sum(1)
            if (lengths == 0).any():
                raise ValueError("empty sequence")
            if not torch.equal(mask, torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]):
                raise ValueError("right padding required")
            h = self.embedding(x)
            if self.kind in ("lstm", "rnntransformer"):
                packed = nn.utils.rnn.pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
                recurrent, (last, _) = self.rnn(packed)
                if self.kind == "lstm":
                    return self.head(self.dropout(last[-1]))
                recurrent, _ = nn.utils.rnn.pad_packed_sequence(recurrent, batch_first=True, total_length=x.shape[1])
                h = self.projection(recurrent)
            h = self.encoder(h + self.position[:, :x.shape[1]], src_key_padding_mask=~mask)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(1)
            return self.head(self.dropout(self.norm(pooled)))

    return Scratch()


LAG_FREE_FEATURES = frozenset({"count26", "count_keys", "all_lag_pairs", "key_order_pairs"})


def feature_matrix(sequences, family, *, keys, lag, n):
    """Sparse CSR counts for the Appendix D feature families."""
    from src.experiments.paper_data import tokens
    import numpy as np
    from scipy.sparse import csr_matrix
    if not isinstance(n, int) or n < 2:
        raise ValueError("features require sequence length n>=2")
    lag_free = family in LAG_FREE_FEATURES
    if lag_free:
        if lag is not None and (not isinstance(lag, int) or isinstance(lag, bool) or not 1 <= lag < n):
            raise ValueError("optional lag must satisfy 1<=lag<n when provided")
    elif not isinstance(lag, int) or isinstance(lag, bool) or not 1 <= lag < n:
        raise ValueError(f"feature family {family!r} requires 1<=lag<n")
    if (not keys or len(set(keys)) != len(keys)
            or any(not isinstance(c, str) or len(c) != 1 for c in keys)):
        raise ValueError("feature keys must be distinct unit symbols")
    key_index = {c: i for i, c in enumerate(keys)}
    m = len(keys)
    dimensions = {"count26": 26, "count_keys": m,
                  "all_lag_pairs": (n - 1) * 676, "key_order_pairs": m * (m - 1) // 2}
    if not lag_free:
        dimensions.update({"residue": 26 * lag, "lag_pairs": 676, "key_lag_pairs": m * m,
                           "position_pairs": (n - lag) * 676, "key_trigrams": m ** 3})
    if family not in dimensions:
        raise ValueError("unknown feature family")
    indices = []
    values = []
    indptr = [0]
    for sequence in sequences:
        row = tokens(sequence)
        counts = Counter()
        if len(row) != n:
            raise ValueError("feature sequence length mismatch")
        if any(len(c) != 1 or not "A" <= c <= "Z" for c in row):
            raise ValueError("feature sequences require uppercase A-Z events")
        if family == "key_order_pairs":
            key_positions = {c: i for i, c in enumerate(row) if c in key_index}
            feature_idx = 0
            for p in range(m):
                for q in range(p + 1, m):
                    kp, kq = keys[p], keys[q]
                    if kp in key_positions and kq in key_positions and key_positions[kp] < key_positions[kq]:
                        counts[feature_idx] = 1
                    feature_idx += 1
        for i, a in enumerate(row):
            if family == "key_order_pairs":
                break
            ai = ord(a) - 65
            if family == "count26":
                counts[ai] += 1
            elif family == "count_keys" and a in key_index:
                counts[key_index[a]] += 1
            elif family == "residue":
                counts[(i % lag) * 26 + ai] += 1
            elif family == "key_trigrams" and i + 2 * lag < n:
                b, c = row[i + lag], row[i + 2 * lag]
                if all(x in key_index for x in (a, b, c)):
                    counts[(key_index[a] * m + key_index[b]) * m + key_index[c]] += 1
            elif family in ("lag_pairs", "key_lag_pairs", "position_pairs", "all_lag_pairs"):
                for offset in (range(1, n - i) if family == "all_lag_pairs" else (lag,)):
                    if i + offset >= n:
                        continue
                    b = row[i + offset]
                    bi = ord(b) - 65
                    if family == "key_lag_pairs":
                        if a in key_index and b in key_index:
                            counts[key_index[a] * m + key_index[b]] += 1
                    else:
                        counts[((offset - 1) * 676 if family == "all_lag_pairs" else i * 676 if family == "position_pairs" else 0) + ai * 26 + bi] += 1
        for j, value in sorted(counts.items()):
            indices.append(j)
            values.append(value)
        indptr.append(len(indices))
    return csr_matrix((np.asarray(values, dtype=np.float32), indices, indptr), shape=(len(sequences), dimensions[family]))


class KGram:
    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def fit(self, sequences, labels):
        from src.experiments.paper_data import tokens
        if not len(labels) or len(sequences) != len(labels) or any(y not in (0, 1) for y in labels):
            raise ValueError("nonempty aligned binary labels required")
        self.counts = Counter()
        self.positives = Counter()
        self.prior = sum(labels) / len(labels)
        for sequence, y in zip(sequences, labels):
            row = tokens(sequence)
            if len(row) < self.k:
                raise ValueError("k exceeds sequence length")
            for i in range(len(row) - self.k + 1):
                gram = tuple(row[i:i + self.k])
                self.counts[gram] += 1
                self.positives[gram] += int(y)
        return self

    def predict(self, sequences):
        from src.experiments.paper_data import tokens
        import numpy as np
        scores = []
        for sequence in sequences:
            row = tokens(sequence)
            grams = [tuple(row[i:i + self.k]) for i in range(len(row) - self.k + 1)]
            if not grams:
                raise ValueError("k exceeds sequence length")
            scores.append(sum(self.positives[g] / self.counts[g] if self.counts[g] else self.prior for g in grams) / len(grams))
        return np.asarray(scores)


def frozen_llama_features(data, model_path, device="cuda", batch_size=16):
    import torch
    import numpy as np
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from src.experiments.paper_protocol import sequence_prompt
    from src.experiments.paper_data import tokens
    source = Path(model_path)
    cfg = AutoConfig.from_pretrained(source, local_files_only=True)
    if cfg.model_type != "llama" or cfg.hidden_size != 2048 or cfg.num_hidden_layers != 16:
        raise ValueError("XGBoost-LLM requires Llama-3.2-1B, hidden 2048 / layers 16")
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, info = AutoModel.from_pretrained(source, local_files_only=True, use_safetensors=True,
                                            dtype=torch.float32, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("incomplete frozen backbone checkpoint")
    model.to(device).eval().requires_grad_(False)
    output = {}
    for split, (sequences, labels) in data.items():
        values = np.empty((len(sequences), 2048), dtype=np.float32)
        for start in range(0, len(sequences), batch_size):
            prompts = [sequence_prompt(s, decoder=True, sequence_length=len(tokens(s)))
                       for s in sequences[start:start + batch_size]]
            batch = tokenizer(prompts, padding=True, truncation=True, max_length=200, return_tensors="pt").to(device)
            with torch.inference_mode():
                h = model(**batch, use_cache=False).last_hidden_state
                mask = batch.attention_mask.unsqueeze(-1)
                values[start:start + len(prompts)] = ((h * mask).sum(1) / mask.sum(1)).cpu().numpy()
        output[split] = values
    return output


def run_baseline(args):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from src.experiments.paper_data import latent_label, read_paper_split, sha256, shuffle_sequences
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("output directory must be empty")
    config = configuration(args.model, args.variant, learning_rate=args.learning_rate, sensitivity=args.sensitivity)
    shuffle_control = getattr(args, "shuffle_control", "none") or "none"
    if shuffle_control not in {"none", "train_shuffled", "test_shuffled"}:
        raise ValueError("shuffle_control must be none, train_shuffled, or test_shuffled")
    if shuffle_control != "none" and args.model not in SCRATCH_CONFIGS:
        raise ValueError("Section 5 shuffle controls are defined for scratch sequence models")
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = {}
    evidence = {}
    selected = {}
    for split, expected in [("train", 400000), ("val", 50000), ("test", 50000)]:
        x, y, ev = read_paper_split(args.data_dir, args.tag, split)
        if len(y) != expected:
            raise ValueError(f"{split} requires {expected} source examples")
        ix = np.arange(len(y))
        if split == "train" and args.train_size < len(y):
            # Remark A.4: stratify on latent Y* (before noise), not the observed label.
            latent = np.asarray([latent_label(x[i], ev["rule"]) for i in ix], dtype=np.int64)
            ix, _ = train_test_split(ix, train_size=args.train_size, random_state=args.seed, stratify=latent)
            ix.sort()
        sequences = [x[i] for i in ix]
        labels = y[ix]
        if shuffle_control == "train_shuffled" and split in {"train", "val"}:
            sequences = shuffle_sequences(sequences, args.seed + (0 if split == "train" else 1))
        elif shuffle_control == "test_shuffled" and split == "test":
            sequences = shuffle_sequences(sequences, args.seed + 2)
        data[split] = (sequences, labels)
        evidence[split] = {**ev, "shuffle_control": shuffle_control if (
            (shuffle_control == "train_shuffled" and split in {"train", "val"})
            or (shuffle_control == "test_shuffled" and split == "test")) else "none"}
        selected[split] = ix
    out.mkdir(parents=True, exist_ok=True)
    for split, ix in selected.items():
        np.save(out / f"{split}_indices.npy", ix, allow_pickle=False)
    manifest = {"protocol": "paper_core_baselines_v1", "protocol_id": "paper_core_baselines_v1", "model": args.model, "config": config,
                "tag": args.tag, "seed": args.seed, "train_size": args.train_size, "data": evidence,
                "feature_family": args.feature, "k": args.k, "model_path": str(args.model_path) if args.model_path else None,
                "shuffle_control": shuffle_control, "label_binding": "preserve_original",
                "threshold_selected_on": "validation",
                "selected_indices_sha256": {s: sha256(out / f"{s}_indices.npy") for s in selected},
                "source_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("paper_data.py"), Path(__file__).with_name("paper_protocol.py"))}}
    if args.model == "xgboost-llm":
        source = Path(args.model_path)
        files = sorted({*source.glob("*.json"), *source.glob("*.safetensors"), *source.glob("*.model"), *source.glob("merges.txt"), *source.glob("vocab.txt")})
        if not any(p.suffix == ".safetensors" for p in files):
            raise FileNotFoundError("frozen Llama safetensors missing")
        manifest["frozen_snapshot_files_sha256"] = {p.name: sha256(p) for p in files}
    if args.model.startswith("xgboost"):
        manifest["xgboost_parameters"] = xgboost_parameters(data["train"][1], args.seed)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    if args.model in SCRATCH_CONFIGS:
        vp, tp, details = _run_scratch(data, args, config, out)
    elif args.model == "kgram":
        model = KGram(args.k).fit(*data["train"])
        vp = model.predict(data["val"][0])
        tp = model.predict(data["test"][0])
        details = {"unseen_gram_score": "training_class_prevalence"}
        (out / "kgram_model.json").write_text(json.dumps({"k": model.k, "prior": float(model.prior),
            "grams": [{"tokens": list(g), "count": count, "positives": int(model.positives[g])} for g, count in sorted(model.counts.items())]}) + "\n")
    else:
        if args.model == "xgboost-basic":
            features = {s: encode_sequences(x) for s, (x, y) in data.items()}
        elif args.model == "xgboost-llm":
            features = frozen_llama_features(data, args.model_path, args.device)
        else:
            rule = evidence["train"]["rule"]
            if rule["lag"] is None and args.feature not in LAG_FREE_FEATURES:
                raise ValueError(f"feature family {args.feature!r} requires an OC lag in the dataset rule")
            lag = rule["lag"]
            features = {s: feature_matrix(x, args.feature, keys=rule["keys"], lag=lag, n=rule["n"])
                        for s, (x, y) in data.items()}
        if args.model.startswith("xgboost"):
            from xgboost import XGBClassifier
            model = XGBClassifier(**xgboost_parameters(data["train"][1], args.seed), n_jobs=args.threads)
            model.fit(features["train"], data["train"][1], eval_set=[(features["val"], data["val"][1])], verbose=False)
            model.save_model(out / "selected_model.ubj")
            details = {"best_iteration": model.best_iteration, "selection": "validation_logloss"}
        elif args.model == "mlp-feature":
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-4,
                                  batch_size=64, learning_rate_init=1e-3, max_iter=30,
                                  early_stopping=True, n_iter_no_change=5, random_state=args.seed)
            model.fit(features["train"], data["train"][1])
            details = {"estimator_parameters": model.get_params(), "feature_width": features["train"].shape[1]}
        else:
            from sklearn.linear_model import LogisticRegression
            import inspect
            penalty = "l1" if args.model == "logreg-l1" else "l2"
            kw = {"solver": "liblinear" if penalty == "l1" else "lbfgs", "C": 1., "max_iter": 3000, "random_state": args.seed}
            params = inspect.signature(LogisticRegression).parameters
            if "penalty" not in params or params["penalty"].default == "deprecated":
                kw["l1_ratio"] = 1. if penalty == "l1" else 0.
            else:
                kw["penalty"] = penalty
            model = LogisticRegression(**kw).fit(features["train"], data["train"][1])
            details = {"estimator_parameters": model.get_params(), "feature_width": features["train"].shape[1]}
            np.savez(out / "linear_parameters.npz", coef=model.coef_, intercept=model.intercept_)
            if args.feature == "all_lag_pairs":
                details["per_lag_weight_l1_norm"] = np.abs(model.coef_[0].reshape(rule["n"] - 1, 676)).sum(1).tolist()
        vp = model.predict_proba(features["val"])[:, 1]
        tp = model.predict_proba(features["test"])[:, 1]
    threshold = select_threshold(data["val"][1], vp)
    for split, p in [("val", vp), ("test", tp)]:
        np.savez(out / f"{split}_predictions.npz", labels=data[split][1], probabilities=p)
    result = {"protocol": "paper_core_baselines_v1", "protocol_id": "paper_core_baselines_v1", "status": "completed",
              "model": args.model, "seed": args.seed, "tag": args.tag, "train_size": args.train_size,
              "rows": {s: len(y) for s, (x, y) in data.items()}, "is_smoke": False,
              "shuffle_control": shuffle_control, "label_binding": "preserve_original",
              "label_convention": evidence["train"]["label_convention"], "threshold_selected_on": "validation",
              "validation": metrics(data["val"][1], vp, threshold), "test": metrics(data["test"][1], tp, threshold),
              "details": details, "seconds": time.monotonic() - started, "manifest_sha256": sha256(out / "manifest.json"),
              "predictions_sha256": {s: sha256(out / f"{s}_predictions.npz") for s in ("val", "test")},
              "artifacts_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()}}
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _run_scratch(data, args, config, out):
    import torch
    import numpy as np
    from torch.nn import functional as F
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    arrays = {s: (torch.from_numpy(encode_sequences(x)), torch.from_numpy(y)) for s, (x, y) in data.items()}
    sequence_length = arrays["train"][0].shape[1]
    if any(x.shape[1] != sequence_length for x, y in arrays.values()):
        raise ValueError("all splits must use the same sequence length")
    model = build_scratch_model(config, n=sequence_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=.01)

    def logits(x):
        return model(input_ids=x, attention_mask=(x != 0)).logits if config["model"] == "llama1m" else model(x)

    def evaluate(split):
        model.eval()
        x, y = arrays[split]
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(y), 256):
                chunks.append(logits(x[start:start + 256].to(device)).softmax(-1)[:, 1].cpu().numpy())
        return np.concatenate(chunks)

    best = -1.
    bad = 0
    state = None
    history = []
    best_epoch = None
    generator = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*arrays["train"]), batch_size=64, shuffle=True, generator=generator)
    for epoch in range(1, 31):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(logits(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            optimizer.step()
        vp = evaluate("val")
        threshold = select_threshold(data["val"][1], vp)
        val_f1 = metrics(data["val"][1], vp, threshold)["f1"]
        if val_f1 > best:
            best, bad, best_epoch = val_f1, 0, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        history.append({"epoch": epoch, "validation_f1": val_f1, "validation_threshold": threshold})
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if bad >= 5:
            break
    model.load_state_dict(state)
    torch.save(state, out / "selected_model.pt")
    return evaluate("val"), evaluate("test"), {"best_epoch": best_epoch, "epochs_completed": len(history), "selection": "validation_f1", "patience": 5}
