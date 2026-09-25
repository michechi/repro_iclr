"""Local DFCI-imaging-student model wrapper.

The LabeledModel class is a verbatim copy of the one in the PhysioNet release at
`data/dfci_models/.../src/imaging/imaging_inference.py`, so the shipped `.pt`
state dict loads with `strict=True`. Do not rename the `peritoneal_head`
attribute — the state dict uses exactly that key.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from transformers import AutoModel, AutoTokenizer


# Order of heads in LabeledModel.forward(). BIND BY NAME, never by position.
# Note: the spec (mimic_cancer.md section 4) lists labels in a different
# order. This one is the ground truth from the reference script.
LABEL_ORDER: list[str] = [
    "any_cancer",
    "response",
    "progression",
    "met_brain",
    "met_bone",
    "met_adrenal",
    "met_liver",
    "met_lung",
    "met_lymph",
    "met_peritoneum",
]

N_LABELS = len(LABEL_ORDER)


class LabeledModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")

        self.any_cancer_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.response_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.progression_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.brain_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.bone_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.adrenal_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.liver_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.lung_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.node_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))
        self.peritoneal_head = Sequential(Linear(768, 128), ReLU(), Linear(128, 1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        main = self.bert(input_ids, attention_mask)
        cls = main.last_hidden_state[:, 0, :].squeeze(1)
        return (
            self.any_cancer_head(cls),
            self.response_head(cls),
            self.progression_head(cls),
            self.brain_head(cls),
            self.bone_head(cls),
            self.adrenal_head(cls),
            self.liver_head(cls),
            self.lung_head(cls),
            self.node_head(cls),
            self.peritoneal_head(cls),
        )


def load_dfci_imaging(weights_path: Path, device: str = "cpu") -> LabeledModel:
    model = LabeledModel()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")


def _normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return text.lower().replace("\n", " ")


def _tokenize_batch(texts: list[str], tokenizer, max_length: int = 512):
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def _iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def run_inference(
    notes_df: pd.DataFrame,
    model: LabeledModel,
    tokenizer,
    device: str = "cpu",
    batch_size: int = 16,
    max_length: int = 512,
    binarization_threshold: float = 0.5,
) -> pd.DataFrame:
    """Run DFCI imaging inference on a dataframe that has `note_id` and `text`.

    Returns a dataframe keyed on `note_id` with columns:
        logit_*   (float32, one per LABEL_ORDER)
        prob_*    (float32, sigmoid)
        flag_*    (uint8, 1 if prob >= threshold)
    """
    if notes_df.empty:
        columns = ["note_id"]
        for label in LABEL_ORDER:
            columns += [f"logit_{label}", f"prob_{label}", f"flag_{label}"]
        return pd.DataFrame(columns=columns)

    note_ids: list[str] = []
    logits: list[np.ndarray] = []  # shape [batch, N_LABELS]

    for batch in _iter_batches(notes_df, batch_size):
        texts = [_normalize_text(t) for t in batch["text"].tolist()]
        input_ids, attention_mask = _tokenize_batch(texts, tokenizer, max_length)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)
        stacked = torch.cat([o.view(-1, 1) for o in outputs], dim=1)
        logits.append(stacked.detach().cpu().numpy().astype(np.float32))
        note_ids.extend(batch["note_id"].tolist())

    logit_arr = np.concatenate(logits, axis=0)
    prob_arr = 1.0 / (1.0 + np.exp(-logit_arr))
    flag_arr = (prob_arr >= binarization_threshold).astype(np.uint8)

    out = pd.DataFrame({"note_id": note_ids})
    for i, label in enumerate(LABEL_ORDER):
        out[f"logit_{label}"] = logit_arr[:, i].astype(np.float32)
        out[f"prob_{label}"] = prob_arr[:, i].astype(np.float32)
        out[f"flag_{label}"] = flag_arr[:, i]
    return out
