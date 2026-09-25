import os
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from huggingface_hub import login

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script with dynamic arguments")

    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name")
    parser.add_argument("--cache_dir", type=str, default=".",
                        help="Directory for cache and model saving")
    parser.add_argument("--input_csv", type=str, default="landmark_df.csv",
                        help="Path to the input CSV file")
    parser.add_argument("--max_visits", type=int, default=3,
                        help="Maximum number of landmarks to process")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for the DataLoader")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum token length")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for reproducibility")

    return parser.parse_args()


def load_tokenizer(model_name, hf_token, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, hf_token, tokenizer, cache_dir):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        token=hf_token,
        cache_dir=cache_dir
    )
    model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# To modify
def narrative_prompt(row):
    narrative = f"Patient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the {row['num_total_visits']} visit."
    if row['days_since_previous_visit'] != -1:
        narrative += f" The last visit happened {row['days_since_previous_visit']} days ago."
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        narrative += f" Medical history includes: {row['diag_text']}."
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f" Current medications are: {row['med_text']}."
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f" Procedures performed: {row['proc_text']}."
    return narrative


def train_one_epoch(model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {k: v.cuda() for k, v in batch.items()}
        loss = model(**inputs).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader):
    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1].cpu().numpy()
            val_preds.extend(probs)
            val_labels.extend(batch['labels'].cpu().numpy())

    val_auc = roc_auc_score(val_labels, val_preds)
    return val_auc, val_labels


def get_best_model_path(model_name, landmark_visit, cache_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    filename = f"best_model_{safe_model_name}_landmark{landmark_visit}_{timestamp}.pt"
    best_model_dir = os.path.join(cache_dir, 'best')
    os.makedirs(best_model_dir, exist_ok=True)
    return os.path.join(best_model_dir, filename)


def fine_tune_model(model, train_loader, val_loader, model_name, landmark_visit, cache_dir, epochs, patience, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )

    best_auc = 0.0
    epochs_no_improve = 0
    best_model_path = get_best_model_path(model_name, landmark_visit, cache_dir)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")

        val_auc, _ = validate(model, val_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs}, Validation AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # model.load_state_dict(torch.load(best_model_path)) # redundant
    return best_auc, best_model_path


def main():
    args = parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Set the HF_TOKEN environment variable for authentication.")
    login(hf_token)

    tokenizer = load_tokenizer(args.model_name, hf_token, args.cache_dir)
    model = load_model(args.model_name, hf_token, tokenizer, args.cache_dir)

    landmark_df = pd.read_csv(args.input_csv)

    visit_counts = landmark_df['subject_id'].value_counts()
    selected_patients = visit_counts[visit_counts == args.max_visits].index
    df_selected = landmark_df[landmark_df['subject_id'].isin(selected_patients)].copy()

    results = []

    for landmark_visit in range(1, args.max_visits + 1):
        logger.info(f"Fine-tuning at Landmark {landmark_visit}")

        df_subset = df_selected[df_selected['landmark_visit'] == landmark_visit]
        patients = df_subset['subject_id'].unique()

        train_patients, test_patients = train_test_split(
            patients,
            test_size=0.2,
            random_state=args.seed,
            stratify=df_subset.groupby('subject_id')['death_in_90days'].max()
        )

        train_df = df_subset[df_subset['subject_id'].isin(train_patients)].copy()
        test_df = df_subset[df_subset['subject_id'].isin(test_patients)].copy()

        train_texts = train_df.apply(narrative_prompt, axis=1).tolist()
        train_labels = train_df['death_in_90days'].tolist()
        test_texts = test_df.apply(narrative_prompt, axis=1).tolist()
        test_labels = test_df['death_in_90days'].tolist()

        train_dataset = ClinicalDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
        test_dataset = ClinicalDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        best_auc, best_model_path = fine_tune_model(
            model, train_loader, test_loader, args.model_name, landmark_visit,
            args.cache_dir, args.epochs, args.patience, args.lr
        )
        logger.info(f"Best model saved at: {best_model_path}")

        mortality_rate = np.mean(test_labels) * 100
        results.append({
            'Landmark Visit': landmark_visit,
            'Num Patients': len(test_df),
            'Mortality (%)': f"{mortality_rate:.1f}%",
            'AUC': f"{best_auc:.4f}",
            'Model Path': best_model_path
        })

    results_df = pd.DataFrame(results)
    logger.info(results_df)


if __name__ == "__main__":
    main()