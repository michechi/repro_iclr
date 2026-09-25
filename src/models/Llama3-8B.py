import pandas as pd
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from huggingface_hub import login
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

YOUR_HF_TOKEN = os.getenv("HF_TOKEN")
if YOUR_HF_TOKEN is None:
    raise ValueError("Set the HF_TOKEN environment variable for authentication.")
login(YOUR_HF_TOKEN)

MODEL_NAME = 'meta-llama/Llama-3.1-8B'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=YOUR_HF_TOKEN, cache_dir = '.')
tokenizer.pad_token = tokenizer.eos_token

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# model = AutoModelForSequenceClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=2,
#     quantization_config=quant_config,
#     device_map='auto',
#     token=YOUR_HF_TOKEN
# )

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    torch_dtype=torch.bfloat16,  # Efficient mixed precision
    device_map='auto',
    token=YOUR_HF_TOKEN,
    cache_dir='.'
)

model.config.pad_token_id = tokenizer.eos_token_id

# Here to test different LoraConfig parameters
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

landmark_df = pd.read_csv('landmark_df.csv')

max_visits = 3
visit_counts = landmark_df['subject_id'].value_counts()
selected_patients = visit_counts[visit_counts == max_visits].index
df_selected = landmark_df[landmark_df['subject_id'].isin(selected_patients)].copy()

SEED = 42

class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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

results = []

for landmark_visit in range(1, max_visits + 1):
    logger.info(f"Fine-tuning at Landmark {landmark_visit}")

    df_subset = df_selected[df_selected['landmark_visit'] == landmark_visit]

    patients = df_subset['subject_id'].unique()

    train_patients, test_patients = train_test_split(
        patients, test_size=0.2, random_state=SEED,
        stratify=df_subset.groupby('subject_id')['death_in_90days'].max()
    )

    train_df = df_subset[df_subset['subject_id'].isin(train_patients)].copy()
    test_df = df_subset[df_subset['subject_id'].isin(test_patients)].copy()

    train_texts = train_df.apply(narrative_prompt, axis=1).tolist()
    train_labels = train_df['death_in_90days'].tolist()
    test_texts = test_df.apply(narrative_prompt, axis=1).tolist()
    test_labels = test_df['death_in_90days'].tolist()

    # TO DEBUG
    # print(set(train_labels))  # Should only print {0, 1}
    # print(train_texts[:3])    # Inspect sample texts

    # print(set(test_labels))  # Should only print {0, 1}
    # print(test_texts[:3])    # Inspect sample texts

    train_dataset = ClinicalDataset(train_texts, train_labels, tokenizer)
    test_dataset = ClinicalDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6) # 2e-5
    epochs = 10
    patience = 2
    best_auc = 0.0
    epochs_no_improve = 0
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

    for epoch in range(epochs):
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

        logger.info(
            f'Epoch {epoch+1}/{epochs}, '
            f'Train Loss: {total_loss/len(train_loader):.4f} '
        )

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits.float(), dim=-1)[:, 1].cpu().numpy()
                # probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch['labels'].cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        logger.info(
            f'Epoch {epoch+1}/{epochs}, '
            f'Validation AUC: {val_auc:.4f} '
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), './best/best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load('./best/best_model.pt'))

    mortality_rate = np.mean(test_labels) * 100
    results.append({'Landmark Visit': landmark_visit, 'Num Patients': len(test_df), 'Mortality (%)': f"{mortality_rate:.1f}%", 'AUC': f"{best_auc:.4f}"})

results_df = pd.DataFrame(results)
logger.info(results_df)
