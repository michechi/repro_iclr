import numpy as np
import gc
import os
import random
import pandas as pd
import ast
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from typing import List, Dict, Callable, Any
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

def load_tokenizer(model_name, model_type, hf_token, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if model_type == "general" else None,
        cache_dir=cache_dir,
        force_download=True,
        local_files_only=False
    )
    return tokenizer

def load_model(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft, use_quantization):
    if model_type == "general":

        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2, # TODO: avoid to hard-code this
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=bnb_config if use_quantization else None  # Use bnb for quantization
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        if use_peft:
            # TODO: test different settings for LoraConfig
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
    else:
        # For clinical models like MedBERT or similar
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2, # TODO: avoid to hard-code this
            cache_dir=cache_dir
        )
    return model

def full_narrative_no_time_rnd(row):
    narrative = f"What is the probability of death in the next 90 days from today for this {row['age_at_landmark']}-year-old {row['gender']} patient?"

    # Diagnoses
    all_diags = []
    diag_dict = ast.literal_eval(row['diag_per_visit'])
    for diags in diag_dict.values():
        all_diags.extend(diags)
    random.shuffle(all_diags)
    narrative += "\nDiagnosis history:"
    narrative += "\n\t"+'; '.join(all_diags).strip()

    # Prescriptions
    all_meds = []
    meds_dict = ast.literal_eval(row['meds_per_visit'])
    for meds in meds_dict.values():
        all_meds.extend(meds)
    random.shuffle(all_meds)
    narrative += "\nPrescriptions history:" 
    narrative += "\n\t" + '; '.join(all_meds).strip()

    # Procedures
    all_procs = []
    proc_dict = ast.literal_eval(row['proc_per_visit'])
    for procs in proc_dict.values():
        all_procs.extend(procs)
    random.shuffle(all_procs)
    narrative += "\nProcedures history:" 
    narrative += "\n\t" + '; '.join(all_procs).strip()

    return narrative

class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512): # depending on the model!! 
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ======================= #
# TO CHANGE PARAMETERS
prompt_type = "full_no_time_rnd"  # To change
path_test_csv = "./data/splitted/landmark_evo_test_dod_fxd.csv" # To change
max_visits = 4 # To change
batch_size = 24 # To change
max_length = 2048 # To change
model_name = "meta-llama/Llama-3.1-8B" # To change
model_type = "general" # To change
cache_dir = "./cache" # To change
hf_token = os.getenv("HF_TOKEN")
use_peft = True # To change
use_quantization = False # To change
# END TO CHANGE PARAMETERS
# ======================= #

narrative_prompt = full_narrative_no_time_rnd

full_test_df = pd.read_csv(path_test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
visit_counts = full_test_df['subject_id'].value_counts()
selected_patients = visit_counts[visit_counts == max_visits].index # >= instead of == to include patients with more than max_visits
dataset_selected = full_test_df[full_test_df['subject_id'].isin(selected_patients)].copy()
test_df_selected = dataset_selected

test_df = test_df_selected[test_df_selected['landmark_visit'] == max_visits].copy()
test_texts = test_df.apply(narrative_prompt, axis=1).tolist()
test_labels = test_df['death_in_90days'].tolist()

tokenizer = load_tokenizer(model_name, model_type, hf_token, cache_dir)
model = load_model(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft, use_quantization)

if tokenizer.pad_token is None:
    print("Tokenizer has no pad_token; adding [PAD] manually.")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

model.load_state_dict(torch.load("./cache/best/best_model_meta-llama_Llama-3.1-8B_4550_landmark4_full_no_time_rnd_4_2048_last_visit_all_landmarks_False_20250827_133041_.pt"))
device = 'cuda'
def predict_fn(text_subset, tokenizer=tokenizer, device=device, model=model):
            """Prediction function for bootstrap sampling."""
            if isinstance(text_subset, np.ndarray):
                texts = text_subset.tolist()
            else:
                texts = text_subset
                
            predictions = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(**inputs)
                    batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    predictions.extend(batch_predictions)
            
            return np.array(predictions)

y_pred = predict_fn(test_texts)

f1 = f1_score(test_labels, y_pred, average='binary')
print(f1)

# From finetuning code:

def set_seed(seed_value=5550):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
set_seed(4550)
model.load_state_dict(torch.load("./cache/best/best_model_meta-llama_Llama-3.1-8B_4550_landmark4_full_no_time_rnd_4_2048_last_visit_all_landmarks_False_20250827_133041_.pt"))
device = 'cuda'
test_dataset = ClinicalDataset(test_texts, test_labels, tokenizer, max_length=max_length)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.eval()
test_preds, test_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        print("!")
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().float().numpy() # on cpu for sklearn metrics
        test_preds.extend(probs)
        test_labels.extend(batch['labels'].cpu().float().numpy()) # on cpu for sklearn metrics

test_auc = roc_auc_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, np.array(test_preds) >= 0.5)
