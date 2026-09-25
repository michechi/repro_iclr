import os
import random
import logging
import datetime
import argparse
import ast
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from huggingface_hub import login
from collections import Counter
from itertools import product

from src.data.prompts import (
    no_narrative_prompt, naive_narrative_prompt, compact_narrative_prompt,
    full_narrative, full_narrative_no_time, full_narrative_no_time_rnd,
    compact_no_time_prompt, compact_no_time_prompt_rnd, compact_narrative_humanstyle_prompt
)

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Universal finetuning script for LLMs or MedBERT-like models.")
    parser.add_argument("--model_type", type=str, choices=["llm", "medbert"], default="llm",
                        help="Model type: llm or medbert")
    
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model from Hugging Face")
    
    parser.add_argument("--peft", action="store_true", help="Use PEFT (LLMs only)")

    parser.add_argument("--use_quantization", action="store_true",
                        help="Quantization 4 bit")
    
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for cache and saving models")
    
    parser.add_argument("--input_csv", type=str, default="./data/landmark_df_evo.csv",
                        help="Path to the input CSV file")
    
    parser.add_argument("--train_csv", type=str, default="./data/landmark_evo_train.csv",
                        help="Path to the training CSV file")
    
    parser.add_argument("--val_csv", type=str, default="./data/landmark_evo_vali.csv",
                        help="Path to the validation CSV file")
    
    parser.add_argument("--test_csv", type=str, default="./data/landmark_evo_test.csv",
                        help="Path to the test CSV file") # evo as well 
    
    parser.add_argument("--prompt_type", type=str, choices=["naive", "compact", "compact_no_time", "compact_narrative", "no", "no_set", "full", "full_no_time", "full_no_time_rnd"], default="compact",
                        help="Prompting type to use: 'naive', 'compact' or 'no' (no narrative)")
    
    parser.add_argument("--max_visits", type=int, default=3,
                        help="number of visits to consider for each patient (max_visits)")
    
    parser.add_argument("--all_landmarks", action="store_true",
                        help="Process all landmark (from 1 to max_visits) or just the last one")
    
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoader")
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                            help="Number of steps for gradient accumulation (useful for large models)")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Num epochs for training")
    
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")
    
    parser.add_argument("--max_length", type=int, default=512,
                        help="Token Max lenght")
    
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    
    parser.add_argument("--early", type=str, choices=["auc", "loss", "f1"], default="auc",
                        help="Early stopping criterion: 'auc', 'loss' or 'f1")
    
    parser.add_argument("--seed", type=int, default=9550,
                        help="Seed for reproducibility")

    parser.add_argument("--when_counting_death", type=str, choices=["last_visit", "landmark"], default="last_visit",
                    help="When counting death: 'last_visit' (considering the last visit) or 'landmark' (considering the current landmark visit)")
    
    args = parser.parse_args()

    # Log arguemnts values
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args


def load_tokenizer(model_name, model_type, hf_token, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if model_type == "llm" else None,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    return tokenizer

def load_model(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft, use_quantization):
    if model_type == "llm":

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
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=None if use_quantization else bnb_config  # Use bnb for quantization
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
            num_labels=2,
            cache_dir=cache_dir
        )
    return model

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

def train_and_evaluate(model, train_loader, val_loader, args, landmark_visit):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )
    best_model_path = get_best_model_path(args, landmark_visit)
    best_auc, best_f1, best_val_loss = 0.0, 0.0, float("inf")
    epochs_no_improve = 0
    
    logger.info(f"Early stopping criterion: {args.early}")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            total_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                val_loss = outputs.loss
                total_val_loss += val_loss.item()
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().float().numpy() # Try
                val_preds.extend(probs)
                val_labels.extend(batch['labels'].cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)

        # Compute F1-score
        threshold = 0.5
        val_preds_binary = (np.array(val_preds) >= threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

        logger.info(
            f"Landmark {landmark_visit} | Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Validation AUC: {val_auc:.4f} | "
            f"F1-Score: {val_f1:.4f}"
        )
        
        # Early stopping logic
        condition = False
        if args.early == "auc":
            condition = val_auc > best_auc
        elif args.early == "loss":
            condition = avg_val_loss < best_val_loss
        elif args.early == "f1":
            condition = val_f1 > best_f1
        else:
            raise ValueError("Invalid early stopping criterion. Use 'auc', 'loss' or 'f1'.")

        if condition == True:
            best_auc = val_auc
            best_f1 = val_f1
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved at {best_model_path} with AUC: {best_auc:.4f}, F1: {best_f1:.4f}, Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in epoch {epoch+1}. Current best AUC: {best_auc:.4f}, F1: {best_f1:.4f}, Val Loss: {best_val_loss:.4f}. "
                        f"Epochs without improvement: {epochs_no_improve}/{args.patience}")
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}. Best AUC: {best_auc:.4f}, F1: {best_f1:.4f}, Val Loss: {best_val_loss:.4f}")
                break

    model.load_state_dict(torch.load(best_model_path))
    return best_auc, best_model_path, best_f1, best_val_loss, epoch + 1

def get_best_model_path(args, landmark_visit):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model_name.replace("/", "_")
    filename = f"best_model_{safe_model_name}_{args.seed}_landmark{landmark_visit}_{args.prompt_type}_{args.max_visits}_{args.max_length}_{args.when_counting_death}_all_landmarks_{args.all_landmarks}_{timestamp}_.pt"
    best_model_dir = os.path.join(args.cache_dir, 'best')
    os.makedirs(best_model_dir, exist_ok=True)
    return os.path.join(best_model_dir, filename)

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95, seed=42):
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        score = roc_auc_score(np.array(y_true)[indices], np.array(y_pred)[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper = sorted_scores[int((alpha + (1.0 - alpha) / 2) * len(sorted_scores))]
    return lower, upper

sys.argv = [''] + [
    '--model_type', 'medbert',
    '--model_name', 'Charangan/MedBERT', #    emilyalsentzer/Bio_ClinicalBERT  meta-llama/Llama-3.1-8B answerdotai/ModernBERT-large Charangan/MedBERT
    #'--peft',
    # '--use_quantization',
    '--cache_dir', './cache',
    '--input_csv', './src/landmark_df_evo_correct.csv',
    '--train_csv', './data/splitted/landmark_evo_train_dod_fxd.csv',
    '--val_csv', './data/splitted/landmark_evo_vali_dod_fxd.csv',
    '--test_csv', './data/splitted/landmark_evo_test_dod_fxd.csv',
    '--prompt_type', 'compact_narrative',  # 'naive', 'compact', 'no', 'full', 'full_no_time', 'full_no_time_rnd'
    '--max_visits', '3',
    '--all_landmarks',
    '--batch_size', '8',
    '--epochs', '20',
    '--gradient_accumulation_steps', '1',
    '--patience', '3',
    '--max_length', '512',
    '--lr', '2e-5',
    '--early', 'loss',
    '--seed', '9550',
    '--when_counting_death', 'last_visit'
]

models = ["emilyalsentzer/Bio_ClinicalBERT", "meta-llama/Llama-3.1-8B", "answerdotai/ModernBERT-large", "Charangan/MedBERT"]
prompts = ["no", "naive", "full_no_time", "compact_no_time", "compact", "compact_narrative", "full"]



torch.cuda.empty_cache()
torch.cuda.is_available()

args = parse_args()

# Set seed for reproducibility
set_seed(args.seed)

hf_token = os.getenv("HF_TOKEN")

if args.model_type == "llm" and hf_token is None:
    raise ValueError("Set the HF_TOKEN environment variable for authentication.")
if args.model_type == "llm":
    login(hf_token)

tokenizer = load_tokenizer(args.model_name, args.model_type, hf_token, args.cache_dir)
model = load_model(args.model_name, args.model_type, tokenizer, args.cache_dir, hf_token, args.peft, args.use_quantization).to(device="cpu")

if tokenizer.pad_token is None:
    logger.warning("Tokenizer has no pad_token; adding [PAD] manually.")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
# After having resized the model, move it to the appropriate device    
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
model.to(device)

# if re.search("evo.csv$", args.input_csv):
#     narrative_prompt = compact_narrative_prompt
# else:
#     narrative_prompt = naive_narrative_prompt
# logger.info(f"Using narrative prompt: {narrative_prompt.__name__}")

if args.prompt_type == "naive":
    narrative_prompt = naive_narrative_prompt
elif args.prompt_type == "compact":
    narrative_prompt = compact_narrative_prompt
elif args.prompt_type == "compact_no_time":
    narrative_prompt = compact_no_time_prompt
elif args.prompt_type == "compact_no_time_rnd":
    narrative_prompt = compact_no_time_prompt_rnd
elif args.prompt_type == "compact_narrative":
    narrative_prompt = compact_narrative_humanstyle_prompt
elif args.prompt_type == "no":
    narrative_prompt = no_narrative_prompt
elif args.prompt_type == "full":
    narrative_prompt = full_narrative
elif args.prompt_type == "full_no_time":
    narrative_prompt = full_narrative_no_time
elif args.prompt_type == "full_no_time_rnd":
    narrative_prompt = full_narrative_no_time_rnd
else:
    raise ValueError("Invalid prompt type. Use 'naive' or 'compact'.")
logger.info(f"Using prompt type: {args.prompt_type}")

# Reading data
#landmark_df = pd.read_csv(args.input_csv, na_values=['', 'None', 'NaN', 'na', 'nan'])
#landmark_df = landmark_df.fillna('')
full_train_df = pd.read_csv(args.train_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
full_val_df = pd.read_csv(args.val_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
full_test_df = pd.read_csv(args.test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

# For testing problems in landmark/last_visit scenarios
full_train_df["correspondence"] = (full_train_df["death_in_90days"] == full_train_df["death_90days_landmark"]).astype(int)
full_test_df["correspondence"] = (full_test_df["death_in_90days"] == full_test_df["death_90days_landmark"]).astype(int)
full_val_df["correspondence"] = (full_val_df["death_in_90days"] == full_val_df["death_90days_landmark"]).astype(int)


# Older version, now we are splitting the data in the splitting_data.py script
filtered_datasets = []
for dataset in [full_train_df, full_val_df, full_test_df]:
    visit_counts = dataset['subject_id'].value_counts()
    selected_patients = visit_counts[visit_counts == args.max_visits].index
    dataset_selected = dataset[dataset['subject_id'].isin(selected_patients)].copy()
    filtered_datasets.append(dataset_selected)

train_df_selected, val_df_selected, test_df_selected = filtered_datasets

predictions_list, labels_list, results = [], [], []

if args.all_landmarks:
    logger.info(f"Processing all landmarks from 1 to {args.max_visits}")
    start = 1
    end = args.max_visits + 1
else:
    logger.info(f"Processing only the last landmark visit: {args.max_visits}")
    start = args.max_visits
    end = args.max_visits + 1

landmark_visit = 3 # >>>>>> Change this to the desired landmark visit <<<<<<<
logger.info(f"Preparing data for Landmark {landmark_visit}")

train_df = train_df_selected[train_df_selected['landmark_visit'] == landmark_visit].copy()
val_df = val_df_selected[val_df_selected['landmark_visit'] == landmark_visit].copy()
test_df = test_df_selected[test_df_selected['landmark_visit'] == landmark_visit].copy()

train_texts = train_df.apply(narrative_prompt, axis=1).tolist()
val_texts = val_df.apply(narrative_prompt, axis=1).tolist()
test_texts = test_df.apply(narrative_prompt, axis=1).tolist()

if args.when_counting_death == "last_visit":
    train_labels = train_df['death_in_90days'].tolist()
    val_labels = val_df['death_in_90days'].tolist()
    test_labels = test_df['death_in_90days'].tolist()
elif args.when_counting_death == "landmark":
    train_labels = train_df['death_90days_landmark'].tolist()
    val_labels = val_df['death_90days_landmark'].tolist()
    test_labels = test_df['death_90days_landmark'].tolist()

# Mean text length
avg_train_length = np.mean([len(text.split()) for text in train_texts])
avg_val_length = np.mean([len(text.split()) for text in val_texts])
avg_test_length = np.mean([len(text.split()) for text in test_texts])
logger.info(f"Average train text length: {avg_train_length:.2f} words")
logger.info(f"Average validation text length: {avg_val_length:.2f} words")      
logger.info(f"Average test text length: {avg_test_length:.2f} words")
logger.info(f"Training on {len(train_texts)} samples, validating on {len(val_texts)}, testing on {len(test_texts)} samples")

from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np

def compute_truncation_stats(texts, tokenizer, max_length=512, show_plots=True):
    token_lengths = []
    truncation_amounts = []   # O_i = overflow in token
    word_truncation_amounts = []
    loss_fractions = []       # r_i = O_i / L_i

    for text in texts:
        tokens_full = tokenizer.encode(text, truncation=False)
        tokens_truncated = tokenizer.encode(text, truncation=True, max_length=max_length)

        L_i = len(tokens_full)
        token_lengths.append(L_i)

        if L_i > 0:
            O_i = max(0, L_i - max_length)
            if O_i > 0:
                truncation_amounts.append(O_i)

                # fraction of content lost (always in [0,1])
                loss_fractions.append(O_i / L_i)

                # Rough word-level estimate
                original_word_count = len(text.split())
                truncated_text = tokenizer.decode(tokens_truncated, skip_special_tokens=True)
                truncated_word_count = len(truncated_text.split())
                word_truncation_amounts.append(max(0, original_word_count - truncated_word_count))
            else:
                # no truncation ⇒ fraction lost = 0
                loss_fractions.append(0.0)

    total = len(texts)
    truncated = len(truncation_amounts)

    p = (truncated / total) if total else 0.0
    mu_tokens = (np.mean(truncation_amounts) if truncated else 0.0)
    L_avg = (np.mean(token_lengths) if total else 0.0)
    L_trunc_avg = (np.mean([L for L in token_lengths if L > max_length]) if truncated else 0.0)

    # 1) Content-Loss Index (mean of O_i / L_i) ∈ [0,1]
    CLI = float(np.mean(loss_fractions)) if total else 0.0

    # 2) Safe conditional index p * (mu / L_trunc_avg) ∈ [0,1]
    CI_trunc = float(p * (mu_tokens / L_trunc_avg)) if L_trunc_avg > 0 else 0.0

    # --- Printouts
    print(f"Total samples: {total}")
    print(f"Truncated samples: {truncated} ({p*100:.2f}%)")

    if truncated:
        print(f"\n--- Token Truncation ---")
        print(f"Avg tokens truncated (μ): {mu_tokens:.2f}")
        print(f"Max tokens truncated: {max(truncation_amounts)}")

        print(f"\n--- Word Truncation Estimate ---")
        print(f"Avg words truncated: {np.mean(word_truncation_amounts):.2f}")
        print(f"Max words truncated: {max(word_truncation_amounts)}")

    else:
        print("No samples were truncated.")

    print(f"\n--- Indices in [0,1] ---")
    print(f"CLI (mean overflow/length): {CLI:.4f}")
    print(f"CI_trunc = p * (μ / L_trunc_avg): {CI_trunc:.4f}")

    if show_plots and truncated:
        plt.hist(truncation_amounts, bins=30)
        plt.title("Distribution of Truncated Tokens")
        plt.xlabel("Tokens truncated")
        plt.ylabel("Number of samples")
        plt.show()

        plt.hist(loss_fractions, bins=30)
        plt.title("Distribution of Content Loss Fraction (O_i / L_i)")
        plt.xlabel("Fraction lost per prompt")
        plt.ylabel("Number of samples")
        plt.show()

    return {
        "total": total,
        "truncated": truncated,
        "p": float(p),
        "mu_tokens": float(mu_tokens),
        "L_avg": float(L_avg),
        "L_trunc_avg": float(L_trunc_avg),
        "CLI": CLI,
        "CI_trunc": CI_trunc,
        "max_tokens_truncated": int(max(truncation_amounts)) if truncated else 0,
        "avg_words_truncated": float(np.mean(word_truncation_amounts)) if truncated else 0.0,
        "max_words_truncated": int(max(word_truncation_amounts)) if truncated else 0,
    }

# Example:
metrics = compute_truncation_stats(train_texts, tokenizer, max_length=args.max_length, show_plots=False)
print(metrics)

