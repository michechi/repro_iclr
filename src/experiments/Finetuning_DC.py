import os
import random
import logging
import datetime
import argparse
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

from src.data.prompts import (
    no_narrative_prompt, naive_narrative_prompt, compact_narrative_prompt,
    full_narrative, full_narrative_no_time, full_narrative_no_time_rnd,
    compact_no_time_prompt, compact_no_time_prompt_rnd, compact_narrative_humanstyle_prompt,
    semi_full_narrative, reversed_naive_narrative_prompt, last_info_prompt, full_narrative_num2words
)

torch.set_float32_matmul_precision('high') 
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

    parser.add_argument("--CI", action="store_true", help="if true, runs for three different seeds for then compute the average results")

    parser.add_argument("--model_type", type=str, choices=["general", "medical"], default="general",
                        help="Model type: general-purpose or medical-purpose (MedBERT-like)")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model from Hugging Face")

    parser.add_argument("--peft", action="store_true", help="Use PEFT (LLMs only)")

    parser.add_argument("--use_quantization", action="store_true",
                        help="Quantization 4 bit")

    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for cache and saving models")

    parser.add_argument("--input_csv", type=str, default="./src/landmark_df_evo_correct.csv",
                        help="Path to the input CSV file")

    parser.add_argument("--train_csv", type=str, default="./data/splitted/landmark_evo_train_dod_fxd.csv",
                        help="Path to the training CSV file")

    parser.add_argument("--val_csv", type=str, default="./data/splitted/landmark_evo_vali_dod_fxd.csv",
                        help="Path to the validation CSV file")

    parser.add_argument("--test_csv", type=str, default="./data/splitted/landmark_evo_test_dod_fxd.csv",
                        help="Path to the test CSV file") # evo as well 
    
    parser.add_argument("--when_counting_death", type=str, choices=["last_visit", "landmark"], default="last_visit",
                        help="When counting death: 'last_visit' (considering the last visit) or 'landmark' (considering the current landmark visit)")

    parser.add_argument("--prompt_type", type=str, choices=["naive", "compact", "compact_no_time", "compact_no_time_rnd", 
                                                            "no", "full", "full_no_time", "full_no_time_rnd", "compact_narrative",
                                                            "semi_full_narrative", "reversed_naive_narrative_prompt", "last_info_prompt", "full_narrative_num2words"], default="compact",
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

    parser.add_argument("--early", type=str, choices=["auc", "loss", "f1"], default="loss",
                        help="Early stopping criterion: 'auc', 'loss' or 'f1")

    parser.add_argument("--seed", type=int, default=9550,
                        help="Seed for reproducibility")
    
    args = parser.parse_args()

    # Log arguemnts values
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args

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
    best_auc = 0.0
    best_val_loss = float("inf")
    best_f1 = 0.0
    epochs_no_improve = 0
    best_model_path = get_best_model_path(args, landmark_visit)
    
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

        # Old without gradient accumulation
        # for batch in train_loader:
        #     optimizer.zero_grad()
        #     inputs = {k: v.to(device) for k, v in batch.items()}
        #     outputs = model(**inputs)
        #     loss = outputs.loss
        #     loss.backward()
        #     optimizer.step()
        #     scheduler.step()
        #     total_train_loss += loss.item()
        # avg_train_loss = total_train_loss / len(train_loader)

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
        
        # Early stopping based on the specified criterion
        condition = False
        if args.early == "auc":
            condition = val_auc > best_auc
        elif args.early == "loss":
            condition = avg_val_loss < best_val_loss
        elif args.early == "f1":
            condition = val_f1 > best_f1
        else:
            raise ValueError("Invalid early stopping criterion. Use 'auc', 'loss' or 'f1'.")

        if condition:
            logger.info(f"Improvement detected at epoch {epoch+1}. Saving model.")
            best_auc = val_auc
            best_f1 = val_f1
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
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

def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    hf_token = os.getenv("HF_TOKEN")

    if args.model_type == "general-purpose" and hf_token is None:
        raise ValueError("Set the HF_TOKEN environment variable for authentication.")
    if args.model_type == "general-purpose":
        login(hf_token)

    tokenizer = load_tokenizer(args.model_name, args.model_type, hf_token, args.cache_dir)
    model = load_model(args.model_name, args.model_type, tokenizer, args.cache_dir, hf_token, args.peft, args.use_quantization).to(device="cpu")
    
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token; adding [PAD] manually.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    # After having resized the model, move it to the appropriate device    
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    if args.prompt_type == "naive":
        narrative_prompt = naive_narrative_prompt
    elif args.prompt_type == "compact":
        narrative_prompt = compact_narrative_prompt
    elif args.prompt_type == "compact_no_time":
        narrative_prompt = compact_no_time_prompt
    elif args.prompt_type == "compact_no_time_rnd":
        narrative_prompt = compact_no_time_prompt_rnd
    elif args.prompt_type == "no":
        narrative_prompt = no_narrative_prompt
    elif args.prompt_type == "full":
        narrative_prompt = full_narrative
    elif args.prompt_type == "full_no_time":
        narrative_prompt = full_narrative_no_time
    elif args.prompt_type == "full_no_time_rnd":
        narrative_prompt = full_narrative_no_time_rnd
    elif args.prompt_type == "compact_narrative":
        narrative_prompt = compact_narrative_humanstyle_prompt
    elif args.prompt_type == "semi_full_narrative":
        narrative_prompt = semi_full_narrative
    elif args.prompt_type == "reversed_naive_narrative_prompt":
        narrative_prompt = reversed_naive_narrative_prompt
    elif args.prompt_type == "last_info_prompt":
        narrative_prompt = last_info_prompt
    elif args.prompt_type == "full_narrative_num2words":
       narrative_prompt = full_narrative_num2words
    else:
        raise ValueError("Invalid prompt type. Use 'naive' or 'compact'.")
    logger.info(f"Using prompt type: {args.prompt_type}")
    
    logger.info(f"Model type: {args.model_type}, Model name: {args.model_name}") 

    # Reading data
    full_train_df = pd.read_csv(args.train_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    full_val_df = pd.read_csv(args.val_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    full_test_df = pd.read_csv(args.test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    # Older version, now we are splitting the data in the splitting_data.py script
    filtered_datasets = []
    for dataset in [full_train_df, full_val_df, full_test_df]:
        visit_counts = dataset['subject_id'].value_counts()
        selected_patients = visit_counts[visit_counts == args.max_visits].index # >= instead of == to include patients with more than max_visits
        dataset_selected = dataset[dataset['subject_id'].isin(selected_patients)].copy()
        filtered_datasets.append(dataset_selected)
    
    # visit_counts = landmark_df['subject_id'].value_counts()
    # selected_patients = visit_counts[visit_counts == args.max_visits].index
    # df_selected = landmark_df[landmark_df['subject_id'].isin(selected_patients)].copy()

    train_df_selected, val_df_selected, test_df_selected = filtered_datasets

    results = []

    if args.all_landmarks:
        logger.info(f"Processing all landmarks from 1 to {args.max_visits}")
        start = 1
        end = args.max_visits + 1
    else:
        logger.info(f"Processing only the last landmark visit: {args.max_visits}")
        start = args.max_visits
        end = args.max_visits + 1

    for landmark_visit in range(start, end):
        logger.info(f"Preparing data for Landmark {landmark_visit}")
        
        # Set the seed for reproducibility (for each landmark visit)
        set_seed(args.seed)

        # First, clean
        if 'model' in globals():
            del model
        if 'tokenizer' in globals():
            del tokenizer
        
        torch.cuda.empty_cache()

        # Load model and tokenizer
        hf_token = os.getenv("HF_TOKEN")
        if args.model_type == "general" and hf_token is None:
            raise ValueError("Set the HF_TOKEN environment variable for authentication.")
        if args.model_type == "general":
            login(hf_token)

        tokenizer = load_tokenizer(args.model_name, args.model_type, hf_token, args.cache_dir)    
        model = load_model(args.model_name, args.model_type, tokenizer, args.cache_dir, hf_token, args.peft, args.use_quantization).to(device="cpu")
    
        if tokenizer.pad_token is None:
            logger.warning("Tokenizer has no pad_token; adding [PAD] manually.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            
        # After having resized the model, move it to the appropriate device    
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        train_df = train_df_selected[train_df_selected['landmark_visit'] == landmark_visit].copy()
        val_df = val_df_selected[val_df_selected['landmark_visit'] == landmark_visit].copy()
        test_df = test_df_selected[test_df_selected['landmark_visit'] == landmark_visit].copy()

        train_texts = train_df.apply(narrative_prompt, axis=1).tolist()
        val_texts = val_df.apply(narrative_prompt, axis=1).tolist()
        test_texts = test_df.apply(narrative_prompt, axis=1).tolist()
        
        logger.info(f"Example train text: {train_texts[0]}")

        if args.when_counting_death == "last_visit":
            # Here we are considering for each landmark if the patient dies in the 90 days after the very last visit
            # (useful to test whether temporal information is more important if we are not considering the last visit which
            # it could contain all the relevant information to determine wheter the patient will die or not)
            train_labels = train_df['death_in_90days'].tolist()
            val_labels = val_df['death_in_90days'].tolist()
            test_labels = test_df['death_in_90days'].tolist()

        elif args.when_counting_death == "landmark":
            # Here we are considering for each landmark if the patient dies in the 90 days after the current landmark (we can determine this sinc
            # we have the information of the date of the death - dod)
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

        train_dataset = ClinicalDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
        val_dataset = ClinicalDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
        test_dataset = ClinicalDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        logger.info(f"Fine-tuning at Landmark {landmark_visit}")
        start_time = datetime.datetime.now()
        best_auc, best_model_path, best_val_f1, best_val_loss, epochs_done = train_and_evaluate(
            model, train_loader, val_loader, args, landmark_visit
        )
        elapsed_time = datetime.datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        logger.info(f"Time taken for Landmark {landmark_visit}: {elapsed_seconds:.1f} seconds")
        logger.info(f"Mean time per epoch: {elapsed_seconds / epochs_done:.1f} seconds") # This is the important one

        # FINAL EVALUATION ON THE TEST SET
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logger.info(f"Loading best model from {best_model_path} for final evaluation on test set")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        test_preds, test_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().float().numpy() # on cpu for sklearn metrics
                test_preds.extend(probs)
                test_labels.extend(batch['labels'].cpu().float().numpy()) # on cpu for sklearn metrics

        test_auc = roc_auc_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, np.array(test_preds) >= 0.5)

        logger.info(
            f"Final Test AUC (Landmark {landmark_visit}): {test_auc:.4f} | "
            f"Test F1-Score: {test_f1:.4f}"
        )

        results.append({
            'Landmark': landmark_visit,
            'Val AUC': f"{best_auc:.4f}",
            'Val F1-Score': f"{best_val_f1:.4f}",
            'Test AUC': f"{test_auc:.4f}",
            'Test F1-Score': f"{test_f1:.4f}",
            'Patients (Test)': len(test_df),
            'Model Path': best_model_path,
            'Validation Loss': f"{best_val_loss:.4f}",
            'Training Time (s)': f"{elapsed_seconds:.1f}"
        })
   
    results_df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_name.replace("/", "_")
    peft_tag = "_peft" if args.peft else ""
    output_filename = (
        f"results_{args.model_type}_{args.all_landmarks}_{model_tag}_{args.when_counting_death}_visits{args.max_visits}{peft_tag}_{args.seed}_{args.max_length}_{args.prompt_type}_{timestamp}.csv"
    )
    logger.info(f"Saving results to {output_filename}")
    file_dir = f"{args.cache_dir}/results/{args.model_type}/{args.all_landmarks}/{args.when_counting_death}/visits{args.max_visits}" 
    os.makedirs(file_dir, exist_ok=True)
    results_df.to_csv(os.path.join(file_dir, output_filename), index=False)

    print(results_df)

if __name__ == "__main__":
    main()