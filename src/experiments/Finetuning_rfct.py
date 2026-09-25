import os, time, re, random, logging, datetime, argparse, ast
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from huggingface_hub import login
from collections import Counter
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig
)

from src.data.prompts import (
    no_narrative_prompt, naive_narrative_prompt, compact_narrative_prompt,
    full_narrative, full_narrative_no_time, full_narrative_no_time_rnd,
    compact_no_time_prompt, compact_no_time_prompt_rnd, compact_narrative_humanstyle_prompt,
    semi_full_narrative, reversed_naive_narrative_prompt, reversed_time_naive_narrative_prompt,
    last_info_prompt, full_narrative_num2words
)

from src.utils.seed_parsing import set_seed, parse_args
from src.models.models_training import load_tokenizer, load_model, train_and_evaluate, get_best_model_path

# Set the float32 matmul precision to 'high' for better performance
torch.set_float32_matmul_precision('high')
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)


class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512): # depending on the model!! 
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length', max_length=max_length  # change padding to 'max_length' for consistency instead of True
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    
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
    elif args.prompt_type == "reversed_time_naive_narrative_prompt":
        narrative_prompt = reversed_time_naive_narrative_prompt
    elif args.prompt_type == "full_narrative_num2words":
        args.prompt_type = full_narrative_num2words
    elif args.prompt_type == "last_info_prompt":
        narrative_prompt = last_info_prompt
    else:
        raise ValueError("Invalid prompt type. Use 'naive' or 'compact'.")
    logger.info(f"Using prompt type: {args.prompt_type}")

    # Reading data
    #landmark_df = pd.read_csv(args.input_csv, na_values=['', 'None', 'NaN', 'na', 'nan'])
    #landmark_df = landmark_df.fillna('')
    full_train_df = pd.read_csv(args.train_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    full_val_df = pd.read_csv(args.val_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    full_test_df = pd.read_csv(args.test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    # Older version, now we are splitting the data in the splitting_data.py script
    filtered_datasets = []
    for dataset in [full_train_df, full_val_df, full_test_df]:
        visit_counts = dataset['subject_id'].value_counts()
        selected_patients = visit_counts[visit_counts == args.max_visits].index
        dataset_selected = dataset[dataset['subject_id'].isin(selected_patients)].copy()
        filtered_datasets.append(dataset_selected)
    
    # visit_counts = landmark_df['subject_id'].value_counts()
    # selected_patients = visit_counts[visit_counts == args.max_visits].index
    # df_selected = landmark_df[landmark_df['subject_id'].isin(selected_patients)].copy()

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