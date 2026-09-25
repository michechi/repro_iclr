import os
import random
import numpy as np
import torch
import argparse
import logging
import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from sklearn.metrics import roc_auc_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

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