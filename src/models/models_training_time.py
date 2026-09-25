import os
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import logging
import datetime
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig, AutoModelForCausalLM
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

# New class - CausalLM
class CausalLMWithClassificationHead(nn.Module):
    """
    Wrapper combining a CausalLM with a classification head
    
    The model learns SIMULTANEOUSLY:
    1. Next token prediction (to learn temporal patterns)
    2. Mortality classification (for the final task)
    """
    
    def __init__(self, backbone_model, num_classes=2):
        super().__init__()
        
        self.backbone = backbone_model
        self.config = backbone_model.config
        self.num_classes = num_classes
        
        # Classification head operating on the hidden states
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size // 2, num_classes)
        )

        self.classification_head = self.classification_head.to(
            dtype=backbone_model.dtype, 
            device=backbone_model.device
            )
        
        
    def forward(self, input_ids, attention_mask=None, labels=None, mortality_labels=None):
        """
        Forward pass computing BOTH losses:
        1. Causal LM loss (next token prediction)
        2. Classification loss (mortality prediction)
        """
        
        # 1. Pass through the CausalLM backbone
        causal_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # For next token prediction
            output_hidden_states=True,
            return_dict=True
        )
        
        # 2. Extract hidden states for classification
        hidden_states = causal_outputs.hidden_states[-1]  # Last layer
        
        if mortality_labels is not None:
            # Extract representations at the mortality token positions
            batch_size = input_ids.shape[0]
            classification_representations = []
            
            for b in range(batch_size):
                # Find the last non-padding position for this example
                if attention_mask is not None:
                    # Use the last non-padding token
                    last_token_pos = attention_mask[b].sum().item() - 1
                else:
                    # Use the last token of the sequence
                    last_token_pos = input_ids.shape[1] - 1
                    
                # Extract representation for classification
                classification_representations.append(hidden_states[b, last_token_pos, :])
            
            # Stack into a tensor
            classification_input = torch.stack(classification_representations)
            
            # 4. Compute classification logits
            classification_logits = self.classification_head(classification_input)
            
            # 5. Compute classification loss
            classification_loss = nn.functional.cross_entropy(
                classification_logits, mortality_labels
            )
            
            # 6. Combine the losses
            total_loss = causal_outputs.loss + classification_loss
            
            return {
                'loss': total_loss,
                'causal_loss': causal_outputs.loss,
                'classification_loss': classification_loss,
                'logits': classification_logits,  # For compatibility with the existing code
                'causal_logits': causal_outputs.logits,
                'hidden_states': hidden_states
            }
        
        else:
            # Inference only - extract representations for classification
            batch_size = input_ids.shape[0]
            classification_representations = []
            
            for b in range(batch_size):
                if attention_mask is not None:
                    last_token_pos = attention_mask[b].sum().item() - 1
                else:
                    last_token_pos = input_ids.shape[1] - 1
                    
                classification_representations.append(hidden_states[b, last_token_pos, :])
            
            classification_input = torch.stack(classification_representations)
            classification_logits = self.classification_head(classification_input)
            
            return {
                'loss': causal_outputs.loss,
                'logits': classification_logits,
                'causal_logits': causal_outputs.logits,
                'hidden_states': hidden_states
            }

    def resize_token_embeddings(self, new_num_tokens):
        """Delegate to the backbone model"""
        return self.backbone.resize_token_embeddings(new_num_tokens)


def load_tokenizer(model_name, model_type, hf_token, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if model_type == "general" else None,
        cache_dir=cache_dir,
        force_download=True,
        local_files_only=False
    )
    return tokenizer

def load_model_causal(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft, use_quantization):
    """
    Modified version that loads a CausalLM instead of SequenceClassification
    """
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

        # LOAD CAUSAL LM INSTEAD OF SEQUENCE CLASSIFICATION
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=bnb_config if use_quantization else None
        )
        
        base_model.config.pad_token_id = tokenizer.pad_token_id
        
        # WRAPPER WITH CLASSIFICATION HEAD
        model = CausalLMWithClassificationHead(base_model, num_classes=2)
        
        if use_peft:
            # PEFT only on the backbone, not on the head
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.1,
                bias='none',
                task_type="CAUSAL_LM"  # Changed from SEQ_CLS
            )
            model.backbone = get_peft_model(model.backbone, lora_config)
            model.backbone.print_trainable_parameters()
    
    else:
        # For medical models, keep the original behaviour
        # LOAD CAUSAL LM INSTEAD OF SEQUENCE CLASSIFICATION
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=bnb_config if use_quantization else None
        )
        
        base_model.config.pad_token_id = tokenizer.pad_token_id
        
        # WRAPPER WITH CLASSIFICATION HEAD
        model = CausalLMWithClassificationHead(base_model, num_classes=2)
    
    return model

def train_and_evaluate_causal(model, train_loader, val_loader, args, landmark_visit):
    """
    Training loop modified to handle the combined losses
    """
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
    
    best_auc, best_f1, best_val_loss = 0.0, 0.0, float("inf")
    epochs_no_improve = 0
    best_model_path = get_best_model_path(args, landmark_visit)
    
    logger.info(f"Early stopping criterion: {args.early}")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        total_causal_loss = 0
        total_classification_loss = 0
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            
            # Total loss (causal + classification)
            loss = outputs['loss'] / args.gradient_accumulation_steps
            loss.backward()
            
            total_train_loss += loss.item()
            total_causal_loss += outputs['causal_loss'].item()
            total_classification_loss += outputs['classification_loss'].item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_causal_loss = total_causal_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)

        # VALIDATION
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                
                val_loss = outputs['loss']
                total_val_loss += val_loss.item()
                
                # Use the classification logits (not the causal ones)
                probs = torch.softmax(outputs['logits'], dim=-1)[:, 1].cpu().float().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch['mortality_labels'].cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        threshold = 0.5
        val_preds_binary = (np.array(val_preds) >= threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

        logger.info(
            f"Landmark {landmark_visit} | Epoch {epoch+1}/{args.epochs} | "
            f"Total Loss: {avg_train_loss:.4f} | "
            f"Causal Loss: {avg_causal_loss:.4f} | "
            f"Classification Loss: {avg_classification_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"F1-Score: {val_f1:.4f}"
        )
        
        # Early stopping (same as the existing code)
        condition = False
        if args.early == "auc":
            condition = val_auc > best_auc
        elif args.early == "loss":
            condition = avg_val_loss < best_val_loss
        elif args.early == "f1":
            condition = val_f1 > best_f1

        if condition:
            best_auc = val_auc
            best_f1 = val_f1
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
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