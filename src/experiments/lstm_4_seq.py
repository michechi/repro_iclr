# Import Libraries ----------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import logging
import os

from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, \
    precision_score, recall_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModel

# Setting Parsing & Logger ---------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def parse_args(args=None):
    parser.add_argument("--cache_dir", type=str, default="./cache/",
                        help="Directory for cache e saving models")

    parser.add_argument("--tokenizer", choices=['LLM', 'Base', 'Countvect'])

    # Data
    parser.add_argument("--csv_to_use", type=str, help="CSV of training")

    parser.add_argument("--path_csv", type=str, default="./data/simulation/",
                        help="Path to file CSV for simulation data")
    
    # Seed
    parser.add_argument("--seed", type=int, default=9550,
                        help="Seed for riproducibility")
    
    # Where to save
    parser.add_argument("--output_dir", type=str, default="./results/legacy/",
                        help="Directory to save outputs")
    
    # LSTM specific arguments
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="LSTM hidden size")
    
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers")
    
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    
    # First partial parsing
    c_args, _ = parser.parse_known_args()

    # Adding conditioned arguments
    if c_args.tokenizer == "LLM":
        parser.add_argument("--model_name", required=True,
            help="Model name for tokenizer LLM")

        parser.add_argument("--max_length", type=int, default=512,
                        help="Token Max length")
        
        parser.add_argument("--embedding_dir", type=str, default="./cache/embeddings/",
                        help="Directory for saving embeddings")
        
        parser.add_argument("--first_time", action="store_true",
                        help="First time of encoding embeddings?")

    # Final parse
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Log arguments values
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args


def ordinal_encoding(X_input):
    """
    Matrix (n_samples, 20) where each values it's 0-25
    Returns 3D array for LSTM: (n_samples, sequence_length, 1)
    """
    n_samples = len(X_input)
    n_cols = len(X_input.Sequences[0].split('\x1f'))
    X = np.zeros((n_samples, n_cols), dtype=int)
    
    for i, seq in enumerate(tqdm(X_input.Sequences, desc="Ordinal encoding")):
        for j, char in enumerate(seq.split('\x1f')):
            X[i, j] = ord(char.upper()) - ord('A')
    
    # Reshape for LSTM: (samples, timesteps, features)
    return X.reshape(n_samples, n_cols, 1).astype(np.float32)


def standard_narrative_prompt(row, to_split='\x1f', column_name="Sequences"):
    events = row[column_name].split(to_split)
    prompt = f'Sequential events: {" ".join(events)}\n'
    prompt += 'Outcome (0 or 1):'
    return prompt


def get_embeddings(texts, model_name, batch_size, max_length, device, cache_dir):
    logger.info(f"Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HF_TOKEN"),
        cache_dir=cache_dir,
        force_download=True,
        local_files_only=False
    )

    model = AutoModel.from_pretrained(
        model_name,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=cache_dir,
        tie_word_embeddings=True
    ).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        all_embeddings.append(embeddings.cpu().float().numpy())
        
        del outputs, embeddings, inputs
        torch.cuda.empty_cache()
    
    embeddings_array = np.vstack(all_embeddings)
    # Reshape for LSTM: (samples, 1, embedding_dim) - treating embedding as single timestep
    return embeddings_array.reshape(embeddings_array.shape[0], 1, embeddings_array.shape[1])


# LSTM Model Definition ------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        out = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)
        
        return out


# Custom Dataset -------------------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.values if isinstance(y, pd.DataFrame) else y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Training Function ----------------------------------------------------------------
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in tqdm(dataloader, desc="Training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


# Evaluation Function --------------------------------------------------------------
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics, all_preds, all_probs, all_labels


# Main Training Loop ---------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Data Ingestion ---------------------------------------------------------------
    logger.info("Loading data...")
    X_train = pd.read_csv(f"{args.path_csv}X_train_{args.csv_to_use}.csv", 
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_train = pd.read_csv(f"{args.path_csv}y_train_{args.csv_to_use}.csv", 
                          na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    X_val = pd.read_csv(f"{args.path_csv}X_val_{args.csv_to_use}.csv", 
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_val = pd.read_csv(f"{args.path_csv}y_val_{args.csv_to_use}.csv", 
                        na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    X_test = pd.read_csv(f"{args.path_csv}X_test_{args.csv_to_use}.csv", 
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_test = pd.read_csv(f"{args.path_csv}y_test_{args.csv_to_use}.csv", 
                         na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    # Type of encoding -------------------------------------------------------------
    if args.tokenizer == "Base":
        X_train_encoded = ordinal_encoding(X_train)
        X_val_encoded = ordinal_encoding(X_val)
        X_test_encoded = ordinal_encoding(X_test)
        input_size = 1  # Single feature per timestep

    elif args.tokenizer == "LLM":
        safe_model_name = args.model_name.replace("/", "_")

        if args.first_time:
            X_train['prompt'] = X_train.apply(standard_narrative_prompt, axis=1)
            X_val['prompt'] = X_val.apply(standard_narrative_prompt, axis=1)
            X_test['prompt'] = X_test.apply(standard_narrative_prompt, axis=1)
            
            logger.info("\nProcessing train set...")
            X_train_encoded = get_embeddings(X_train['prompt'].tolist(), args.model_name, 
                                            args.batch_size, args.max_length, device, args.cache_dir)
            np.save(f"{args.embedding_dir}X_train_{safe_model_name}_{args.csv_to_use}_embeddings.npy", 
                   X_train_encoded)

            logger.info("\nProcessing validation set...")
            X_val_encoded = get_embeddings(X_val['prompt'].tolist(), args.model_name, 
                                          args.batch_size, args.max_length, device, args.cache_dir)
            np.save(f"{args.embedding_dir}X_val_{safe_model_name}_{args.csv_to_use}_embeddings.npy", 
                   X_val_encoded)

            logger.info("\nProcessing test set...")
            X_test_encoded = get_embeddings(X_test['prompt'].tolist(), args.model_name, 
                                           args.batch_size, args.max_length, device, args.cache_dir)
            np.save(f"{args.embedding_dir}X_test_{safe_model_name}_{args.csv_to_use}_embeddings.npy", 
                   X_test_encoded)
        else:
            X_train_encoded = np.load(f"{args.embedding_dir}X_train_{safe_model_name}_{args.csv_to_use}_embeddings.npy")
            X_val_encoded = np.load(f"{args.embedding_dir}X_val_{safe_model_name}_{args.csv_to_use}_embeddings.npy")
            X_test_encoded = np.load(f"{args.embedding_dir}X_test_{safe_model_name}_{args.csv_to_use}_embeddings.npy")
        
        input_size = X_train_encoded.shape[2]  # Embedding dimension

    logger.info(f"Encoded shapes - Train: {X_train_encoded.shape}, Val: {X_val_encoded.shape}, Test: {X_test_encoded.shape}")

    # Create Datasets and DataLoaders -----------------------------------------------
    train_dataset = SequenceDataset(X_train_encoded, y_train)
    val_dataset = SequenceDataset(X_val_encoded, y_val)
    test_dataset = SequenceDataset(X_test_encoded, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize Model --------------------------------------------------------------
    logger.info("Initializing LSTM model...")
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    logger.info(f"Model architecture:\n{model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=5, verbose=True)

    # Training Loop -----------------------------------------------------------------
    logger.info("\nStarting training...")
    best_val_f1 = 0
    patience_counter = 0
    train_losses = []
    val_metrics_history = []

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        val_metrics_history.append(val_metrics)
        
        logger.info(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Val Metrics - Acc: {val_metrics['accuracy']:.4f}, "
                   f"Prec: {val_metrics['precision']:.4f}, "
                   f"Rec: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"AUC: {val_metrics['auc']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])
        
        # Early stopping and model saving
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'args': args
            }, f"{args.output_dir}best_lstm_model_{args.csv_to_use}.pt")
            logger.info(f"✓ New best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                logger.info("\nEarly stopping triggered!")
                break

    # Load best model for final evaluation ------------------------------------------
    logger.info("\nLoading best model for final evaluation...")
    checkpoint = torch.load(f"{args.output_dir}best_lstm_model_{args.csv_to_use}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final Test Evaluation ---------------------------------------------------------
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*60)
    
    test_metrics, y_test_pred, y_test_probs, y_test_true = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"\nTest Loss:      {test_metrics['loss']:.4f}")
    logger.info(f"Accuracy:       {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision:      {test_metrics['precision']:.4f}")
    logger.info(f"Recall:         {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score:       {test_metrics['f1']:.4f}")
    logger.info(f"AUC:            {test_metrics['auc']:.4f}")

    logger.info("\n=== CLASSIFICATION REPORT ===")
    logger.info(classification_report(y_test_true, y_test_pred, target_names=['Not Ordered', 'Ordered']))

    logger.info("\n=== CONFUSION MATRIX ===")
    logger.info(confusion_matrix(y_test_true, y_test_pred))

    # Plotting Training History -----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Loss
    axes[0, 0].plot(epochs_range, train_losses, label='Train Loss')
    axes[0, 0].plot(epochs_range, [m['loss'] for m in val_metrics_history], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs_range, [m['accuracy'] for m in val_metrics_history])
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 2].plot(epochs_range, [m['f1'] for m in val_metrics_history])
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('Validation F1 Score')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs_range, [m['precision'] for m in val_metrics_history])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Validation Precision')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs_range, [m['recall'] for m in val_metrics_history])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Validation Recall')
    axes[1, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 2].plot(epochs_range, [m['auc'] for m in val_metrics_history])
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('AUC')
    axes[1, 2].set_title('Validation AUC')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}lstm_training_history_{args.csv_to_use}.png", dpi=300)
    logger.info(f"\nTraining history plot saved to {args.output_dir}lstm_training_history_{args.csv_to_use}.png")

    logger.info("\n✓ Training complete!")