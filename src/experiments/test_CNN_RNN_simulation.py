#!/usr/bin/env python
"""
Complete script for training baseline models (CNN/RNN) on the CSV data
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, f1_score
import time
import os
from pathlib import Path

# ================================================
# DATA LOADING FUNCTIONS
# ================================================
def load_csv_data(X_path, y_path):
    """
    Load the data from the CSV files
    """
    print(f"Loading data from:\n  X: {X_path}\n  y: {y_path}")
    
    # Load the CSVs
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)
    
    # Print data info
    print(f"  Shape X: {X_df.shape}, Shape y: {y_df.shape}")
    print(f"  Columns X: {X_df.columns.tolist()}")
    print(f"  Columns y: {y_df.columns.tolist()}")
    
    # Extract the sequences and the outcomes
    # ADAPT THESE NAMES TO YOUR COLUMNS!
    if 'Sequences' in X_df.columns:
        sequences = X_df['Sequences'].tolist()
    elif 'sequence' in X_df.columns:
        sequences = X_df['sequence'].tolist()
    else:
        # If the sequence is the only or the first column
        sequences = X_df.iloc[:, 0].tolist()
    
    if 'Outcome' in y_df.columns:
        labels = y_df['Outcome'].values
    elif 'outcome' in y_df.columns:
        labels = y_df['outcome'].values
    elif 'label' in y_df.columns:
        labels = y_df['label'].values
    else:
        # If the outcome is the only or the first column
        labels = y_df.iloc[:, 0].values
    
    print(f"  Loaded {len(sequences)} sequences")
    print(f"  Label distribution: 0={sum(labels==0)}, 1={sum(labels==1)}")
    print(f"  Example sequence: {sequences[0][:100]}...")  # First 100 chars
    
    return sequences, labels


# ================================================
# DATASET CLASS
# ================================================
class SequenceDataset(Dataset):
    """Dataset for sequences from CSV"""
    def __init__(self, sequences, labels, vocab_to_idx=None, max_len=10, separator='\x1f'):
        """
        Args:
            sequences: list of strings (sequences)
            labels: array of labels
            vocab_to_idx: vocabulary dictionary (if None, it is built)
            max_len: maximum sequence length
            separator: separator character in the sequences (default \x1f)
        """
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.separator = separator
        
        # Inspect the sequence format
        sample_seq = sequences[0] if sequences else ""
        
        # Auto-detect the separator if not specified
        if separator not in sample_seq:
            # Try common separators
            for sep in [',', ' ', '\t', '|', '-', '_']:
                if sep in sample_seq:
                    self.separator = sep
                    print(f"  Auto-detected separator: '{sep}'")
                    break
        
        # Build the vocabulary if not provided
        if vocab_to_idx is None:
            unique_tokens = set()
            for seq in sequences:
                # Tokenize the sequence
                if self.separator:
                    tokens = seq.split(self.separator)
                else:
                    tokens = list(seq)  # Character by character
                unique_tokens.update(tokens)
            
            # Build the mapping
            self.vocab_to_idx = {'<PAD>': 0, '<UNK>': 1}
            for idx, token in enumerate(sorted(unique_tokens), start=2):
                self.vocab_to_idx[token] = idx
            
            print(f"  Created vocabulary with {len(self.vocab_to_idx)} tokens")
            print(f"  Sample tokens: {list(self.vocab_to_idx.keys())[:10]}")
        else:
            self.vocab_to_idx = vocab_to_idx
            
        self.vocab_size = len(self.vocab_to_idx)
        self.idx_to_vocab = {idx: token for token, idx in self.vocab_to_idx.items()}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Tokenize the sequence
        seq = self.sequences[idx]
        if self.separator:
            tokens = seq.split(self.separator)
        else:
            tokens = list(seq)
        
        # Truncate if necessary
        tokens = tokens[:self.max_len]
        
        # Convert to indices
        indices = [self.vocab_to_idx.get(token, 1) for token in tokens]  # 1 = UNK
        
        # Padding
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))  # 0 = PAD
            
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


# ================================================
# MODELS (same as before)
# ================================================
class CNN1DClassifier(nn.Module):
    """1D CNN for classification"""
    def __init__(self, vocab_size, embed_dim=64, num_filters=100, 
                 filter_sizes=[3,4,5], dropout=0.5, num_classes=1):
        super(CNN1DClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multiple convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return torch.sigmoid(x).squeeze()


class GRUClassifier(nn.Module):
    """GRU for classification"""
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, 
                 num_layers=2, dropout=0.5, bidirectional=True, num_classes=1):
        super(GRUClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0,
                         bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(dropout)
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * factor, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.gru(x)
        
        # Mean pooling
        x = torch.mean(output, dim=1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return torch.sigmoid(x).squeeze()


# ================================================
# TRAINING FUNCTIONS
# ================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(dataloader), all_preds, all_labels


def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / len(dataloader), all_preds, all_labels


def train_model(model, train_loader, val_loader, args, device):
    """Full training loop"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=3, factor=0.5
        )
    
    criterion = nn.BCELoss()
    
    best_val_loss = np.inf
    best_model_state = None
    patience_counter = 0
    
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_auc = roc_auc_score(train_labels, train_preds)
        
        # Validate
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, np.round(val_preds))
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train: Loss={train_loss:.4f}, AUC={train_auc:.4f} - "
              f"Val: Loss={val_loss:.4f}, AUC={val_auc:.4f}, Acc={val_acc:.4f}")
        
        # Learning rate scheduling
        if args.scheduler:
            scheduler.step(val_auc)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f" New best model! Val AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss


# ================================================
# MAIN FUNCTION
# ================================================
def main(args):
    """Main training script"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading Data...")
    print("="*60)
    
    X_train, y_train = load_csv_data(args.X_train_csv, args.y_train_csv)
    X_val, y_val = load_csv_data(args.X_val_csv, args.y_val_csv)
    X_test, y_test = load_csv_data(args.X_test_csv, args.y_test_csv)
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating Datasets...")
    print("="*60)
    
    # Compute max_len from the sequences
    if args.max_seq_len is None:
        # Compute automatically
        seq_lens = []
        for seq in X_train[:1000]:  # Sample the first 1000
            tokens = seq.split(args.separator) if args.separator else list(seq)
            seq_lens.append(len(tokens))
        args.max_seq_len = int(np.percentile(seq_lens, 95))
        print(f"Auto-detected max_seq_len: {args.max_seq_len}")
    
    train_dataset = SequenceDataset(X_train, y_train, max_len=args.max_seq_len, 
                                    separator=args.separator)
    val_dataset = SequenceDataset(X_val, y_val, 
                                  vocab_to_idx=train_dataset.vocab_to_idx,
                                  max_len=args.max_seq_len, separator=args.separator)
    test_dataset = SequenceDataset(X_test, y_test, 
                                   vocab_to_idx=train_dataset.vocab_to_idx,
                                   max_len=args.max_seq_len, separator=args.separator)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    # Initialize model
    print("\n" + "="*60)
    print(f"Initializing Model: {args.model}")
    print("="*60)
    
    vocab_size = train_dataset.vocab_size
    
    if args.model.lower() == 'cnn':
        model = CNN1DClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_filters=args.num_filters,
            filter_sizes=[int(x) for x in args.filter_sizes.split(',')],
            dropout=args.dropout
        )
    elif args.model.lower() == 'gru':
        model = GRUClassifier(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    model, best_val_auc = train_model(model, train_loader, val_loader, args, device)
    
    # Test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation...")
    print("="*60)
    
    criterion = nn.BCELoss()
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_auc = roc_auc_score(test_labels, test_preds)
    test_acc = accuracy_score(test_labels, np.round(test_preds))
    test_f1 = f1_score(test_labels, np.array(test_preds) >= 0.5)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, np.round(test_preds)))
    
    # Save model if requested
    if args.save_model:
        model_path = f"{args.model}_{args.experiment_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_to_idx': train_dataset.vocab_to_idx,
            'args': args,
            'best_val_auc': best_val_auc,
            'test_auc': test_auc,
            'test_f1': test_f1
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    return test_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN/RNN baseline models')
    
    # Data paths
    parser.add_argument("--X_train_csv", type=str, 
                       default="./data/simulation/X_train_5.csv",
                       help="Path to training features CSV")
    parser.add_argument("--y_train_csv", type=str, 
                       default="./data/simulation/y_train_5.csv",
                       help="Path to training labels CSV")
    parser.add_argument("--X_val_csv", type=str, 
                       default="./data/simulation/X_val_5.csv",
                       help="Path to validation features CSV")
    parser.add_argument("--y_val_csv", type=str, 
                       default="./data/simulation/y_val_5.csv",
                       help="Path to validation labels CSV")
    parser.add_argument("--X_test_csv", type=str, 
                       default="./data/simulation/X_test_5.csv",
                       help="Path to test features CSV")
    parser.add_argument("--y_test_csv", type=str, 
                       default="./data/simulation/y_test_5.csv",
                       help="Path to test labels CSV")
    
    # Model selection
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "gru"],
                       help="Model type to use")
    parser.add_argument("--experiment_name", type=str, default="baseline",
                       help="Name for experiment")
    
    # Data processing
    parser.add_argument("--separator", type=str, default="\x1f",
                       help="Token separator in sequences")
    parser.add_argument("--max_seq_len", type=int, default=None,
                       help="Max sequence length (None = auto-detect)")
    
    # Model hyperparameters
    parser.add_argument("--embed_dim", type=int, default=64,
                       help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.3,
                       help="Dropout rate")
    
    # CNN specific
    parser.add_argument("--num_filters", type=int, default=100,
                       help="Number of CNN filters")
    parser.add_argument("--filter_sizes", type=str, default="3,4,5,6",
                       help="CNN filter sizes (comma-separated)")
    
    # GRU specific
    parser.add_argument("--hidden_dim", type=int, default=128,
                       help="GRU hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of GRU layers")
    parser.add_argument("--bidirectional", action='store_true',
                       help="Use bidirectional GRU")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=60,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--early_stopping", action='store_true',
                       help="Use early stopping")
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--scheduler", action='store_true',
                       help="Use learning rate scheduler")
    
    # Other
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of dataloader workers")
    parser.add_argument("--cpu", action='store_true',
                       help="Force CPU usage")
    parser.add_argument("--save_model", action='store_true',
                       help="Save trained model")
    
    args = parser.parse_args()
    
    # Run training
    test_auc = main(args)
    
    print(f"\n{'='*60}")
    print(f"Training completed! Final test AUC: {test_auc:.4f}")
    print(f"{'='*60}")