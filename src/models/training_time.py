import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import wandb  # For logging (optional)

# We assume ClinicalLLMClassifier has already been defined by the previous code

class ClinicalDataset(Dataset):
    """
    Dataset for temporal clinical data
    """
    def __init__(self, patient_visits, labels, max_visits=10):
        self.patient_visits = patient_visits
        self.labels = labels
        self.max_visits = max_visits
    
    def __len__(self):
        return len(self.patient_visits)
    
    def __getitem__(self, idx):
        visits = self.patient_visits[idx]
        label = self.labels[idx]
        
        # Limit the number of visits if necessary
        if len(visits) > self.max_visits:
            visits = visits[:self.max_visits]
        
        return {
            'visits': visits,
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """
    Function to combine the batch samples
    """
    visits = [item['visits'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'visits': visits,
        'labels': labels
    }

class HyperParameters:
    """
    Class holding all hyperparameters
    """
    def __init__(self):
        # === ARCHITECTURE ===
        self.model_name = "microsoft/DialoGPT-medium"  # Replace with Llama/Mistral
        self.max_visits = 10
        self.max_sequence_length = 512  # Maximum length for each visit
        
        # === TRAINING ===
        self.batch_size = 8  # Start small because of memory constraints
        self.learning_rate = 2e-5  # Learning rate for the LLM (conservative)
        self.classification_lr = 1e-3  # Higher learning rate for the head
        self.num_epochs = 20
        self.warmup_steps = 100
        self.weight_decay = 0.01
        self.gradient_clip_norm = 1.0
        
        # === REGULARIZATION ===
        self.dropout_rate = 0.1
        self.label_smoothing = 0.1  # To avoid overconfidence
        
        # === LOSS ===
        self.class_weights = None  # Set it if the classes are imbalanced: [1.0, 2.0]
        self.focal_loss_alpha = 0.25  # For focal loss (optional)
        self.focal_loss_gamma = 2.0
        
        # === EARLY STOPPING ===
        self.patience = 5
        self.min_delta = 1e-4
        
        # === SCHEDULER ===
        self.scheduler_type = "reduce_on_plateau"  # "cosine", "linear", "reduce_on_plateau"
        self.scheduler_factor = 0.5
        self.scheduler_patience = 3
        
        # === VALIDATION ===
        self.validation_split = 0.2
        self.stratify = True
        
        # === LOGGING ===
        self.log_every = 10  # Log every N steps
        self.save_every_epoch = True
        self.use_wandb = False  # Set True to use Weights & Biases

class FocalLoss(nn.Module):
    """
    Focal Loss for handling imbalanced classes
    """
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class EarlyStopping:
    """
    Early stopping to avoid overfitting
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Compute evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_prob[:, 1]) if y_prob.shape[1] > 1 else roc_auc_score(y_true, y_prob)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

class ClinicalTrainer:
    """
    Main training class
    """
    def __init__(self, hyperparams: HyperParameters):
        self.hp = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize wandb if requested
        if self.hp.use_wandb:
            wandb.init(project="clinical-classification", config=vars(self.hp))
    
    def setup_model(self):
        """
        Initialize the model
        """
        self.model = ClinicalLLMClassifier(
            model_name=self.hp.model_name,
            max_visits=self.hp.max_visits
        ).to(self.device)
        
        print(f"Model loaded: {self.hp.model_name}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def setup_optimizer_scheduler(self):
        """
        Configure optimizer and scheduler with different learning rates
        """
        # Base LLM parameters (lower learning rate)
        llm_params = list(self.model.llm.parameters())
        
        # Classification head parameters (higher learning rate)
        classification_params = list(self.model.classification_head.parameters())
        
        # Optimizer with differentiated learning rates
        self.optimizer = AdamW([
            {'params': llm_params, 'lr': self.hp.learning_rate, 'weight_decay': self.hp.weight_decay},
            {'params': classification_params, 'lr': self.hp.classification_lr, 'weight_decay': self.hp.weight_decay}
        ])
        
        # Scheduler
        if self.hp.scheduler_type == "reduce_on_plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self.hp.scheduler_factor, 
                patience=self.hp.scheduler_patience,
                verbose=True
            )
        elif self.hp.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=self.hp.num_epochs,
                eta_min=1e-7
            )
    
    def setup_loss(self):
        """
        Configure the loss function
        """
        if self.hp.class_weights is not None:
            weights = torch.tensor(self.hp.class_weights, dtype=torch.float).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.hp.label_smoothing)
        else:
            # Use Focal Loss for imbalanced classes
            self.criterion = FocalLoss(
                alpha=self.hp.focal_loss_alpha,
                gamma=self.hp.focal_loss_gamma
            )
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            visits = batch['visits']
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(visits)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hp.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Logging
            if batch_idx % self.hp.log_every == 0:
                progress_bar.set_postfix({'loss': loss.item()})
                
                if self.hp.use_wandb:
                    wandb.log({
                        'train_loss_step': loss.item(),
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
        
        # Compute metrics for the whole epoch
        avg_loss = total_loss / len(dataloader)
        metrics = calculate_metrics(all_labels, all_predictions, np.array(all_probs))
        
        return avg_loss, metrics
    
    def validate_epoch(self, dataloader):
        """
        Validate for one epoch
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                visits = batch['visits']
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(visits)
                logits = outputs['logits']
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                probs = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        metrics = calculate_metrics(all_labels, all_predictions, np.array(all_probs))
        
        return avg_loss, metrics
    
    def train(self, patient_visits, labels):
        """
        Main training loop
        """
        # Splits
        if self.hp.stratify:
            train_visits, val_visits, train_labels, val_labels = train_test_split(
                patient_visits, labels, 
                test_size=self.hp.validation_split,
                stratify=labels,
                random_state=42
            )
        else:
            train_visits, val_visits, train_labels, val_labels = train_test_split(
                patient_visits, labels, 
                test_size=self.hp.validation_split,
                random_state=42
            )
        
        # Datasets
        train_dataset = ClinicalDataset(train_visits, train_labels, self.hp.max_visits)
        val_dataset = ClinicalDataset(val_visits, val_labels, self.hp.max_visits)
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.hp.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.hp.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Setup
        self.setup_model()
        self.setup_optimizer_scheduler()
        self.setup_loss()
        
        # Early stopping
        early_stopping = EarlyStopping(patience=self.hp.patience, min_delta=self.hp.min_delta)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
        print("\n=== STARTING TRAINING ===")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(self.hp.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.hp.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            if self.hp.scheduler_type == "reduce_on_plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Logging
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # History
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['train_f1'].append(train_metrics['f1'])
            history['val_f1'].append(val_metrics['f1'])
            
            # Wandb logging
            if self.hp.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics['auc']
                })
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if self.hp.save_every_epoch:
                self.save_checkpoint(epoch, val_loss, val_metrics)
        
        print("\n=== TRAINING COMPLETED ===")
        return history
    
    def save_checkpoint(self, epoch, val_loss, val_metrics):
        """
        Save a checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'hyperparameters': vars(self.hp)
        }
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pth')

# ===== USAGE EXAMPLE =====
def main():
    """
    Complete training example
    """
    # Configure hyperparameters
    hp = HyperParameters()
    
    # HYPERPARAMETERS TO EXPERIMENT WITH:
    hp.batch_size = 4  # Start small
    hp.learning_rate = 1e-5  # Conservative for the LLM
    hp.classification_lr = 5e-4  # Higher for the head
    hp.num_epochs = 15
    hp.max_visits = 8
    hp.dropout_rate = 0.2  # Increase if overfitting
    hp.label_smoothing = 0.1
    hp.use_wandb = False  # Set True for advanced logging
    
    # Example data (replace with your own)
    patient_visits = [
        ["Visit 1: high fever", "Visit 2: worsening", "Visit 3: hospitalization"],
        ["Visit 1: routine check-up", "Visit 2: all normal"],
        ["Visit 1: chest pain", "Visit 2: heart attack", "Visit 3: intensive care"],
        ["Visit 1: headache", "Visit 2: improvement"],
        # ... add your real data
    ]
    
    labels = [1, 0, 1, 0]  # 1 = death, 0 = survival
    
    # Training
    trainer = ClinicalTrainer(hp)
    history = trainer.train(patient_visits, labels)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_f1'], label='Train F1')
    plt.plot(history['val_f1'], label='Val F1')
    plt.title('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    main()