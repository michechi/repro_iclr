import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Optional, Tuple, List
import numpy as np

class TemporalClassificationHead(nn.Module):
    """
    Classification head that incorporates temporal information
    """
    def __init__(self, hidden_size: int, num_visits: int, num_classes: int = 2):
        super().__init__()
        
        # Parameters
        self.hidden_size = hidden_size
        self.num_visits = num_visits
        self.num_classes = num_classes
        
        # Layer that processes the representation of each visit
        self.visit_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # LSTM to capture temporal dependencies between visits
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size // 4,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism to weight the importance of the visits
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,  # bidirectional LSTM output
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, visit_representations: torch.Tensor, 
                visit_mask: Optional[torch.Tensor] = None):
        """
        Args:
            visit_representations: (batch_size, num_visits, hidden_size)
            visit_mask: (batch_size, num_visits) - True for valid visits
        """
        batch_size, num_visits, hidden_size = visit_representations.shape
        
        # 1. Process each visit individually
        visit_encoded = self.visit_encoder(visit_representations)
        # Shape: (batch_size, num_visits, hidden_size//2)
        
        # 2. Apply LSTM to capture temporal dependencies
        lstm_output, (hidden, cell) = self.temporal_lstm(visit_encoded)
        # Shape: (batch_size, num_visits, hidden_size//2)
        
        # 3. Apply attention to weight the visits
        if visit_mask is not None:
            # Convert the mask to the attention format (True = do not mask)
            attention_mask = ~visit_mask  # Invert because attention expects False for valid positions
        else:
            attention_mask = None
            
        attended_output, attention_weights = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=attention_mask
        )
        
        # 4. Aggregate the information (weighted mean excluding padding)
        if visit_mask is not None:
            # Mask out padding before averaging
            mask_expanded = visit_mask.unsqueeze(-1).float()
            attended_masked = attended_output * mask_expanded
            aggregated = attended_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            aggregated = attended_output.mean(dim=1)
        
        # 5. Final classification
        logits = self.classifier(aggregated)
        
        return logits, attention_weights


class ClinicalLLMClassifier(nn.Module):
    """
    Full model combining a base LLM with temporal classification
    """
    def __init__(self, model_name: str, max_visits: int = 10):
        super().__init__()
        
        # Load the base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModel.from_pretrained(model_name)
        
        # Make sure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Temporal classification head
        self.classification_head = TemporalClassificationHead(
            hidden_size=self.config.hidden_size,
            num_visits=max_visits,
            num_classes=2
        )
        
        # Special tokens delimiting the visits
        self.visit_start_token = "<VISIT_START>"
        self.visit_end_token = "<VISIT_END>"
        
        # Add the special tokens to the vocabulary
        special_tokens = [self.visit_start_token, self.visit_end_token]
        self.tokenizer.add_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
    
    def encode_visits(self, visit_texts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the visits of each patient
        
        Args:
            visit_texts: List of lists, where each inner list contains the visit texts of one patient
        
        Returns:
            visit_representations: (batch_size, max_visits, hidden_size)
            visit_mask: (batch_size, max_visits)
        """
        batch_size = len(visit_texts)
        max_visits = max(len(visits) for visits in visit_texts)
        hidden_size = self.config.hidden_size
        
        # Initialize tensors
        visit_representations = torch.zeros(batch_size, max_visits, hidden_size)
        visit_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool)
        
        for patient_idx, patient_visits in enumerate(visit_texts):
            for visit_idx, visit_text in enumerate(patient_visits):
                # Add special tokens
                formatted_text = f"{self.visit_start_token} {visit_text} {self.visit_end_token}"
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Get the representation from the LLM
                with torch.no_grad():
                    outputs = self.llm(**inputs)
                    # Use the mean hidden state as the visit representation
                    visit_repr = outputs.last_hidden_state.mean(dim=1)  # (1, hidden_size)
                    
                visit_representations[patient_idx, visit_idx] = visit_repr.squeeze(0)
                visit_mask[patient_idx, visit_idx] = True
        
        return visit_representations, visit_mask
    
    def forward(self, visit_texts: List[List[str]]):
        """
        Full forward pass
        """
        # 1. Encode the visits
        visit_representations, visit_mask = self.encode_visits(visit_texts)
        
        # 2. Temporal classification
        logits, attention_weights = self.classification_head(
            visit_representations, visit_mask
        )
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'visit_mask': visit_mask
        }


# Usage example
def example_usage():
    """
    Practical example of how to use the model
    """
    
    # Initialize the model
    model = ClinicalLLMClassifier("microsoft/DialoGPT-medium", max_visits=5)
    
    # Example data
    patient_data = [
        [  # Patient 1
            "Visit 1: Patient presents with high fever, persistent cough",
            "Visit 2: Symptoms worsened, breathing difficulties",
            "Visit 3: Admitted to intensive care"
        ],
        [  # Patient 2
            "Visit 1: Routine check-up, parameters within normal range",
            "Visit 2: Slight increase in blood pressure",
            "Visit 3: Blood pressure stabilized with therapy",
            "Visit 4: Patient in good general condition"
        ]
    ]
    
    # Labels (0 = survives, 1 = dies)
    labels = torch.tensor([1, 0])  # Patient 1 dies, Patient 2 survives
    
    # Forward pass
    model.train()
    outputs = model(patient_data)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs['logits'], labels)
    
    print(f"Logits: {outputs['logits']}")
    print(f"Predictions: {torch.argmax(outputs['logits'], dim=1)}")
    print(f"Loss: {loss.item()}")
    
    # Show attention weights
    attention = outputs['attention_weights'][0]  # First patient
    print(f"Attention weights shape: {attention.shape}")
    
    return model, outputs, loss

if __name__ == "__main__":
    model, outputs, loss = example_usage()