import os
import numpy as np
import gc
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from typing import List, Dict, Callable, Any
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

from src.data.prompts import (
    no_narrative_prompt, naive_narrative_prompt, compact_narrative_prompt,
    full_narrative, full_narrative_no_time, full_narrative_no_time_rnd,
    compact_no_time_prompt, compact_no_time_prompt_rnd, compact_narrative_humanstyle_prompt,
    semi_full_narrative, reversed_naive_narrative_prompt, last_info_prompt, full_narrative_num2words
)
from src.experiments.Finetuning_DC import load_tokenizer, load_model

class BootstrapModelEvaluator:
    """
    Bootstrap evaluation for already fine-tuned models.
    Framework-agnostic - works with any model that can predict on data.
    """
    
    def __init__(self, n_bootstrap: int = 100, confidence_level: float = 0.95, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_evaluate(
        self, 
        model_predict_fn: Callable, 
        test_data: Any,
        test_labels: np.ndarray,
        metrics: List[str] = ['f1_weighted', 'f1_macro', 'f1_micro','f1', 'precision_weighted', 'recall_weighted', 'AUC'],
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap evaluate a model on test data.
        
        Args:
            model_predict_fn: Function that takes test_data subset and returns predictions
            test_data: Test dataset (can be any format your model accepts)
            test_labels: True labels as numpy array
            metrics: List of metrics to compute
            verbose: Show progress bar
            
        Returns:
            Dictionary with bootstrap statistics for each metric
        """
        n_samples = len(test_labels)
        
        # Store bootstrap results for each metric
        bootstrap_results = {metric: [] for metric in metrics}
        
        # Original evaluation (full dataset)
        original_predictions = model_predict_fn(test_data)
        original_scores = self._compute_metrics(test_labels, original_predictions, metrics)
        
        # Bootstrap sampling
        iterator = tqdm(range(self.n_bootstrap), desc="Bootstrap sampling") if verbose else range(self.n_bootstrap)
        
        for i in iterator:
            # Sample with replacement
            bootstrap_indices = np.random.randint(0, n_samples, size=n_samples, dtype=int)
            
            # Get bootstrap sample
            bootstrap_data = self._subset_data(test_data, bootstrap_indices)
            bootstrap_labels = np.array(test_labels)[bootstrap_indices].tolist()
            
            # Get predictions for bootstrap sample
            bootstrap_predictions = model_predict_fn(bootstrap_data)
            
            # Compute metrics
            bootstrap_scores = self._compute_metrics(bootstrap_labels, bootstrap_predictions, metrics)
            
            # Store results
            for metric in metrics:
                bootstrap_results[metric].append(bootstrap_scores[metric])
        
        # Compute confidence intervals and statistics
        final_results = {}
        for metric in metrics:
            bootstrap_scores = np.array(bootstrap_results[metric])
            
            # Confidence interval
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
            
            final_results[metric] = {
                'original_score': original_scores[metric],
                'bootstrap_mean': np.mean(bootstrap_scores),
                'bootstrap_std': np.std(bootstrap_scores, ddof=1),
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'ci_width': ci_upper - ci_lower,
                'bootstrap_samples': bootstrap_scores.tolist()
            }
        
        return final_results
    
    def _subset_data(self, data: Any, indices: np.ndarray) -> Any:
        """Subset data based on indices. Override for custom data formats."""
        if isinstance(data, np.ndarray):
            return data[indices]
        elif isinstance(data, list):
            return [data[i] for i in indices]
        elif hasattr(data, '__getitem__'):  # Pandas, torch datasets, etc.
            return data[indices]
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: List[str]) -> Dict[str, float]:
        """Compute specified metrics."""
        results = {}
        
        for metric in metrics:
            if metric == 'f1_weighted':
                results[metric] = f1_score(y_true, y_pred, average='weighted')
            elif metric == 'f1_macro':
                results[metric] = f1_score(y_true, y_pred, average='macro')
            elif metric == 'f1_micro':
                results[metric] = f1_score(y_true, y_pred, average='micro')
            elif metric == "f1":
                results[metric] = f1_score(y_true, y_pred, average='binary') # default
            elif metric == 'precision_weighted':
                p, _, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                results[metric] = p
            elif metric == 'recall_weighted':
                _, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                results[metric] = r
            elif metric == 'accuracy':
                results[metric] = np.mean(y_true == y_pred)
            elif metric == 'AUC':
                results[metric] = roc_auc_score(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        
        return results

# Hugging Face Transformers specific implementation
class HuggingFaceBootstrapEvaluator(BootstrapModelEvaluator):
    """Bootstrap evaluator specifically for Hugging Face models."""
    
    def __init__(self, model, tokenizer, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def evaluate_model(
        self, 
        test_texts: List[str], 
        test_labels: np.ndarray,
        batch_size: int = 24, # TODO change
        max_length: int = 2048, # TODO change
        metrics: List[str] = ['f1_weighted', 'f1_macro', 'f1', 'precision_weighted', 'recall_weighted', 'AUC']
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap evaluate a Hugging Face model.
        
        Args:
            test_texts: List of input texts
            test_labels: True labels
            batch_size: Batch size for inference
            max_length: Max sequence length
            metrics: Metrics to compute
            
        Returns:
            Bootstrap evaluation results
        """
        
        def predict_fn(text_subset):
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
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Predict
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                    predictions.extend(batch_predictions)
            
            return np.array(predictions)
        
        return self.bootstrap_evaluate(predict_fn, test_texts, test_labels, metrics)

# PyTorch Dataset specific implementation
class PyTorchBootstrapEvaluator(BootstrapModelEvaluator):
    """Bootstrap evaluator for PyTorch models with DataLoader."""
    
    def __init__(self, model, device='cuda', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate_model(
        self,
        dataset,  # PyTorch Dataset
        test_labels: np.ndarray,
        batch_size: int = 24,
        num_workers: int = 4,
        metrics: List[str] = ['f1_weighted', 'f1_macro']
    ) -> Dict[str, Dict[str, float]]:
        """
        Bootstrap evaluate a PyTorch model.
        
        Args:
            dataset: PyTorch Dataset containing test data
            test_labels: True labels
            batch_size: Batch size for DataLoader
            num_workers: Number of DataLoader workers
            metrics: Metrics to compute
            
        Returns:
            Bootstrap evaluation results
        """
        from torch.utils.data import DataLoader, Subset
        
        def predict_fn(indices):
            """Prediction function for bootstrap sampling."""
            # Create subset dataset
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            
            subset_dataset = Subset(dataset, indices)
            dataloader = DataLoader(
                subset_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=num_workers
            )
            
            predictions = []
            
            with torch.no_grad():
                for batch in dataloader:
                    # Assume batch is (inputs, labels) or just inputs
                    if isinstance(batch, (list, tuple)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)
                    
                    outputs = self.model(inputs)
                    batch_predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                    predictions.extend(batch_predictions)
            
            return np.array(predictions)
        
        # Create indices for the dataset
        dataset_indices = np.arange(len(dataset))
        
        return self.bootstrap_evaluate(predict_fn, dataset_indices, test_labels, metrics)

# Utility functions for results analysis
def print_bootstrap_results(results: Dict[str, Dict[str, float]], model_name: str = "Model"):
    """Pretty print bootstrap evaluation results."""
    print(f"\nBootstrap Evaluation Results for {model_name}")
    print("=" * 60)
    
    for metric, stats in results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Original Score:    {stats['original_score']:.4f}")
        print(f"  Bootstrap Mean:    {stats['bootstrap_mean']:.4f}")
        print(f"  Bootstrap Std:     {stats['bootstrap_std']:.4f}")
        print(f"  95% CI:           [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
        print(f"  CI Width:          {stats['ci_width']:.4f}")

def compare_models_bootstrap(results_dict: Dict[str, Dict], metric: str = 'f1_weighted'):
    """Compare multiple models using bootstrap results."""
    comparison_data = []
    
    for model_name, results in results_dict.items():
        if metric in results:
            stats = results[metric]
            comparison_data.append({
                'model': model_name,
                'score': stats['bootstrap_mean'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'ci_width': stats['ci_width'],
                'std': stats['bootstrap_std']
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    print(f"\nModel Comparison - {metric}")
    print("-" * 50)
    print(df.round(4))
    
    return df

def save_bootstrap_results(results: Dict, filepath: str):
    """Save bootstrap results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for metric, stats in results.items():
        serializable_results[metric] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in stats.items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {filepath}")

# Example usage
def example_usage():
    """Example of how to use the bootstrap evaluator."""
    
    # Example 1: Simple numpy arrays
    print("Example 1: Simple Arrays")
    print("-" * 30)
    
    # Simulate some test data and predictions
    np.random.seed(42)
    n_test = 500
    test_labels = np.random.choice([0, 1, 2], n_test, p=[0.3, 0.4, 0.3])
    
    # Simulate a model that's 85% accurate
    test_predictions = test_labels.copy()
    noise_mask = np.random.random(n_test) < 0.15
    test_predictions[noise_mask] = np.random.choice([0, 1, 2], np.sum(noise_mask))
    
    # Simple predict function
    def simple_predict_fn(indices):
        return test_predictions[indices]
    
    # Bootstrap evaluate
    evaluator = BootstrapModelEvaluator(n_bootstrap=100)
    results = evaluator.bootstrap_evaluate(
        simple_predict_fn, 
        np.arange(n_test), 
        test_labels,
        metrics=['f1_weighted', 'f1_macro', 'accuracy']
    )
    
    print_bootstrap_results(results, "Example Model")
    
    return results

if __name__ == "__main__":
    # Data preparation
    gc.collect()
    torch.cuda.empty_cache()
    print("Data Preparation")

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
    prompt_type = "no"  # To change
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
    checkpoint_path = "best_model_meta-llama_Llama-3.1-8B_4550_landmark4_no_4_2048_last_visit_all_landmarks_False_20250906_190031_.pt" # To change 
    # END TO CHANGE PARAMETERS
    # ======================= #

    if prompt_type == "naive":
        narrative_prompt = naive_narrative_prompt
    elif prompt_type == "compact":
        narrative_prompt = compact_narrative_prompt
    elif prompt_type == "compact_no_time":
        narrative_prompt = compact_no_time_prompt
    elif prompt_type == "compact_no_time_rnd":
        narrative_prompt = compact_no_time_prompt_rnd
    elif prompt_type == "no":
        narrative_prompt = no_narrative_prompt
    elif prompt_type == "full":
        narrative_prompt = full_narrative
    elif prompt_type == "full_no_time":
        narrative_prompt = full_narrative_no_time
    elif prompt_type == "full_no_time_rnd":
        narrative_prompt = full_narrative_no_time_rnd
    elif prompt_type == "compact_narrative":
        narrative_prompt = compact_narrative_humanstyle_prompt
    elif prompt_type == "semi_full_narrative":
        narrative_prompt = semi_full_narrative
    elif prompt_type == "reversed_naive_narrative_prompt":
        narrative_prompt = reversed_naive_narrative_prompt
    elif prompt_type == "last_info_prompt":
        narrative_prompt = last_info_prompt
    elif prompt_type == "full_narrative_num2words":
       narrative_prompt = full_narrative_num2words
    else:
        raise ValueError("Invalid prompt type!")

    full_test_df = pd.read_csv(path_test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    visit_counts = full_test_df['subject_id'].value_counts()
    selected_patients = visit_counts[visit_counts == max_visits].index # >= instead of == to include patients with more than max_visits
    dataset_selected = full_test_df[full_test_df['subject_id'].isin(selected_patients)].copy()
    test_df_selected = dataset_selected

    test_df = test_df_selected[test_df_selected['landmark_visit'] == max_visits].copy()
    test_texts = test_df.apply(narrative_prompt, axis=1).tolist()
    test_labels = test_df['death_in_90days'].tolist()


    print("Bootstrap Model Evaluation Framework")
    print("=" * 45)
    
    # Load your fine-tuned model
    tokenizer = load_tokenizer(model_name, model_type, hf_token, cache_dir)
    model = load_model(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft, use_quantization)
    
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token; adding [PAD] manually.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(f"{cache_dir}/best/{checkpoint_path}"))

    # test_dataset = ClinicalDataset(test_texts, test_labels, tokenizer, max_length=max_length)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Bootstrap evaluate
    evaluator = HuggingFaceBootstrapEvaluator(model, tokenizer)
    results = evaluator.evaluate_model(test_texts, test_labels)
    print_bootstrap_results(results, "Your Prompt")
    
    print(f"\nKey Insights:")
    print(f"• CI width tells you evaluation uncertainty")
    print(f"• Narrow CI (< 0.01) suggests evaluation variance is small")
    print(f"• Wide CI (> 0.03) suggests you need more test data or training variance dominates") 
    print(f"• Use this to determine if differences between prompts are meaningful")