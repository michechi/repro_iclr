import os
import random
import numpy as np
import torch
import argparse
import logging

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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    parser.add_argument("--input_csv", type=str, default="./data/splitted/splitted/landmark_df_evo.csv",
                        help="Path to the input CSV file")

    parser.add_argument("--train_csv", type=str, default="./data/splitted/splitted/landmark_evo_train_dod_fxd.csv",
                        help="Path to the training CSV file")

    parser.add_argument("--val_csv", type=str, default="./data/splitted/splitted/landmark_evo_vali_dod_fxd.csv",
                        help="Path to the validation CSV file")

    parser.add_argument("--test_csv", type=str, default="./data/splitted/splitted/landmark_evo_test_dod_fxd.csv",
                        help="Path to the test CSV file") # evo as well 
    
    parser.add_argument("--when_counting_death", type=str, choices=["last_visit", "landmark"], default="last_visit",
                        help="When counting death: 'last_visit' (considering the last visit) or 'landmark' (considering the current landmark visit)")

    parser.add_argument("--prompt_type", type=str, choices=["naive", "compact", "compact_no_time", "compact_no_time_rnd", 
                                                            "no", "full", "full_no_time", "full_no_time_rnd", "compact_narrative",
                                                            "semi_full_narrative", "reversed_naive_narrative_prompt", "reversed_time_naive_narrative_prompt",
                                                            "last_info_prompt", "full_narrative_num2words", "temporal_causal_prompt"], default="compact",
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