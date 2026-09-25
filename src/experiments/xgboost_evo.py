import os
import logging
import random
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import csr_matrix
from scipy.sparse import hstack

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="scikit-learn Gradient Boosting training script.")
    parser.add_argument("--input_csv", type=str, default="landmark_df.csv", help="Input CSV file path")
    parser.add_argument("--train_csv", type=str, default="./data/landmark_evo_train.csv",
                        help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, default="./data/landmark_evo_vali.csv",
                        help="Path to the validation CSV file")
    parser.add_argument("--test_csv", type=str, default="./data/landmark_evo_test.csv",
                        help="Path to the test CSV file") # evo as well
    parser.add_argument("--prompt_type", type=str, choices=["naive", "compact"], default="compact",
                            help="Prompt type to use: 'naive' or 'compact'") 
    parser.add_argument("--max_visits", type=int, default=3, help="Max number of landmark visits")
    parser.add_argument("--random_state", type=int, default=9550, help="Random seed")
    parser.add_argument("--max_features", type=int, default=1000, help="Max features for vectorizer")
    parser.add_argument("--vectorizer_type", type=str, choices=["count", "tfidf"], default="count",
                    help="Vectorizer type to use: 'count' or 'tfidf'")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
    
    args = parser.parse_args()

    # Log arguemnts values
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args

def safe_vectorize(column_name, vectorizer, train_df, test_df):
    """
    Safely apply a CountVectorizer to two dataframe columns.
    If the content is empty or raises an error (e.g. empty vocabulary), return empty matrices.

    Args:
        column_name (str): name of the column to transform
        vectorizer (CountVectorizer): already-initialised instance
        train_df (pd.DataFrame): training dataframe
        test_df (pd.DataFrame): test dataframe

    Returns:
        tuple: (X_train_vec, X_test_vec), both scipy sparse matrices
    """
    train_col = train_df[column_name].fillna('').astype(str)
    test_col = test_df[column_name].fillna('').astype(str)

    # Check that there is at least one valid document
    if train_col.str.strip().replace('', np.nan).dropna().empty:
        return csr_matrix((len(train_df), 0)), csr_matrix((len(test_df), 0))

    try:
        X_train_vec = vectorizer.fit_transform(train_col)
        X_test_vec = vectorizer.transform(test_col)
    except ValueError:
        # e.g. vocabulary empty because of stopwords
        return csr_matrix((len(train_df), 0)), csr_matrix((len(test_df), 0))

    return X_train_vec, X_test_vec


def train_model(train_df, test_df, numeric_features, vectorizer, args):
    med_vectorizer = vectorizer(max_features=args.max_features)
    diag_vectorizer = vectorizer(max_features=args.max_features)
    proc_vectorizer = vectorizer(max_features=args.max_features)

    train_med_vec = med_vectorizer.fit_transform(train_df['med_text'])
    test_med_vec = med_vectorizer.transform(test_df['med_text'])

    train_diag_vec = diag_vectorizer.fit_transform(train_df['diag_text'])
    test_diag_vec = diag_vectorizer.transform(test_df['diag_text'])

    train_proc_vec = proc_vectorizer.fit_transform(train_df['proc_text'])
    test_proc_vec = proc_vectorizer.transform(test_df['proc_text'])

    if args.prompt_type == "compact":
        # We need to create additional vectorizers for the compact prompt
        admission_vectorizer = vectorizer(max_features=args.max_features)
        dose_vectorizer = vectorizer(max_features=args.max_features)
        new_med_vectorizer = vectorizer(max_features=args.max_features)
        new_diag_vectorizer = vectorizer(max_features=args.max_features)
        new_proc_vectorizer = vectorizer(max_features=args.max_features)
        new_dose_vectorizer = vectorizer(max_features=args.max_features)
        no_more_med_vectorizer = vectorizer(max_features=args.max_features)
        no_more_diag_vectorizer = vectorizer(max_features=args.max_features)
        no_more_proc_vectorizer = vectorizer(max_features=args.max_features)
        no_more_dose_vectorizer = vectorizer(max_features=args.max_features)

        train_admission_vec, test_admission_vec = safe_vectorize(
            'admission_category', admission_vectorizer, train_df, test_df)
        train_dose_vec, test_dose_vec = safe_vectorize(
            'dose_text', dose_vectorizer, train_df, test_df)
        train_new_med_vec, test_new_med_vec = safe_vectorize(
            'new_medications', new_med_vectorizer, train_df, test_df)
        train_new_diag_vec, test_new_diag_vec = safe_vectorize(
            'new_diagnoses', new_diag_vectorizer, train_df, test_df)
        train_new_proc_vec, test_new_proc_vec = safe_vectorize(
            'new_procedures', new_proc_vectorizer, train_df, test_df)
        train_new_dose_vec, test_new_dose_vec = safe_vectorize(
            'new_dose', new_dose_vectorizer, train_df, test_df)
        train_no_more_med_vec, test_no_more_med_vec = safe_vectorize(
            'no_more_medications', no_more_med_vectorizer, train_df, test_df)
        train_no_more_diag_vec, test_no_more_diag_vec = safe_vectorize(
            'no_more_diagnoses', no_more_diag_vectorizer, train_df, test_df)
        train_no_more_proc_vec, test_no_more_proc_vec = safe_vectorize(
            'no_more_procedures', no_more_proc_vectorizer, train_df, test_df)
        train_no_more_dose_vec, test_no_more_dose_vec = safe_vectorize(
            'no_more_dose', no_more_dose_vectorizer, train_df, test_df)

    X_train_numeric = train_df[numeric_features].values
    X_test_numeric = test_df[numeric_features].values

    if args.prompt_type == "compact":
        X_train = hstack([
            train_med_vec, train_diag_vec, train_proc_vec, train_admission_vec,
            train_dose_vec, train_new_med_vec, train_new_diag_vec,
            train_new_proc_vec, train_new_dose_vec, train_no_more_med_vec,
            train_no_more_diag_vec, train_no_more_proc_vec, train_no_more_dose_vec,
            X_train_numeric
        ])
        X_test = hstack([
            test_med_vec, test_diag_vec, test_proc_vec, test_admission_vec,
            test_dose_vec, test_new_med_vec, test_new_diag_vec,
            test_new_proc_vec, test_new_dose_vec, test_no_more_med_vec,
            test_no_more_diag_vec, test_no_more_proc_vec, test_no_more_dose_vec,
            X_test_numeric
        ])
    else:
        X_train = hstack([train_med_vec, train_diag_vec, train_proc_vec, X_train_numeric])
        X_test = hstack([test_med_vec, test_diag_vec, test_proc_vec, X_test_numeric])
    
    y_train = train_df['death_in_90days'].values
    y_test = test_df['death_in_90days'].values


    # Parameter definition for RandomizedSearchCV
    param_dist = {
        'max_iter': [100, 300, 500],
        'max_depth': [3, 5, 7, None],
        'learning_rate': [0.01, 0.05, 0.1],
        'min_samples_leaf': [10, 20, 30],
        'l2_regularization': [0, 0.1, 1.0],
        'max_leaf_nodes': [15, 31, 63]
    }

    search = RandomizedSearchCV(
        HistGradientBoostingClassifier(random_state=args.random_state),
        param_distributions=param_dist,
        scoring='roc_auc',
        n_iter=10,
        cv=5,
        verbose=2,
        random_state=args.random_state,
        n_jobs=-1
    )

    # Fitting (HistGradientBoostingClassifier does not accept sparse input!)
    search.fit(X_train.toarray(), y_train)

    best_model = search.best_estimator_
    logger.info(f"Best Parameters for Landmark: {search.best_params_}")

    best_model.fit(X_train.toarray(), y_train)

    y_pred_proba = best_model.predict_proba(X_test.toarray())[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_binary)
    mortality_rate = np.mean(y_test) * 100

    return auc, f1, len(test_df), mortality_rate

def set_seed(seed_value=5550):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = "9550"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed_value)
    np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    # torch.cuda.manual_seed(seed_value)
    # torch.cuda.manual_seed_all(seed_value)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()

    set_seed(args.random_state)

    # Let us choose vectorizer
    if args.vectorizer_type == "count":
        Vectorizer = CountVectorizer
    elif args.vectorizer_type == "tfidf":
        Vectorizer = TfidfVectorizer
    elif args.vectorizer_type == "medBERT":
        # MedBERT feature extraction using mean pooling instead of CLS token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        medbert_model_name = "emilyalsentzer/Bio_ClinicalBERT"
        tokenizer = AutoTokenizer.from_pretrained(medbert_model_name)
        model = AutoModel.from_pretrained(medbert_model_name).to(device)
        def medbert_vectorize(texts, batch_size=8, max_length=args.max_features):
            embeddings = []
            model.eval()
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    # Mean pooling instead of CLS token
                    mean_embedding = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(mean_embedding.squeeze().cpu().numpy())
            return np.vstack(embeddings)
        Vectorizer = medbert_vectorize
    else:
        raise ValueError(f"Vectorizer type '{args.vectorizer_type}' not supported.")

    logger.info(f"Using data required to create prompt: {args.prompt_type}")

    numeric_features = ['age_at_landmark', 'landmark_visit', 'num_total_visits', 'days_since_previous_visit', 'gender_numeric']
    
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

    train_df_selected, val_df_selected, test_df_selected = filtered_datasets

    results = []

    for landmark_visit in range(1, args.max_visits + 1): # TODO: to readjust to 1, now just for finishing the test
        logger.info(f"Processing Landmark Visit: {landmark_visit}")
        
        train_df = train_df_selected[train_df_selected['landmark_visit'] == landmark_visit].copy()
        val_df = val_df_selected[val_df_selected['landmark_visit'] == landmark_visit].copy()
        test_df = test_df_selected[test_df_selected['landmark_visit'] == landmark_visit].copy()

        # Combine train and validation sets since we are using validation for hyperparameter tuning 
        # and we are using randomized search which will handle the validation internally.
        train_df = pd.concat([train_df, val_df]).reset_index(drop=True)

        if train_df.empty or test_df.empty or len(np.unique(test_df['death_in_90days'])) < 2:
            logger.warning(f"Skipping Landmark {landmark_visit} due to insufficient data.")
            continue

        auc, f1, num_patients, mortality_rate = train_model(
            train_df, test_df, numeric_features, Vectorizer, args
        )

        results.append({
            'Landmark (Visit Number)': landmark_visit,
            'Number of Patients (Test)': num_patients,
            'Mortality (%)': f"{mortality_rate:.1f}%",
            'AUC (Test)': f"{auc:.4f}",
            'F1-Score': f"{f1:.4f}"
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(os.path.join(args.output_dir, "sklearn_gb_results.csv"), index=False)
    logger.info("Results saved to CSV.")

if __name__ == "__main__":
    main()