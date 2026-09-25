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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import csr_matrix, hstack

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="scikit-learn Gradient Boosting training script.")
    parser.add_argument("--input_csv", type=str, default="landmark_df.csv", help="Input CSV file path")
    parser.add_argument("--max_visits", type=int, default=3, help="Max number of landmark visits")
    parser.add_argument("--random_state", type=int, default=9550, help="Random seed")
    parser.add_argument("--max_features", type=int, default=1000, help="Max features for CountVectorizer")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
    parser.add_argument("--vectorizer_type", type=str, choices=["count", "tfidf", "medbert"], default="count",
                        help="Vectorizer type to use: 'count', 'tfidf', 'medbert'")
    parser.add_argument("--train_csv", type=str, default="./data/landmark_evo_train.csv",
                        help="Path to the training CSV file")
    parser.add_argument("--val_csv", type=str, default="./data/landmark_evo_vali.csv",
                        help="Path to the validation CSV file")
    parser.add_argument("--test_csv", type=str, default="./data/landmark_evo_test.csv",
                        help="Path to the test CSV file")
    parser.add_argument("--prompt_type", type=str, choices=["naive", "compact"], default="compact",
                        help="Prompt type to use: 'naive' or 'compact'")

    args = parser.parse_args()

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args

class MedBERTVectorizer:
    def __init__(self, max_features=512, model_name="emilyalsentzer/Bio_ClinicalBERT", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()
        self.max_features = max_features

    def fit_transform(self, texts):
        return self._embed(texts.to_list())

    def transform(self, texts):
        return self._embed(texts.to_list())

    def _embed(self, texts, batch_size=8):
        embeddings = []
        texts = [" " if not t.strip() else t for t in texts]
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=self.max_features, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                mean_embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(mean_embedding.cpu().numpy())
        return np.vstack(embeddings)

def safe_vectorize(column_name, vectorizer, train_df, test_df):
    train_col = train_df[column_name].fillna('').astype(str)
    test_col = test_df[column_name].fillna('').astype(str)

    if train_col.str.strip().replace('', np.nan).dropna().empty:
        return csr_matrix((len(train_df), 0)), csr_matrix((len(test_df), 0))

    try:
        X_train_vec = vectorizer.fit_transform(train_col)
        X_test_vec = vectorizer.transform(test_col)
    except ValueError:
        return csr_matrix((len(train_df), 0)), csr_matrix((len(test_df), 0))

    return X_train_vec, X_test_vec

def train_model(train_df, test_df, numeric_features, hyperparams, vectorizer_cls, args):
    is_medbert = vectorizer_cls == MedBERTVectorizer

    def get_vectors(column_name):
        if is_medbert:
            vectorizer = vectorizer_cls(max_features=args.max_features)
            X_train_vec = vectorizer.fit_transform(train_df[column_name].fillna('').astype(str))
            X_test_vec = vectorizer.transform(test_df[column_name].fillna('').astype(str))
            return X_train_vec, X_test_vec
        else:
            vectorizer = vectorizer_cls(max_features=args.max_features)
            return safe_vectorize(column_name, vectorizer, train_df, test_df)

    train_med_vec, test_med_vec = get_vectors('med_text')
    train_diag_vec, test_diag_vec = get_vectors('diag_text')
    train_proc_vec, test_proc_vec = get_vectors('proc_text')

    additional_features_train = []
    additional_features_test = []

    if args.prompt_type == "compact":
        columns = [
            'admission_category', 'dose_text', 'new_medications', 'new_diagnoses',
            'new_procedures', 'new_dose', 'no_more_medications', 'no_more_diagnoses',
            'no_more_procedures', 'no_more_dose'
        ]
        for col in columns:
            X_train_vec, X_test_vec = get_vectors(col)
            additional_features_train.append(X_train_vec)
            additional_features_test.append(X_test_vec)

    X_train_numeric = train_df[numeric_features].values
    X_test_numeric = test_df[numeric_features].values

    if is_medbert:
        X_train = np.hstack([train_med_vec, train_diag_vec, train_proc_vec, *additional_features_train, X_train_numeric]) if args.prompt_type == "compact" else np.hstack([train_med_vec, train_diag_vec, train_proc_vec, X_train_numeric])
        X_test = np.hstack([test_med_vec, test_diag_vec, test_proc_vec, *additional_features_test, X_test_numeric]) if args.prompt_type == "compact" else np.hstack([test_med_vec, test_diag_vec, test_proc_vec, X_test_numeric])
    else:
        X_train = hstack([train_med_vec, train_diag_vec, train_proc_vec, *additional_features_train, csr_matrix(X_train_numeric)]) if args.prompt_type == "compact" else hstack([train_med_vec, train_diag_vec, train_proc_vec, csr_matrix(X_train_numeric)])
        X_test = hstack([test_med_vec, test_diag_vec, test_proc_vec, *additional_features_test, csr_matrix(X_test_numeric)]) if args.prompt_type == "compact" else hstack([test_med_vec, test_diag_vec, test_proc_vec, csr_matrix(X_test_numeric)])

    y_train = train_df['death_in_90days'].values
    y_test = test_df['death_in_90days'].values

    model = HistGradientBoostingClassifier(random_state=args.random_state, **hyperparams)

    X_train_dense = X_train.toarray() if not is_medbert else X_train
    X_test_dense = X_test.toarray() if not is_medbert else X_test

    model.fit(X_train_dense, y_train)

    y_pred_proba = model.predict_proba(X_test_dense)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred_binary)
    mortality_rate = np.mean(y_test) * 100

    return auc, f1, len(test_df), mortality_rate

def set_seed(seed_value=5550):
    os.environ["PYTHONHASHSEED"] = "9550"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed_value)
    np.random.seed(seed_value)

def main():
    args = parse_args()

    BEST_HYPERPARAMS = {
        1: {'min_samples_leaf': 30, 'max_leaf_nodes': 15, 'max_iter': 300, 'max_depth': 5, 'learning_rate': 0.05, 'l2_regularization': 0},
        2: {'min_samples_leaf': 30, 'max_leaf_nodes': 15, 'max_iter': 300, 'max_depth': 5, 'learning_rate': 0.05, 'l2_regularization': 0},
        3: {'min_samples_leaf': 10, 'max_leaf_nodes': 31, 'max_iter': 500, 'max_depth': 7, 'learning_rate': 0.1, 'l2_regularization': 0.1},
    }

    set_seed(args.random_state)

    if args.vectorizer_type == "count":
        Vectorizer = CountVectorizer
    elif args.vectorizer_type == "tfidf":
        Vectorizer = TfidfVectorizer
    elif args.vectorizer_type == "medbert":
        Vectorizer = MedBERTVectorizer
    else:
        raise ValueError(f"Vectorizer type '{args.vectorizer_type}' not supported.")

    numeric_features = ['age_at_landmark', 'landmark_visit', 'num_total_visits', 'days_since_previous_visit', 'gender_numeric']

    full_train_df = pd.read_csv(args.train_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    full_test_df = pd.read_csv(args.test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    filtered_datasets = []
    for dataset in [full_train_df, full_test_df]:
        visit_counts = dataset['subject_id'].value_counts()
        selected_patients = visit_counts[visit_counts == args.max_visits].index
        dataset_selected = dataset[dataset['subject_id'].isin(selected_patients)].copy()
        filtered_datasets.append(dataset_selected)

    train_df_selected, test_df_selected = filtered_datasets

    results = []

    for landmark_visit in range(1, args.max_visits + 1):
        logger.info(f"Processing Landmark Visit: {landmark_visit}")

        start_time = datetime.datetime.now()

        train_df = train_df_selected[train_df_selected['landmark_visit'] == landmark_visit].copy()
        test_df = test_df_selected[test_df_selected['landmark_visit'] == landmark_visit].copy()

        if train_df.empty or test_df.empty or len(np.unique(test_df['death_in_90days'])) < 2:
            logger.warning(f"Skipping Landmark {landmark_visit} due to insufficient data.")
            continue

        hyperparams = BEST_HYPERPARAMS.get(landmark_visit)
        if hyperparams is None:
            logger.error(f"No hyperparameters found for Landmark {landmark_visit}.")
            continue

        logger.info(f"Using hyperparameters for Landmark {landmark_visit}: {hyperparams}")

        auc, f1, num_patients, mortality_rate = train_model(
            train_df, test_df, numeric_features, hyperparams, Vectorizer, args
        )

        end_time = datetime.datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        logger.info(f"Landmark {landmark_visit} started at {start_time.strftime('%H:%M:%S')} and ended at {end_time.strftime('%H:%M:%S')} — duration: {elapsed_time:.2f} seconds")

        results.append({
            'Landmark (Visit Number)': landmark_visit,
            'Number of Patients (Test)': num_patients,
            'Mortality (%)': f"{mortality_rate:.1f}%",
            'AUC (Test)': f"{auc:.4f}",
            'F1-Score': f"{f1:.4f}",
            'Elapsed Time (seconds)': f"{elapsed_time:.2f}"
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    results_path = os.path.join(args.output_dir, f"sklearn_gb_results_{args.prompt_type}.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("Results saved to CSV.")

if __name__ == "__main__":
    main()