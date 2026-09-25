import os
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import hstack

# Configura logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="scikit-learn Gradient Boosting training script.")
    parser.add_argument("--input_csv", type=str, default="landmark_df.csv", help="Input CSV file path")
    parser.add_argument("--max_visits", type=int, default=3, help="Max number of landmark visits")
    parser.add_argument("--random_state", type=int, default=9550, help="Random seed")
    parser.add_argument("--max_features", type=int, default=1000, help="Max features for CountVectorizer")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for results")
    return parser.parse_args()

def train_model(train_df, test_df, numeric_features, args):
    # Vectorizer per testi
    med_vectorizer = CountVectorizer(max_features=args.max_features)
    diag_vectorizer = CountVectorizer(max_features=args.max_features)
    proc_vectorizer = CountVectorizer(max_features=args.max_features)

    train_med_vec = med_vectorizer.fit_transform(train_df['med_text'])
    test_med_vec = med_vectorizer.transform(test_df['med_text'])

    train_diag_vec = diag_vectorizer.fit_transform(train_df['diag_text'])
    test_diag_vec = diag_vectorizer.transform(test_df['diag_text'])

    train_proc_vec = proc_vectorizer.fit_transform(train_df['proc_text'])
    test_proc_vec = proc_vectorizer.transform(test_df['proc_text'])

    X_train_numeric = train_df[numeric_features].values
    X_test_numeric = test_df[numeric_features].values

    X_train = hstack([train_med_vec, train_diag_vec, train_proc_vec, X_train_numeric])
    X_test = hstack([test_med_vec, test_diag_vec, test_proc_vec, X_test_numeric])

    y_train = train_df['death_in_90days'].values
    y_test = test_df['death_in_90days'].values

    # Definizione parametri per RandomizedSearchCV
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

    # Fitting (HistGradientBoostingClassifier non accetta sparse!)
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

def main():
    args = parse_args()

    numeric_features = ['age_at_landmark', 'landmark_visit', 'num_total_visits', 'days_since_previous_visit', 'gender_numeric']
    landmark_df = pd.read_csv(args.input_csv)

    visit_counts = landmark_df['subject_id'].value_counts()
    selected_patients = visit_counts[visit_counts == args.max_visits].index
    df_selected = landmark_df[landmark_df['subject_id'].isin(selected_patients)].copy()

    results = []

    for landmark_visit in range(2, args.max_visits + 1): # TODO: to readjust to 1, now just for finishing the test
        logger.info(f"Processing Landmark Visit: {landmark_visit}")
        df_subset = df_selected[df_selected['landmark_visit'] == landmark_visit]
        patients = df_subset['subject_id'].unique()

        train_patients, test_patients = train_test_split(
            patients, test_size=0.2, random_state=args.random_state,
            stratify=df_subset.groupby('subject_id')['death_in_90days'].max()
        )

        train_df = df_subset[df_subset['subject_id'].isin(train_patients)].copy()
        test_df = df_subset[df_subset['subject_id'].isin(test_patients)].copy()

        if train_df.empty or test_df.empty or len(np.unique(test_df['death_in_90days'])) < 2:
            logger.warning(f"Skipping Landmark {landmark_visit} due to insufficient data.")
            continue

        auc, f1, num_patients, mortality_rate = train_model(
            train_df, test_df, numeric_features, args
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