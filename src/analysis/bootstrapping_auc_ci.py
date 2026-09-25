import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95, seed=42):
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        score = roc_auc_score(np.array(y_true)[indices], np.array(y_pred)[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper = sorted_scores[int((alpha + (1.0 - alpha) / 2) * len(sorted_scores))]
    return lower, upper