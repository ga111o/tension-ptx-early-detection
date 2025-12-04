import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from omegaconf import DictConfig


def optimize_ensemble_weights(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    cfg: DictConfig
) -> float:
    ensemble_cfg = cfg.ensemble
    metric_name = ensemble_cfg.soft_voting_metric

    if metric_name == "auroc":
        metric_func = roc_auc_score
    elif metric_name == "auprc":
        metric_func = average_precision_score
    else:
        metric_func = roc_auc_score

    best_score = -1
    best_w = 0.5

    for w in np.linspace(0, 1, 101):
        p_ensemble = w * p1 + (1 - w) * p2
        score = metric_func(y_true, p_ensemble)
        if score > best_score:
            best_score = score
            best_w = w

    return best_w


def train_stacking_model(
    X_meta_train: np.ndarray,
    y_train: np.ndarray,
    cfg: DictConfig
) -> LogisticRegression:
    ensemble_cfg = cfg.ensemble

    meta_model = LogisticRegression(
        solver=ensemble_cfg.stacking_solver,
        C=ensemble_cfg.stacking_C,
        random_state=ensemble_cfg.stacking_random_state
    )
    meta_model.fit(X_meta_train, y_train)
    return meta_model
