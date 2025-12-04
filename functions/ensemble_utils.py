"""
Ensemble model utilities
"""

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
    """
    Optimize weights for Soft Voting ensemble

    Args:
        y_true: True labels
        p1: Predictions from model 1
        p2: Predictions from model 2
        cfg: Ensemble configuration

    Returns:
        Optimal weight for model 1 (weight for model 2 = 1 - weight)
    """
    ensemble_cfg = cfg.ensemble
    metric_name = ensemble_cfg.soft_voting_metric

    # 메트릭 함수 선택
    if metric_name == "auroc":
        metric_func = roc_auc_score
    elif metric_name == "auprc":
        metric_func = average_precision_score
    else:
        metric_func = roc_auc_score  # 기본값

    best_score = -1
    best_w = 0.5

    # 0.0 ~ 1.0 사이를 100등분하여 탐색
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
    """
    Train stacking meta model (Logistic Regression)

    Args:
        X_meta_train: Meta features for training
        y_train: Training labels
        cfg: Ensemble configuration

    Returns:
        Trained meta model
    """
    ensemble_cfg = cfg.ensemble

    meta_model = LogisticRegression(
        solver=ensemble_cfg.stacking_solver,
        C=ensemble_cfg.stacking_C,
        random_state=ensemble_cfg.stacking_random_state
    )
    meta_model.fit(X_meta_train, y_train)
    return meta_model
