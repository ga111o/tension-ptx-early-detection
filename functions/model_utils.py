"""
Model training utilities for XGBoost and LightGBM
"""

import numpy as np
from typing import Dict, Optional

import xgboost as xgb
import lightgbm as lgb
from omegaconf import DictConfig, OmegaConf

from .gpu_utils import GPU_AVAILABLE
from .focal_loss import UnifiedFocalLoss


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: DictConfig,
    cfg: Optional[DictConfig] = None,
    params: Optional[Dict] = None,
    use_cost_sensitive: bool = True,
    use_gpu: bool = True,
    random_seed: int = 42,
) -> xgb.XGBClassifier:
    """
    Train XGBoost model

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        model_cfg: Model configuration
        params: Additional parameters to override defaults
        use_cost_sensitive: Whether to use cost-sensitive learning
        use_gpu: Whether to use GPU if available
        random_seed: Random seed

    Returns:
        Trained XGBoost model
    """
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    default_params = OmegaConf.to_container(model_cfg.default_params, resolve=True)
    default_params["random_state"] = random_seed

    if params:
        default_params.update(params)

    if use_gpu and GPU_AVAILABLE and GPU_AVAILABLE.get("xgboost"):
        gpu_cfg = OmegaConf.to_container(model_cfg.gpu, resolve=True)
        default_params.update(gpu_cfg)
    else:
        cpu_cfg = OmegaConf.to_container(model_cfg.cpu, resolve=True)
        default_params.update(cpu_cfg)

    # Cost-Sensitive와 Focal Loss를 동시에 사용하지 않도록 조정
    if use_cost_sensitive and "scale_pos_weight" not in default_params:
        # Focal Loss 사용 시 scale_pos_weight를 적용하지 않음
        if not (cfg and cfg.focal_loss.enabled):
            default_params["scale_pos_weight"] = scale_pos_weight

    # Apply focal loss for XGBoost
    if cfg and cfg.focal_loss.enabled:
        focal_loss = UnifiedFocalLoss(gamma=cfg.focal_loss.gamma, alpha=cfg.focal_loss.alpha)
        default_params["obj"] = focal_loss.xgb_obj

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: DictConfig,
    cfg: Optional[DictConfig] = None,
    params: Optional[Dict] = None,
    use_cost_sensitive: bool = True,
    use_gpu: bool = True,
    random_seed: int = 42,
) -> lgb.LGBMClassifier:
    """
    Train LightGBM model
    """
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    default_params = OmegaConf.to_container(model_cfg.default_params, resolve=True)
    default_params["random_state"] = random_seed

    if params:
        default_params.update(params)

    if use_gpu and GPU_AVAILABLE and GPU_AVAILABLE.get("lightgbm"):
        gpu_cfg = OmegaConf.to_container(model_cfg.gpu, resolve=True)
        default_params.update(gpu_cfg)
    else:
        cpu_cfg = OmegaConf.to_container(model_cfg.cpu, resolve=True)
        default_params.update(cpu_cfg)

    # Cost-Sensitive와 Focal Loss를 동시에 사용하지 않도록 조정
    if use_cost_sensitive and "scale_pos_weight" not in default_params:
        # Focal Loss 사용 시 scale_pos_weight를 적용하지 않음
        if not (cfg and cfg.focal_loss.enabled):
            default_params["scale_pos_weight"] = scale_pos_weight
    
    # 기본 objective 설정을 가져옵니다 (없으면 None)
    # custom objective를 쓸 경우, params 딕셔너리 안에 'objective'나 'fobj' 키가 남아있으면 충돌나므로 제거합니다.
    objective_func = default_params.pop("objective", None)
    if "fobj" in default_params:
        del default_params["fobj"]

    # Focal Loss가 활성화된 경우 objective_func를 함수 객체로 교체
    if cfg and cfg.focal_loss.enabled:
        focal_loss = UnifiedFocalLoss(gamma=cfg.focal_loss.gamma, alpha=cfg.focal_loss.alpha)
        objective_func = focal_loss.lgb_obj
    
    # Early stopping rounds 추출
    early_stopping = default_params.pop("early_stopping_rounds", None)

    # LGBMClassifier 생성 시 objective 인자를 명시적으로 전달
    # **default_params에는 문자열/숫자 파라미터만 남겨둠
    model = lgb.LGBMClassifier(objective=objective_func, **default_params)
    
    # --- [수정 끝] ---

    callbacks = []
    if early_stopping is not None:
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False))

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    return model


def train_with_best_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict,
    model_type: str,
    model_cfg: DictConfig,
    cfg: Optional[DictConfig] = None,
    use_cost_sensitive: bool = True,
    use_gpu: bool = True,
    random_seed: int = 42,
):
    """
    Train model with best parameters from optimization

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        best_params: Best parameters from optimization
        model_type: Model type ('xgboost' or 'lightgbm')
        model_cfg: Model configuration
        use_cost_sensitive: Whether to use cost-sensitive learning
        use_gpu: Whether to use GPU if available
        random_seed: Random seed

    Returns:
        Trained model
    """
    if model_type.lower() == "xgboost":
        return train_xgboost(X_train, y_train, X_val, y_val, model_cfg, cfg, best_params, use_cost_sensitive, use_gpu, random_seed)
    elif model_type.lower() == "lightgbm":
        return train_lightgbm(X_train, y_train, X_val, y_val, model_cfg, cfg, best_params, use_cost_sensitive, use_gpu, random_seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
    method: str,
    default_threshold: float,
) -> tuple[float, Dict]:
    """
    Find optimal classification threshold based on various methods

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_recall: Target recall for 'target_recall' method
        method: Threshold selection method
        default_threshold: Default threshold to use if method fails

    Returns:
        Optimal threshold and metrics dictionary
    """
    from sklearn.metrics import precision_recall_curve, roc_curve, precision_score, recall_score, f1_score

    precision_arr, recall_arr, thresholds_pr = precision_recall_curve(y_true, y_prob)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)

    if method == "target_recall":
        valid_idx = np.where(recall_arr[:-1] >= target_recall)[0]
        if len(valid_idx) > 0:
            best_precision_idx = valid_idx[np.argmax(precision_arr[valid_idx])]
            opt_threshold = thresholds_pr[best_precision_idx] if best_precision_idx < len(thresholds_pr) else default_threshold
        else:
            f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
            opt_idx = np.argmax(f1_scores[:-1])
            opt_threshold = thresholds_pr[opt_idx] if opt_idx < len(thresholds_pr) else default_threshold
    elif method == "f2":
        f2_scores = (5 * precision_arr * recall_arr) / (4 * precision_arr + recall_arr + 1e-8)
        opt_idx = np.argmax(f2_scores[:-1])
        opt_threshold = thresholds_pr[opt_idx] if opt_idx < len(thresholds_pr) else default_threshold
    elif method == "f1":
        f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
        opt_idx = np.argmax(f1_scores[:-1])
        opt_threshold = thresholds_pr[opt_idx] if opt_idx < len(thresholds_pr) else default_threshold
    elif method == "youden":
        opt_idx = np.argmax(tpr - fpr)
        opt_threshold = thresholds_roc[opt_idx] if opt_idx < len(thresholds_roc) else default_threshold
    else:
        opt_threshold = default_threshold

    opt_threshold = np.clip(opt_threshold, 0.1, 0.7)

    y_pred = (y_prob >= opt_threshold).astype(int)
    metrics = {
        "threshold": opt_threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return opt_threshold, metrics
