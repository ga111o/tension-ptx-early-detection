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

    if use_cost_sensitive and "scale_pos_weight" not in default_params:
        if not (cfg and cfg.focal_loss.enabled):
            default_params["scale_pos_weight"] = scale_pos_weight

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

    if use_cost_sensitive and "scale_pos_weight" not in default_params:
        if not (cfg and cfg.focal_loss.enabled):
            default_params["scale_pos_weight"] = scale_pos_weight
    
    objective_func = default_params.pop("objective", None)
    if "fobj" in default_params:
        del default_params["fobj"]

    if cfg and cfg.focal_loss.enabled:
        focal_loss = UnifiedFocalLoss(gamma=cfg.focal_loss.gamma, alpha=cfg.focal_loss.alpha)
        objective_func = focal_loss.lgb_obj
    
    early_stopping = default_params.pop("early_stopping_rounds", None)

    model = lgb.LGBMClassifier(objective=objective_func, **default_params)

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
