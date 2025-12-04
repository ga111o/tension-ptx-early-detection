"""
Optuna optimization utilities
"""

import numpy as np
from typing import Callable
import pandas as pd

import optuna
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner, NopPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from omegaconf import DictConfig

from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer

from .model_utils import train_xgboost, train_lightgbm
from .preprocessing_utils import apply_resampling


def suggest_param(trial: optuna.Trial, name: str, param_cfg: DictConfig):
    """
    Suggest parameter value based on configuration

    Args:
        trial: Optuna trial
        name: Parameter name
        param_cfg: Parameter configuration

    Returns:
        Suggested parameter value
    """
    param_type = param_cfg.get("type", "float")

    if param_type == "int":
        step = param_cfg.get("step", None)
        if step:
            return trial.suggest_int(name, param_cfg.min, param_cfg.max, step=step)
        return trial.suggest_int(name, param_cfg.min, param_cfg.max)

    elif param_type == "float":
        log = param_cfg.get("log", False)
        return trial.suggest_float(name, param_cfg.min, param_cfg.max, log=log)

    elif param_type == "categorical":
        choices = list(param_cfg.get("choices", []))
        return trial.suggest_categorical(name, choices)

    else:
        raise ValueError(f"Unknown param type: {param_type}")


def create_sampler(sampler_cfg: DictConfig):
    """
    Create Optuna sampler based on configuration

    Args:
        sampler_cfg: Sampler configuration

    Returns:
        Optuna sampler
    """
    sampler_type = sampler_cfg.get("type", "TPESampler")
    seed = sampler_cfg.get("seed", 42)

    if sampler_type == "TPESampler":
        n_startup_trials = sampler_cfg.get("n_startup_trials", 10)
        multivariate = sampler_cfg.get("multivariate", False)
        return TPESampler(seed=seed, n_startup_trials=n_startup_trials, multivariate=multivariate)
    elif sampler_type == "RandomSampler":
        return RandomSampler(seed=seed)
    elif sampler_type == "CmaEsSampler":
        return CmaEsSampler(seed=seed)
    else:
        return TPESampler(seed=seed)


def create_pruner(pruner_cfg: DictConfig):
    """
    Create Optuna pruner based on configuration

    Args:
        pruner_cfg: Pruner configuration

    Returns:
        Optuna pruner
    """
    pruner_type = pruner_cfg.get("type", "MedianPruner")

    if pruner_type == "MedianPruner":
        return MedianPruner(
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 2),
            interval_steps=pruner_cfg.get("interval_steps", 1),
        )
    elif pruner_type == "PercentilePruner":
        return PercentilePruner(
            percentile=pruner_cfg.get("percentile", 25.0),
            n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
            n_warmup_steps=pruner_cfg.get("n_warmup_steps", 0),
        )
    elif pruner_type == "HyperbandPruner":
        return HyperbandPruner()
    elif pruner_type == "NopPruner":
        return NopPruner()
    else:
        return MedianPruner()


def _create_objective(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    cfg: DictConfig,
    model_cfg: DictConfig,
) -> Callable:
    """
    Create Optuna objective function

    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels for GroupKFold
        model_type: Model type ('xgboost' or 'lightgbm')
        cfg: Main configuration
        model_cfg: Model-specific configuration

    Returns:
        Objective function for Optuna
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import precision_recall_curve, recall_score

    pipeline_cfg = cfg.pipeline
    optuna_cfg = cfg.optuna

    n_folds = pipeline_cfg.n_folds
    use_cost_sensitive = pipeline_cfg.use_cost_sensitive
    use_gpu = pipeline_cfg.use_gpu
    resampling_method = pipeline_cfg.resampling_method
    random_seed = pipeline_cfg.random_seed
    metric = optuna_cfg.metric
    imputer_strategy = cfg.imputer.strategy

    def objective(trial: optuna.Trial) -> float:
        optuna_range = model_cfg.optuna_range

        params = {}

        for param_name in ["n_estimators", "max_depth", "learning_rate", "subsample",
                           "colsample_bytree", "reg_alpha", "reg_lambda"]:
            if param_name in optuna_range:
                params[param_name] = suggest_param(trial, param_name, optuna_range[param_name])

        if model_type == "xgboost":
            if "min_child_weight" in optuna_range:
                params["min_child_weight"] = suggest_param(trial, "min_child_weight", optuna_range["min_child_weight"])
            if "gamma" in optuna_range:
                params["gamma"] = suggest_param(trial, "gamma", optuna_range["gamma"])
        else:  # lightgbm
            if "min_child_samples" in optuna_range:
                params["min_child_samples"] = suggest_param(trial, "min_child_samples", optuna_range["min_child_samples"])
            if "num_leaves" in optuna_range:
                params["num_leaves"] = suggest_param(trial, "num_leaves", optuna_range["num_leaves"])
            if "min_split_gain" in optuna_range:
                params["min_split_gain"] = suggest_param(trial, "min_split_gain", optuna_range["min_split_gain"])

        if use_cost_sensitive and "pos_weight_multiplier" in optuna_range:
            pos_weight_multiplier = suggest_param(trial, "pos_weight_multiplier", optuna_range["pos_weight_multiplier"])
            n_pos = np.sum(y == 1)
            n_neg = len(y) - n_pos
            base_weight = n_neg / max(n_pos, 1)
            params["scale_pos_weight"] = base_weight * pos_weight_multiplier

        gkf = GroupKFold(n_splits=n_folds)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            imputer = SimpleImputer(strategy=imputer_strategy)
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            X_train_res, y_train_res = apply_resampling(X_train, y_train, resampling_method, random_seed, cfg.resampling)
            effective_cost = use_cost_sensitive and resampling_method == "none"

            if model_type == "xgboost":
                model = train_xgboost(X_train_res, y_train_res, X_val, y_val, model_cfg, cfg, params, effective_cost, use_gpu, random_seed)
            else:
                model = train_lightgbm(X_train_res, y_train_res, X_val, y_val, model_cfg, cfg, params, effective_cost, use_gpu, random_seed)

            y_prob_raw = model.predict_proba(X_val)
            if y_prob_raw.ndim == 1:
                y_prob = y_prob_raw
            else:
                y_prob = y_prob_raw[:, 1]

            if metric == "auprc":
                score = average_precision_score(y_val, y_prob)
            elif metric == "auroc":
                score = roc_auc_score(y_val, y_prob)
            elif metric in ["logloss", "log_loss", "binary_logloss"]:
                score = -log_loss(y_val, y_prob)
            elif metric == "recall":
                precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, y_prob)
                target = pipeline_cfg.target_recall
                valid_idx = np.where(recall_arr >= target)[0]
                if len(valid_idx) > 0:
                    best_idx = valid_idx[np.argmax(precision_arr[valid_idx])]
                    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                else:
                    best_threshold = thresholds[0] if len(thresholds) > 0 else 0.1

                y_pred = (y_prob >= best_threshold).astype(int)
                current_recall = recall_score(y_val, y_pred, zero_division=0)
                current_precision = precision_score(y_val, y_pred, zero_division=0)
                score = current_recall + 0.5 * current_precision
            elif metric == "recall_at_threshold":
                threshold = 0.3
                y_pred = (y_prob >= threshold).astype(int)
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == "f1_score":
                precision, recall, _ = precision_recall_curve(y_val, y_prob)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                score = np.max(f1_scores)
            elif metric == "f2_score":
                precision, recall, _ = precision_recall_curve(y_val, y_prob)
                f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
                score = np.max(f2_scores)
            elif metric == "f3_score":
                precision, recall, _ = precision_recall_curve(y_val, y_prob)
                beta = 3
                f_beta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-8)
                score = np.max(f_beta)
            else:
                score = average_precision_score(y_val, y_prob)

            scores.append(score)

            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    return objective


def optimize_hyperparameters(
    features_df: pd.DataFrame,
    model_type: str,
    cfg: DictConfig,
    model_cfg: DictConfig,
) -> dict:
    """
    Optimize hyperparameters using Optuna

    Args:
        features_df: Feature DataFrame
        model_type: Model type ('xgboost' or 'lightgbm')
        cfg: Main configuration
        model_cfg: Model-specific configuration

    Returns:
        Optimization results
    """
    from .gpu_utils import GPU_AVAILABLE, check_gpu_availability
    from .data_utils import get_feature_columns

    pipeline_cfg = cfg.pipeline
    optuna_cfg = cfg.optuna
    data_cfg = cfg.data
    meta_cols = list(data_cfg.meta_cols)

    if pipeline_cfg.use_gpu and GPU_AVAILABLE is None:
        GPU_AVAILABLE = check_gpu_availability()

    feature_cols = get_feature_columns(features_df, meta_cols)
    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_trials = optuna_cfg.n_trials
    timeout = optuna_cfg.get("timeout", None)
    n_jobs = optuna_cfg.get("n_jobs", 1)
    show_progress = optuna_cfg.get("show_progress_bar", True)

    print(f"\n{'=' * 60}")
    print(f"Run Optuna {model_type.upper()}")
    print(f"{'=' * 60}")
    print(f"Trials: {n_trials}, CV Folds: {pipeline_cfg.n_folds}, Metric: {optuna_cfg.metric}")

    objective = _create_objective(X, y, groups, model_type, cfg, model_cfg)

    sampler = create_sampler(optuna_cfg.sampler)
    pruner = create_pruner(optuna_cfg.pruner)

    study = optuna.create_study(
        study_name=f"{model_type}_opt",
        direction=optuna_cfg.direction,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=show_progress, n_jobs=n_jobs)

    print(f"\nBest {optuna_cfg.metric}: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    return {"best_params": study.best_params, "best_value": study.best_value, "study": study, "model_type": model_type}
