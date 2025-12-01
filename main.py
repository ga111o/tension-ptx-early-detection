import json
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ML Libraries
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from imblearn.over_sampling import ADASYN, SMOTE
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.pruners import MedianPruner, PercentilePruner, HyperbandPruner, NopPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

GPU_AVAILABLE: Optional[Dict[str, bool]] = None

def check_gpu_availability() -> Dict[str, bool]:
    gpu_status = {"xgboost": False, "lightgbm": False}

    try:
        test_dm = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
        xgb.train({"tree_method": "hist", "device": "cuda"}, test_dm, num_boost_round=1)
        gpu_status["xgboost"] = True
        print("XGBoost CUDA")
    except Exception:
        try:
            xgb.train({"tree_method": "gpu_hist"}, test_dm, num_boost_round=1)
            gpu_status["xgboost"] = True
            print("XGBoost gpu_hist")
        except Exception:
            print("XGBoost CPU only")

    try:
        test_ds = lgb.Dataset(np.array([[1, 2], [3, 4]]), label=[0, 1])
        lgb.train({"device": "gpu", "verbose": -1, "num_iterations": 1}, test_ds)
        gpu_status["lightgbm"] = True
        print("LightGBM CUDA")
    except Exception:
        print("LightGBM CPU only")

    return gpu_status

def set_gpu_status(status: Dict[str, bool]):
    global GPU_AVAILABLE
    GPU_AVAILABLE = status

def load_preprocessed_features(path: str, meta_cols: List[str]) -> pd.DataFrame:
    print(f"\n[전처리된 Features 로드] {path}")
    df = pd.read_csv(path)
    
    feature_cols = [c for c in df.columns if c not in meta_cols]
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    
    print(f"  샘플: {len(df)}개 (Positive={n_pos}, Negative={n_neg})")
    print(f"  Features: {len(feature_cols)}개")
    
    missing_count = df[feature_cols].isna().sum().sum()
    if missing_count > 0:
        print(f"  결측치: {missing_count}개")
    else:
        print(f"  결측치 없음")
    
    return df


def get_feature_columns(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in meta_cols]

def apply_resampling(
    X: np.ndarray, 
    y: np.ndarray, 
    method: str,
    random_seed: int,
    resampling_cfg: DictConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return X, y

    print(f"  리샘플링: {method.upper()} (전: Pos={sum(y)}, Neg={len(y) - sum(y)})")

    if method == "smote":
        k_neighbors = resampling_cfg.smote.k_neighbors
        sampler = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
    elif method == "adasyn":
        n_neighbors = resampling_cfg.adasyn.n_neighbors
        sampler = ADASYN(random_state=random_seed, n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unknown method: {method}")

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"       (후: Pos={sum(y_res)}, Neg={len(y_res) - sum(y_res)})")
    return X_res, y_res


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: DictConfig,
    params: Optional[Dict] = None,
    use_cost_sensitive: bool = True,
    use_gpu: bool = True,
    random_seed: int = 42,
) -> xgb.XGBClassifier:
    global GPU_AVAILABLE

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
        default_params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_cfg: DictConfig,
    params: Optional[Dict] = None,
    use_cost_sensitive: bool = True,
    use_gpu: bool = True,
    random_seed: int = 42,
) -> lgb.LGBMClassifier:
    global GPU_AVAILABLE

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
        default_params["scale_pos_weight"] = scale_pos_weight

    early_stopping = default_params.pop("early_stopping_rounds")

    model = lgb.LGBMClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
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
    use_cost_sensitive: bool = True,
    use_gpu: bool = True,
    random_seed: int = 42,
):
    if model_type.lower() == "xgboost":
        return train_xgboost(X_train, y_train, X_val, y_val, model_cfg, best_params, use_cost_sensitive, use_gpu, random_seed)
    return train_lightgbm(X_train, y_train, X_val, y_val, model_cfg, best_params, use_cost_sensitive, use_gpu, random_seed)

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
    method: str,
    default_threshold: float,
) -> Tuple[float, Dict]:
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


def get_feature_importance(model, feature_names: List[str], model_name: str, top_n: int) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\n{model_name} - Top {top_n} Features:")
    print(importance_df.head(top_n).to_string(index=False))
    return importance_df


def suggest_param(trial: optuna.Trial, name: str, param_cfg: DictConfig):
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


def run_cross_validation(
    features_df: pd.DataFrame,
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
    lgb_model_cfg: DictConfig,
) -> Dict:
    """GroupKFold Cross-Validation"""
    global GPU_AVAILABLE

    pipeline_cfg = cfg.pipeline
    data_cfg = cfg.data
    
    n_folds = pipeline_cfg.n_folds
    resampling_method = pipeline_cfg.resampling_method
    use_cost_sensitive = pipeline_cfg.use_cost_sensitive
    use_gpu = pipeline_cfg.use_gpu
    target_recall = pipeline_cfg.target_recall
    threshold_method = pipeline_cfg.threshold_method
    random_seed = pipeline_cfg.random_seed
    meta_cols = list(data_cfg.meta_cols)
    default_threshold = cfg.threshold.default

    if use_gpu and GPU_AVAILABLE is None:
        print("\nCheck GPU")
        GPU_AVAILABLE = check_gpu_availability()

    feature_cols = get_feature_columns(features_df, meta_cols)
    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    print(f"Data: {len(y)} (Pos={sum(y)}, Neg={len(y) - sum(y)}), Features: {len(feature_cols)}")

    gkf = GroupKFold(n_splits=n_folds)
    cv_results = {
        "XGBoost": {"auroc": [], "auprc": [], "f1": [], "precision": [], "recall": [], "accuracy": []},
        "LightGBM": {"auroc": [], "auprc": [], "f1": [], "precision": [], "recall": [], "accuracy": []},
    }
    optimal_thresholds = {"XGBoost": [], "LightGBM": []}

    effective_cost_sensitive = use_cost_sensitive and resampling_method == "none"

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\nCV Fold {fold + 1}/{n_folds}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"  Train: {len(y_train)} (Pos={sum(y_train)}), Test: {len(y_test)} (Pos={sum(y_test)})")

        if np.isnan(X_train).sum() > 0:
            imputer = SimpleImputer(strategy=cfg.imputer.strategy)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_res, y_train_res = apply_resampling(X_train, y_train, resampling_method, random_seed, cfg.resampling)

        # XGBoost
        xgb_model = train_xgboost(X_train_res, y_train_res, X_test, y_test, xgb_model_cfg,
                                   use_cost_sensitive=effective_cost_sensitive, use_gpu=use_gpu, random_seed=random_seed)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        opt_thresh_xgb, _ = find_optimal_threshold(y_train, xgb_model.predict_proba(X_train)[:, 1], 
                                                    target_recall, threshold_method, default_threshold)
        optimal_thresholds["XGBoost"].append(opt_thresh_xgb)
        y_pred_xgb = (y_prob_xgb >= opt_thresh_xgb).astype(int)

        cv_results["XGBoost"]["auroc"].append(roc_auc_score(y_test, y_prob_xgb))
        cv_results["XGBoost"]["auprc"].append(average_precision_score(y_test, y_prob_xgb))
        cv_results["XGBoost"]["f1"].append(f1_score(y_test, y_pred_xgb, zero_division=0))
        cv_results["XGBoost"]["precision"].append(precision_score(y_test, y_pred_xgb, zero_division=0))
        cv_results["XGBoost"]["recall"].append(recall_score(y_test, y_pred_xgb, zero_division=0))
        cv_results["XGBoost"]["accuracy"].append(accuracy_score(y_test, y_pred_xgb))

        print(f"  XGBoost  AUROC={cv_results['XGBoost']['auroc'][-1]:.4f}, AUPRC={cv_results['XGBoost']['auprc'][-1]:.4f}")

        # LightGBM
        lgb_model = train_lightgbm(X_train_res, y_train_res, X_test, y_test, lgb_model_cfg,
                                    use_cost_sensitive=effective_cost_sensitive, use_gpu=use_gpu, random_seed=random_seed)
        y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        opt_thresh_lgb, _ = find_optimal_threshold(y_train, lgb_model.predict_proba(X_train)[:, 1], 
                                                    target_recall, threshold_method, default_threshold)
        optimal_thresholds["LightGBM"].append(opt_thresh_lgb)
        y_pred_lgb = (y_prob_lgb >= opt_thresh_lgb).astype(int)

        cv_results["LightGBM"]["auroc"].append(roc_auc_score(y_test, y_prob_lgb))
        cv_results["LightGBM"]["auprc"].append(average_precision_score(y_test, y_prob_lgb))
        cv_results["LightGBM"]["f1"].append(f1_score(y_test, y_pred_lgb, zero_division=0))
        cv_results["LightGBM"]["precision"].append(precision_score(y_test, y_pred_lgb, zero_division=0))
        cv_results["LightGBM"]["recall"].append(recall_score(y_test, y_pred_lgb, zero_division=0))
        cv_results["LightGBM"]["accuracy"].append(accuracy_score(y_test, y_pred_lgb))

        print(f"  LightGBM AUROC={cv_results['LightGBM']['auroc'][-1]:.4f}, AUPRC={cv_results['LightGBM']['auprc'][-1]:.4f}")

    summary_data = []
    for model_name in ["XGBoost", "LightGBM"]:
        for metric in ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]:
            scores = cv_results[model_name][metric]
            summary_data.append({
                "Model": model_name,
                "Metric": metric.upper(),
                "Mean": np.mean(scores),
                "Std": np.std(scores),
            })

    summary_df = pd.DataFrame(summary_data)
    for model in ["XGBoost", "LightGBM"]:
        print(f"\n[{model}]")
        for _, row in summary_df[summary_df["Model"] == model].iterrows():
            print(f"  {row['Metric']}: {row['Mean']:.4f} +- {row['Std']:.4f}")

    print(f"\nThreshold Average")
    print(f"  XGBoost:  {np.mean(optimal_thresholds['XGBoost']):.4f}")
    print(f"  LightGBM: {np.mean(optimal_thresholds['LightGBM']):.4f}")

    return {"cv_results": cv_results, "summary": summary_df, "optimal_thresholds": optimal_thresholds, "feature_cols": feature_cols}


def _create_objective(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    cfg: DictConfig,
    model_cfg: DictConfig,
) -> Callable:
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

        if "pos_weight_multiplier" in optuna_range:
            pos_weight_multiplier = suggest_param(trial, "pos_weight_multiplier", optuna_range["pos_weight_multiplier"])
        else:
            pos_weight_multiplier = 1.0
        
        n_pos = np.sum(y == 1)
        n_neg = len(y) - n_pos
        base_weight = n_neg / max(n_pos, 1)
        
        params["scale_pos_weight"] = base_weight * pos_weight_multiplier

        gkf = GroupKFold(n_splits=n_folds)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 전처리
            imputer = SimpleImputer(strategy=imputer_strategy)
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            X_train_res, y_train_res = apply_resampling(X_train, y_train, resampling_method, random_seed, cfg.resampling)
            effective_cost = use_cost_sensitive and resampling_method == "none"

            if model_type == "xgboost":
                model = train_xgboost(X_train_res, y_train_res, X_val, y_val, model_cfg, params, effective_cost, use_gpu, random_seed)
            else:
                model = train_lightgbm(X_train_res, y_train_res, X_val, y_val, model_cfg, params, effective_cost, use_gpu, random_seed)

            y_prob = model.predict_proba(X_val)[:, 1]

            # 메트릭 계산
            if metric == "auprc":
                score = average_precision_score(y_val, y_prob)
            elif metric == "auroc":
                score = roc_auc_score(y_val, y_prob)
            elif metric == "recall":
                # Target recall을 달성하면서 precision을 최대화하는 threshold 찾기
                precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, y_prob)
                target = pipeline_cfg.target_recall
                # target recall 이상인 지점에서 precision이 최대인 threshold
                valid_idx = np.where(recall_arr >= target)[0]
                if len(valid_idx) > 0:
                    # valid한 recall 중 precision이 가장 높은 지점
                    best_idx = valid_idx[np.argmax(precision_arr[valid_idx])]
                    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
                else:
                    # target recall 달성 불가 시, recall이 가장 높은 threshold (낮은 threshold)
                    best_threshold = thresholds[0] if len(thresholds) > 0 else 0.1
                
                y_pred = (y_prob >= best_threshold).astype(int)
                current_recall = recall_score(y_val, y_pred, zero_division=0)
                current_precision = precision_score(y_val, y_pred, zero_division=0)
                # Recall + Precision/2 (Recall에 더 가중치)
                score = current_recall + 0.5 * current_precision
            elif metric == "recall_at_threshold":
                # 특정 threshold에서의 recall 최적화
                threshold = 0.3  # 낮은 threshold로 recall 우선
                y_pred = (y_prob >= threshold).astype(int)
                score = recall_score(y_val, y_pred, zero_division=0)
            elif metric == "f1_score":
                precision, recall, _ = precision_recall_curve(y_val, y_prob)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                score = np.max(f1_scores)
            elif metric == "f2_score":
                # F2 score: Recall에 2배 가중치
                precision, recall, _ = precision_recall_curve(y_val, y_prob)
                f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
                score = np.max(f2_scores)
            elif metric == "f3_score":
                # F3 score: Recall에 3배 가중치 (더 aggressive)
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
) -> Dict:
    global GPU_AVAILABLE

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


def run_optimization_pipeline(
    features_df: pd.DataFrame,
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
    lgb_model_cfg: DictConfig,
) -> Dict:
    global GPU_AVAILABLE

    pipeline_cfg = cfg.pipeline
    data_cfg = cfg.data
    meta_cols = list(data_cfg.meta_cols)

    if pipeline_cfg.use_gpu and GPU_AVAILABLE is None:
        GPU_AVAILABLE = check_gpu_availability()

    results = {}

    xgb_opt = optimize_hyperparameters(features_df, "xgboost", cfg, xgb_model_cfg)
    results["xgboost_optimization"] = xgb_opt

    lgb_opt = optimize_hyperparameters(features_df, "lightgbm", cfg, lgb_model_cfg)
    results["lightgbm_optimization"] = lgb_opt

    cv_results = _run_cv_with_best_params(
        features_df, xgb_opt["best_params"], lgb_opt["best_params"],
        cfg, xgb_model_cfg, lgb_model_cfg,
    )
    results["cv_results"] = cv_results

    feature_cols = get_feature_columns(features_df, meta_cols)
    X = features_df[feature_cols].values
    y = features_df["label"].values

    imputer = None
    if np.isnan(X).sum() > 0:
        print(f"  Missing values: {np.isnan(X).sum()} -> MICE")
        imputer = IterativeImputer(random_state=pipeline_cfg.random_seed, max_iter=cfg.imputer.iterative.max_iter)
        X = imputer.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_res, y_res = apply_resampling(X, y, pipeline_cfg.resampling_method, pipeline_cfg.random_seed, cfg.resampling)
    effective_cost = pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none"

    print("\nTrain XGBoost")
    xgb_model = train_with_best_params(X_res, y_res, X, y, xgb_opt["best_params"], "xgboost", 
                                        xgb_model_cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    print("Train LightGBM")
    lgb_model = train_with_best_params(X_res, y_res, X, y, lgb_opt["best_params"], "lightgbm", 
                                        lgb_model_cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    results["models"] = {"xgboost": xgb_model, "lightgbm": lgb_model}
    results["feature_cols"] = feature_cols
    results["imputer"] = imputer
    results["scaler"] = scaler
    results["optimal_thresholds"] = cv_results["optimal_thresholds"]

    return results


def _run_cv_with_best_params(
    features_df: pd.DataFrame,
    xgb_params: Dict,
    lgb_params: Dict,
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
    lgb_model_cfg: DictConfig,
) -> Dict:
    pipeline_cfg = cfg.pipeline
    data_cfg = cfg.data
    meta_cols = list(data_cfg.meta_cols)
    
    n_folds = pipeline_cfg.n_folds
    use_cost_sensitive = pipeline_cfg.use_cost_sensitive
    use_gpu = pipeline_cfg.use_gpu
    resampling_method = pipeline_cfg.resampling_method
    target_recall = pipeline_cfg.target_recall
    threshold_method = pipeline_cfg.threshold_method
    random_seed = pipeline_cfg.random_seed
    default_threshold = cfg.threshold.default
    imputer_strategy = cfg.imputer.strategy

    feature_cols = get_feature_columns(features_df, meta_cols)
    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    cv_results = {
        "XGBoost": {"auroc": [], "auprc": [], "f1": [], "precision": [], "recall": [], "accuracy": []},
        "LightGBM": {"auroc": [], "auprc": [], "f1": [], "precision": [], "recall": [], "accuracy": []},
    }
    optimal_thresholds = {"XGBoost": [], "LightGBM": []}
    effective_cost = use_cost_sensitive and resampling_method == "none"

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\nCV Fold {fold + 1}/{n_folds}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if np.isnan(X_train).sum() > 0:
            imputer = SimpleImputer(strategy=imputer_strategy)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_res, y_train_res = apply_resampling(X_train, y_train, resampling_method, random_seed, cfg.resampling)

        # XGBoost
        xgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, xgb_params, 
                                            "xgboost", xgb_model_cfg, effective_cost, use_gpu, random_seed)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        opt_thresh, _ = find_optimal_threshold(y_train, xgb_model.predict_proba(X_train)[:, 1], 
                                                target_recall, threshold_method, default_threshold)
        optimal_thresholds["XGBoost"].append(opt_thresh)
        y_pred = (y_prob_xgb >= opt_thresh).astype(int)

        cv_results["XGBoost"]["auroc"].append(roc_auc_score(y_test, y_prob_xgb))
        cv_results["XGBoost"]["auprc"].append(average_precision_score(y_test, y_prob_xgb))
        cv_results["XGBoost"]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        cv_results["XGBoost"]["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        cv_results["XGBoost"]["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        cv_results["XGBoost"]["accuracy"].append(accuracy_score(y_test, y_pred))

        print(f"  XGBoost  AUROC={cv_results['XGBoost']['auroc'][-1]:.4f}, AUPRC={cv_results['XGBoost']['auprc'][-1]:.4f}")

        # LightGBM
        lgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, lgb_params, 
                                            "lightgbm", lgb_model_cfg, effective_cost, use_gpu, random_seed)
        y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        opt_thresh, _ = find_optimal_threshold(y_train, lgb_model.predict_proba(X_train)[:, 1], 
                                                target_recall, threshold_method, default_threshold)
        optimal_thresholds["LightGBM"].append(opt_thresh)
        y_pred = (y_prob_lgb >= opt_thresh).astype(int)

        cv_results["LightGBM"]["auroc"].append(roc_auc_score(y_test, y_prob_lgb))
        cv_results["LightGBM"]["auprc"].append(average_precision_score(y_test, y_prob_lgb))
        cv_results["LightGBM"]["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        cv_results["LightGBM"]["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        cv_results["LightGBM"]["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        cv_results["LightGBM"]["accuracy"].append(accuracy_score(y_test, y_pred))

        print(f"  LightGBM AUROC={cv_results['LightGBM']['auroc'][-1]:.4f}, AUPRC={cv_results['LightGBM']['auprc'][-1]:.4f}")

    summary_data = []
    for model in ["XGBoost", "LightGBM"]:
        for metric in ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]:
            scores = cv_results[model][metric]
            summary_data.append({"Model": model, "Metric": metric.upper(), "Mean": np.mean(scores), "Std": np.std(scores)})

    summary_df = pd.DataFrame(summary_data)
    for model in ["XGBoost", "LightGBM"]:
        print(f"\n[{model}]")
        for _, row in summary_df[summary_df["Model"] == model].iterrows():
            print(f"  {row['Metric']}: {row['Mean']:.4f} +- {row['Std']:.4f}")

    return {"cv_results": cv_results, "summary": summary_df, "optimal_thresholds": optimal_thresholds, "feature_cols": feature_cols}


def run_pipeline(cfg: DictConfig) -> Optional[Dict]:
    global GPU_AVAILABLE

    xgb_model_cfg = OmegaConf.load("conf/model/xgboost.yaml")
    lgb_model_cfg = OmegaConf.load("conf/model/lightgbm.yaml")
    
    pipeline_cfg = cfg.pipeline
    output_cfg = cfg.output
    data_cfg = cfg.data
    meta_cols = list(data_cfg.meta_cols)
    top_n = cfg.feature_importance.top_n

    print(f"{pipeline_cfg.n_folds}-Fold Cross-Validation")

    print("\nHydra")
    print(f"  데이터 경로: {data_cfg.features_path}")
    print(f"  CV Folds: {pipeline_cfg.n_folds}")
    print(f"  리샘플링: {pipeline_cfg.resampling_method}")
    print(f"  Cost-sensitive: {pipeline_cfg.use_cost_sensitive}")
    print(f"  GPU: {pipeline_cfg.use_gpu}")
    print(f"  Target Recall: {pipeline_cfg.target_recall}")
    print(f"  Optuna 시행: {cfg.optuna.n_trials}회")
    print(f"  최적화 메트릭: {cfg.optuna.metric}")

    if pipeline_cfg.use_gpu:
        gpu_status = check_gpu_availability()
        set_gpu_status(gpu_status)
    else:
        set_gpu_status({"xgboost": False, "lightgbm": False})
        print("\nCPU only")

    # 전처리된 Feature 로드
    features_df = load_preprocessed_features(data_cfg.features_path, meta_cols)

    if len(features_df) == 0:
        print("features_df == 0")
        return None

    print("Run Optuna")

    opt_results = run_optimization_pipeline(
        features_df, cfg, xgb_model_cfg, lgb_model_cfg,
    )

    xgb_model = opt_results["models"]["xgboost"]
    lgb_model = opt_results["models"]["lightgbm"]
    feature_cols = opt_results["feature_cols"]
    cv_results = opt_results["cv_results"]

    avg_thresh_xgb = np.mean(cv_results["optimal_thresholds"]["XGBoost"])
    avg_thresh_lgb = np.mean(cv_results["optimal_thresholds"]["LightGBM"])

    xgb_imp = get_feature_importance(xgb_model, feature_cols, "XGBoost", top_n)
    lgb_imp = get_feature_importance(lgb_model, feature_cols, "LightGBM", top_n)

    if output_cfg.save_models:
        xgb_model.save_model(output_cfg.xgboost_model_path)
        lgb_model.booster_.save_model(output_cfg.lightgbm_model_path)
        print("\n모델 저장:")
        print(f"  - {output_cfg.xgboost_model_path}")
        print(f"  - {output_cfg.lightgbm_model_path}")

    best_params = {
        "xgboost": opt_results["xgboost_optimization"]["best_params"],
        "lightgbm": opt_results["lightgbm_optimization"]["best_params"],
    }
    if output_cfg.save_params:
        with open(output_cfg.best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"  - {output_cfg.best_params_path}")

    if output_cfg.save_feature_importance:
        xgb_imp.to_csv(f"{data_cfg.output_dir}/feature_importance_xgboost_optuna.csv", index=False)
        lgb_imp.to_csv(f"{data_cfg.output_dir}/feature_importance_lightgbm_optuna.csv", index=False)

    if output_cfg.save_cv_results:
        cv_results["summary"].to_csv(f"{data_cfg.output_dir}/cv_results_summary_optuna.csv", index=False)
        print(f"  - {data_cfg.output_dir}/cv_results_summary_optuna.csv")

    return {
        "models": {"xgboost": xgb_model, "lightgbm": lgb_model},
        "cv_results": cv_results,
        "feature_importance": {"xgboost": xgb_imp, "lightgbm": lgb_imp},
        "feature_names": feature_cols,
        "optimal_thresholds": {"xgboost": avg_thresh_xgb, "lightgbm": avg_thresh_lgb},
        "best_params": best_params,
        "optimization_results": opt_results,
        "scaler": opt_results["scaler"],
        "imputer": opt_results["imputer"],
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    import os
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)
    
    
    logger.info(f"Original CWD: {original_cwd}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    results = run_pipeline(cfg)
    logger.info(results)
    

if __name__ == "__main__":
    main()
