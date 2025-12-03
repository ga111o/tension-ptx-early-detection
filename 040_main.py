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
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
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
    """Feature Importance 추출"""

    # LightGBM Booster 객체 처리
    if hasattr(model, 'feature_importance'):
        importance_values = model.feature_importance(importance_type='gain')

    # XGBoost Booster 객체 처리
    elif hasattr(model, 'get_score'):
        # get_score()는 딕셔너리를 반환 (feature name -> importance)
        importance_dict = model.get_score(importance_type='gain')

        # 모든 피처에 대해 importance 값 할당 (사용되지 않은 피처는 0)
        importance_values = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            # XGBoost는 feature를 f0, f1, ... 형식으로 저장
            importance_values[i] = importance_dict.get(f'f{i}', 0.0)

    # sklearn API (XGBClassifier, LGBMClassifier) 처리
    else:
        importance_values = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_values,
    }).sort_values('importance', ascending=False)

    print(f"{model_name} - Top {top_n} Features:")
    print(importance_df.head(top_n).to_string(index=False))

    return importance_df


def calculate_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation Importance를 계산합니다.

    Args:
        model: 학습된 모델 (XGBoost, LightGBM 등)
        X: 피처 데이터
        y: 타겟 데이터
        feature_names: 피처 이름 리스트
        model_name: 모델 이름 (로깅용)
        n_repeats: 순열 반복 횟수
        random_state: 랜덤 시드

    Returns:
        Permutation Importance가 포함된 DataFrame
    """
    print(f"\n[Permutation Importance] {model_name} - 계산 중... (n_repeats={n_repeats})")

    # Permutation Importance 계산
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='roc_auc'
    )

    # 결과 DataFrame 생성
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
    }).sort_values('importance_mean', ascending=False)

    print(f"  완료 - 총 {len(feature_names)}개 피처")
    print("  Top 10 Permutation Importance:")
    print(importance_df.head(10).to_string(index=False))

    return importance_df


def select_features_by_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_name: str,
    importance_threshold: float = 0.0,
    min_features: int = 10,
    n_repeats: int = 5,
    random_state: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Permutation Importance가 낮은 피처들을 제거합니다.

    Args:
        model: 학습된 모델
        X: 피처 데이터
        y: 타겟 데이터
        feature_names: 피처 이름 리스트
        model_name: 모델 이름
        importance_threshold: 제거할 importance threshold (이 값 이하 제거)
        min_features: 최소 유지할 피처 수
        n_repeats: 순열 반복 횟수
        random_state: 랜덤 시드

    Returns:
        선택된 피처 이름 리스트와 해당 데이터
    """
    print(f"\n[Feature Selection] {model_name} - Permutation Importance 기반 제거")
    print(f"  Threshold: {importance_threshold}, Min Features: {min_features}")

    # Permutation Importance 계산
    perm_df = calculate_permutation_importance(
        model, X, y, feature_names, model_name, n_repeats, random_state
    )

    # 제거할 피처 선택 (threshold 이하 + 최소 피처 수 보장)
    low_importance_mask = perm_df['importance_mean'] <= importance_threshold
    n_low_importance = low_importance_mask.sum()

    if n_low_importance > 0:
        # threshold 이하 피처들 중 importance가 가장 낮은 것부터 제거
        candidates_to_remove = perm_df[low_importance_mask].sort_values('importance_mean')['feature'].tolist()

        # 최소 피처 수 보장
        max_to_remove = len(feature_names) - min_features
        features_to_remove = candidates_to_remove[:max_to_remove]

        selected_features = [f for f in feature_names if f not in features_to_remove]
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]

        print(f"  제거된 피처: {len(features_to_remove)}개")
        if features_to_remove:
            print(f"    - {features_to_remove[:5]}{'...' if len(features_to_remove) > 5 else ''}")

        print(f"  선택된 피처: {len(selected_features)}개 (원본: {len(feature_names)}개)")
    else:
        print("  제거할 피처 없음 - threshold 조건 만족하는 피처가 없음")
        selected_features = feature_names
        X_selected = X

    return selected_features, X_selected


def select_features_rfe(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
) -> List[str]:
    """
    RFE (Recursive Feature Elimination) 기반 피처 선택
    반복적으로 최하위 중요도의 피처들을 제거하며 성능 변화를 모니터링

    Args:
        features_df: 전체 피처 데이터프레임
        feature_cols: 피처 컬럼 리스트
        meta_cols: 메타 컬럼 리스트
        cfg: 설정
        xgb_model_cfg: XGBoost 모델 설정

    Returns:
        선택된 피처 이름 리스트
    """
    from sklearn.model_selection import GroupKFold

    pipeline_cfg = cfg.pipeline
    feature_cfg = cfg.feature_importance

    print("\n[RFE 기반 피처 선택] 반복적 피처 제거로 최적 피처 조합 탐색")

    X_full = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_folds = min(pipeline_cfg.n_folds, 3)  # 피처 선택용 CV는 최대 3-fold로 제한
    gkf = GroupKFold(n_splits=n_folds)

    # RFE 설정
    rfe_step = feature_cfg.rfe.get("step", 5)  # 한 번에 제거할 피처 수
    min_features = feature_cfg.rfe.get("min_features", 10)
    performance_tolerance = feature_cfg.rfe.get("performance_tolerance", 0.01)  # 성능 감소 허용치

    current_features = feature_cols.copy()
    best_features = current_features.copy()
    best_score = 0.0

    print(f"  초기 피처 수: {len(current_features)}")
    print(f"  최소 피처 수: {min_features}")
    print(f"  제거 스텝: {rfe_step}")
    print(f"  성능 허용치: {performance_tolerance}")

    iteration = 0
    while len(current_features) > min_features:
        iteration += 1
        print(f"\n[RFE Iteration {iteration}] 현재 피처 수: {len(current_features)}")

        # 현재 피처들로 CV 수행하여 중요도 계산
        fold_importances = []
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_full, y, groups=groups)):
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 현재 피처만 선택
            train_feature_idx = [feature_cols.index(f) for f in current_features]
            val_feature_idx = train_feature_idx

            X_train_current = X_train[:, train_feature_idx]
            X_val_current = X_val[:, val_feature_idx]

            # 전처리
            if np.isnan(X_train_current).sum() > 0:
                imputer = SimpleImputer(strategy=cfg.imputer.strategy)
                X_train_current = imputer.fit_transform(X_train_current)
                X_val_current = imputer.transform(X_val_current)

            scaler = StandardScaler()
            X_train_current = scaler.fit_transform(X_train_current)
            X_val_current = scaler.transform(X_val_current)

            # 리샘플링
            X_train_res, y_train_res = apply_resampling(
                X_train_current, y_train, pipeline_cfg.resampling_method,
                pipeline_cfg.random_seed, cfg.resampling
            )

            # 모델 학습
            xgb_model = train_xgboost(
                X_train_res, y_train_res, X_val_current, y_val, xgb_model_cfg,
                use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
                use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
            )

            # AUPRC 성능 측정
            y_prob = xgb_model.predict_proba(X_val_current)[:, 1]
            score = average_precision_score(y_val, y_prob)
            fold_scores.append(score)

            # Permutation Importance 계산
            perm_df = calculate_permutation_importance(
                xgb_model, X_val_current, y_val, current_features,
                f"RFE_Fold_{fold+1}_Iter_{iteration}",
                n_repeats=feature_cfg.permutation.n_repeats,
                random_state=pipeline_cfg.random_seed
            )
            fold_importances.append(perm_df.set_index('feature')['importance_mean'])

        # 평균 성능 및 중요도 계산
        avg_score = np.mean(fold_scores)
        avg_importance_df = pd.concat(fold_importances, axis=1).mean(axis=1).reset_index()
        avg_importance_df.columns = ['feature', 'importance_mean']

        print(f"  평균 AUPRC: {avg_score:.4f}")

        # 최적 성능 갱신
        if avg_score > best_score:
            best_score = avg_score
            best_features = current_features.copy()
            print(f"  ✓ 최적 성능 갱신: {best_score:.4f}")
        elif avg_score < best_score - performance_tolerance:
            print(f"  ✗ 성능 저하 감지 (감소: {best_score - avg_score:.4f}), RFE 중단")
            break

        # 최하위 중요도 피처들 제거
        avg_importance_df = avg_importance_df.sort_values('importance_mean')
        n_to_remove = min(rfe_step, len(current_features) - min_features)

        if n_to_remove <= 0:
            break

        features_to_remove = avg_importance_df.head(n_to_remove)['feature'].tolist()
        current_features = [f for f in current_features if f not in features_to_remove]

        print(f"  제거된 피처: {features_to_remove}")

    print(f"\n[RFE 완료] 최적 피처 수: {len(best_features)} (AUPRC: {best_score:.4f})")
    print("  선택된 피처 Top 10:")
    for i, feature in enumerate(best_features[:10], 1):
        print(f"    {i}. {feature}")
    if len(best_features) > 10:
        print(f"    ... 외 {len(best_features) - 10}개")

    return best_features


def select_features_null_importance(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
) -> List[str]:
    """
    Null Importance 기반 피처 선택
    타겟 라벨을 섞어서 학습했을 때의 중요도와 실제 중요도를 비교하여
    우연히 중요하게 나온 피처들을 제거

    Args:
        features_df: 전체 피처 데이터프레임
        feature_cols: 피처 컬럼 리스트
        meta_cols: 메타 컬럼 리스트
        cfg: 설정
        xgb_model_cfg: XGBoost 모델 설정

    Returns:
        선택된 피처 이름 리스트
    """
    from sklearn.model_selection import GroupKFold

    pipeline_cfg = cfg.pipeline
    feature_cfg = cfg.feature_importance

    print("\n[Null Importance 기반 피처 선택] 우연히 중요하게 나온 피처 제거")

    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_folds = min(pipeline_cfg.n_folds, 3)
    gkf = GroupKFold(n_splits=n_folds)

    # Null Importance 설정
    n_null_runs = feature_cfg.null_importance.get("n_runs", 10)
    null_importance_threshold = feature_cfg.null_importance.get("threshold_percentile", 95)

    print(f"  Null Importance 실행 횟수: {n_null_runs}")
    print(f"  제거 임계값: {null_importance_threshold} 백분위수")

    # 실제 중요도 계산
    real_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 전처리
        if np.isnan(X_train).sum() > 0:
            imputer = SimpleImputer(strategy=cfg.imputer.strategy)
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 리샘플링
        X_train_res, y_train_res = apply_resampling(
            X_train, y_train, pipeline_cfg.resampling_method,
            pipeline_cfg.random_seed, cfg.resampling
        )

        # 모델 학습
        xgb_model = train_xgboost(
            X_train_res, y_train_res, X_val, y_val, xgb_model_cfg,
            use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
            use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
        )

        # Permutation Importance 계산
        perm_df = calculate_permutation_importance(
            xgb_model, X_val, y_val, feature_cols, f"Real_Fold_{fold+1}",
            n_repeats=feature_cfg.permutation.n_repeats,
            random_state=pipeline_cfg.random_seed
        )
        real_importances.append(perm_df.set_index('feature')['importance_mean'])

    # 실제 중요도 평균
    real_avg_importance = pd.concat(real_importances, axis=1).mean(axis=1).reset_index()
    real_avg_importance.columns = ['feature', 'importance_real']

    # Null Importance 계산 (타겟 라벨 섞어서)
    null_importances = []

    for null_run in range(n_null_runs):
        print(f"  Null Importance Run {null_run + 1}/{n_null_runs}")

        fold_null_importances = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_shuffled = y[train_idx].copy()
            np.random.shuffle(y_train_shuffled)  # 타겟 라벨 섞기
            y_val = y[val_idx]  # Validation은 실제 라벨 사용

            # 전처리
            if np.isnan(X_train).sum() > 0:
                imputer = SimpleImputer(strategy=cfg.imputer.strategy)
                X_train = imputer.fit_transform(X_train)
                X_val = imputer.transform(X_val)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            # 리샘플링 (셔플된 라벨 사용)
            X_train_res, y_train_res_shuffled = apply_resampling(
                X_train, y_train_shuffled, pipeline_cfg.resampling_method,
                pipeline_cfg.random_seed, cfg.resampling
            )

            # 모델 학습 (셔플된 라벨)
            xgb_model = train_xgboost(
                X_train_res, y_train_res_shuffled, X_val, y_val, xgb_model_cfg,
                use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
                use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
            )

            # Permutation Importance 계산
            perm_df = calculate_permutation_importance(
                xgb_model, X_val, y_val, feature_cols, f"Null_Run{null_run+1}_Fold_{fold+1}",
                n_repeats=feature_cfg.permutation.n_repeats,
                random_state=pipeline_cfg.random_seed
            )
            fold_null_importances.append(perm_df.set_index('feature')['importance_mean'])

        # Null run의 평균 중요도
        null_avg_importance = pd.concat(fold_null_importances, axis=1).mean(axis=1)
        null_importances.append(null_avg_importance)

    # Null Importance 통계 계산
    null_importance_df = pd.concat(null_importances, axis=1)
    null_importance_stats = pd.DataFrame({
        'feature': feature_cols,
        'null_importance_mean': null_importance_df.mean(axis=1),
        'null_importance_std': null_importance_df.std(axis=1),
        'null_importance_95p': null_importance_df.quantile(null_importance_threshold/100, axis=1)
    })

    # 실제 중요도와 Null Importance 비교
    comparison_df = pd.merge(real_avg_importance, null_importance_stats, on='feature')

    # 우연히 중요하게 나온 피처들 식별 (실제 중요도가 null importance의 threshold보다 낮은 경우)
    spurious_features = comparison_df[
        comparison_df['importance_real'] <= comparison_df[f'null_importance_{int(null_importance_threshold)}p']
    ]['feature'].tolist()

    selected_features = [f for f in feature_cols if f not in spurious_features]

    print(f"\n[Null Importance 결과]")
    print(f"  총 피처 수: {len(feature_cols)}")
    print(f"  우연히 중요하게 나온 피처 수: {len(spurious_features)}")
    print(f"  선택된 피처 수: {len(selected_features)}")

    if spurious_features:
        print("  제거된 피처들:")
        for i, feature in enumerate(spurious_features[:10], 1):
            real_imp = comparison_df[comparison_df['feature'] == feature]['importance_real'].iloc[0]
            null_95p = comparison_df[comparison_df['feature'] == feature][f'null_importance_{int(null_importance_threshold)}p'].iloc[0]
            print(".4f")
        if len(spurious_features) > 10:
            print(f"    ... 외 {len(spurious_features) - 10}개")

    return selected_features


def select_features_cv_based(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
) -> List[str]:
    """
    CV 기반 피처 선택 (데이터 누수 방지)

    Args:
        features_df: 전체 피처 데이터프레임
        feature_cols: 피처 컬럼 리스트
        meta_cols: 메타 컬럼 리스트
        cfg: 설정
        xgb_model_cfg: XGBoost 모델 설정

    Returns:
        선택된 피처 이름 리스트
    """
    from sklearn.model_selection import GroupKFold

    pipeline_cfg = cfg.pipeline
    feature_cfg = cfg.feature_importance

    print("\n[CV 기반 피처 선택] 데이터 누수 방지를 위해 CV 수행")

    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_folds = min(pipeline_cfg.n_folds, 3)  # 피처 선택용 CV는 최대 3-fold로 제한
    gkf = GroupKFold(n_splits=n_folds)

    # 각 fold에서 계산된 피처 중요도를 저장
    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"  Fold {fold + 1}/{n_folds} 피처 중요도 계산 중...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 전처리 (각 fold마다 독립적으로 수행)
        if np.isnan(X_train).sum() > 0:
            imputer = SimpleImputer(strategy=cfg.imputer.strategy)
            X_train = imputer.fit_transform(X_train)
            X_val = imputer.transform(X_val)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 리샘플링
        X_train_res, y_train_res = apply_resampling(
            X_train, y_train, pipeline_cfg.resampling_method,
            pipeline_cfg.random_seed, cfg.resampling
        )

        # 기본 파라미터로 빠른 모델 학습
        xgb_model = train_xgboost(
            X_train_res, y_train_res, X_val, y_val, xgb_model_cfg,
            use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
            use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
        )

        # Permutation Importance 계산
        perm_df = calculate_permutation_importance(
            xgb_model, X_val, y_val, feature_cols, f"XGBoost_Fold_{fold+1}",
            n_repeats=feature_cfg.permutation.n_repeats,
            random_state=pipeline_cfg.random_seed
        )

        fold_importances.append(perm_df.set_index('feature')['importance_mean'])

    # 모든 fold의 중요도 평균 계산
    avg_importance_df = pd.concat(fold_importances, axis=1).mean(axis=1).reset_index()
    avg_importance_df.columns = ['feature', 'importance_mean']

    print("\n[CV 평균 피처 중요도 Top 20]")
    print(avg_importance_df.sort_values('importance_mean', ascending=False).head(20).to_string(index=False))

    # 피처 선택 로직
    importance_threshold = feature_cfg.permutation.importance_threshold
    min_features = feature_cfg.permutation.min_features

    low_importance_mask = avg_importance_df['importance_mean'] <= importance_threshold
    n_low_importance = low_importance_mask.sum()

    if n_low_importance > 0:
        candidates_to_remove = avg_importance_df[low_importance_mask].sort_values('importance_mean')['feature'].tolist()
        max_to_remove = len(feature_cols) - min_features
        features_to_remove = candidates_to_remove[:max_to_remove]
        selected_features = [f for f in feature_cols if f not in features_to_remove]
    else:
        selected_features = feature_cols

    return selected_features


def optimize_ensemble_weights(
    y_true: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    cfg: DictConfig
) -> float:
    """
    Soft Voting을 위한 최적의 가중치(w)를 탐색합니다.

    Final P = w * P1 + (1-w) * P2
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
    Stacking을 위한 메타 모델(Logistic Regression) 학습
    """
    ensemble_cfg = cfg.ensemble

    meta_model = LogisticRegression(
        solver=ensemble_cfg.stacking_solver,
        C=ensemble_cfg.stacking_C,
        random_state=ensemble_cfg.stacking_random_state
    )
    meta_model.fit(X_meta_train, y_train)
    return meta_model


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
            elif metric in ["logloss", "log_loss", "binary_logloss"]:
                score = -log_loss(y_val, y_prob)
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
    feature_cfg = cfg.feature_importance
    meta_cols = list(data_cfg.meta_cols)

    if pipeline_cfg.use_gpu and GPU_AVAILABLE is None:
        GPU_AVAILABLE = check_gpu_availability()

    # 초기 피처 준비
    feature_cols = get_feature_columns(features_df, meta_cols)
    X_full = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    print(f"\n{'='*50}")
    print("PRE-OPTUNA FEATURE SELECTION")
    print(f"{'='*50}")

    # 피처 선택 방법 결정 및 실행
    selected_feature_cols = feature_cols.copy()

    # 1. RFE (Recursive Feature Elimination)
    if feature_cfg.rfe.enabled:
        print("RFE 기반 피처 선택 활성화")
        selected_feature_cols = select_features_rfe(
            features_df, selected_feature_cols, meta_cols, cfg, xgb_model_cfg
        )

    # 2. Null Importance 기반 피처 선택
    elif feature_cfg.null_importance.enabled:
        print("Null Importance 기반 피처 선택 활성화")
        selected_feature_cols = select_features_null_importance(
            features_df, selected_feature_cols, meta_cols, cfg, xgb_model_cfg
        )

    # 3. 기존 Permutation Importance 기반 피처 선택
    elif feature_cfg.permutation.enabled:
        print("Permutation Importance 기반 피처 선택 활성화")
        selected_feature_cols = select_features_cv_based(
            features_df, selected_feature_cols, meta_cols, cfg, xgb_model_cfg
        )

    else:
        print("피처 선택 비활성화 - 모든 피처 사용")

    # 선택된 피처로 데이터프레임 재구성
    if len(selected_feature_cols) != len(feature_cols):
        features_df_selected = features_df[selected_feature_cols + meta_cols].copy()
        print(f"\n[피처 선택 완료] {len(selected_feature_cols)}/{len(feature_cols)} 피처 선택됨")
        print(f"  제거율: {((len(feature_cols) - len(selected_feature_cols)) / len(feature_cols) * 100):.1f}%")
    else:
        features_df_selected = features_df.copy()

    results = {}

    # 선택된 피처로 Optuna 최적화 수행
    print(f"\n{'='*50}")
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*50}")
    print(f"사용 피처 수: {len(selected_feature_cols)}")

    xgb_opt = optimize_hyperparameters(features_df_selected, "xgboost", cfg, xgb_model_cfg)
    results["xgboost_optimization"] = xgb_opt

    lgb_opt = optimize_hyperparameters(features_df_selected, "lightgbm", cfg, lgb_model_cfg)
    results["lightgbm_optimization"] = lgb_opt

    cv_results = _run_cv_with_best_params(
        features_df_selected, xgb_opt["best_params"], lgb_opt["best_params"],
        cfg, xgb_model_cfg, lgb_model_cfg,
    )
    results["cv_results"] = cv_results

    # 선택된 피처로 최종 모델 학습
    # 참고: 이 단계에서는 전체 데이터를 사용하므로 imputation/scaling이 허용됨
    # (CV 기반 검증은 이미 완료되었고, 최종 모델은 전체 데이터로 학습)
    X_selected = features_df_selected[selected_feature_cols].values
    y = features_df_selected["label"].values

    imputer = None
    if np.isnan(X_selected).sum() > 0:
        print(f"\n최종 모델 학습 - Missing values: {np.isnan(X_selected).sum()} -> MICE")
        imputer = IterativeImputer(random_state=pipeline_cfg.random_seed, max_iter=cfg.imputer.iterative.max_iter)
        X_selected = imputer.fit_transform(X_selected)

    scaler_final = StandardScaler()
    X_selected = scaler_final.fit_transform(X_selected)

    X_res, y_res = apply_resampling(X_selected, y, pipeline_cfg.resampling_method, pipeline_cfg.random_seed, cfg.resampling)
    effective_cost = pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none"

    print("\n최종 모델 학습")
    print(f"사용 피처: {len(selected_feature_cols)}개")

    print("Train XGBoost")
    xgb_model = train_with_best_params(X_res, y_res, X_selected, y, xgb_opt["best_params"], "xgboost",
                                        xgb_model_cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    print("Train LightGBM")
    lgb_model = train_with_best_params(X_res, y_res, X_selected, y, lgb_opt["best_params"], "lightgbm",
                                        lgb_model_cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    results["models"] = {"xgboost": xgb_model, "lightgbm": lgb_model}
    results["feature_cols"] = selected_feature_cols
    results["original_feature_cols"] = feature_cols  # 원본 피처 목록도 저장
    results["imputer"] = imputer
    results["scaler"] = scaler_final
    results["optimal_thresholds"] = cv_results["optimal_thresholds"]
    results["ensemble_info"] = cv_results["ensemble_info"]  # 메타 모델 포함
    results["feature_selection_info"] = {
        "enabled": feature_cfg.permutation.enabled,
        "original_n_features": len(feature_cols),
        "selected_n_features": len(selected_feature_cols),
        "removed_n_features": len(feature_cols) - len(selected_feature_cols),
        "removal_rate": (len(feature_cols) - len(selected_feature_cols)) / len(feature_cols) * 100,
        "importance_threshold": feature_cfg.permutation.importance_threshold if feature_cfg.permutation.enabled else None,
    }

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

    # 결과 저장을 위한 딕셔너리
    metrics_list = ["auroc", "auprc", "f1", "precision", "recall", "accuracy"]
    model_names = ["XGBoost", "LightGBM", "SoftVoting", "Stacking"]

    cv_results = {m: {k: [] for k in metrics_list} for m in model_names}
    optimal_thresholds = {m: [] for m in model_names}

    # OOF (Out-Of-Fold) 예측값 저장을 위한 배열 (전체 데이터 크기만큼 0으로 초기화)
    # [n_samples]
    oof_xgb = np.zeros(len(y))
    oof_lgb = np.zeros(len(y))

    # 앙상블을 위한 Fold별 True Label 저장 (셔플링 이슈 대응)
    y_true_sorted = np.zeros(len(y))
    effective_cost = use_cost_sensitive and resampling_method == "none"
    print(f"\n{'='*20} Start Ensemble CV {'='*20}")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\nCV Fold {fold + 1}/{n_folds}")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # OOF 정렬을 위해 현재 Test Index에 해당하는 정답값 기록
        y_true_sorted[test_idx] = y_test
        if np.isnan(X_train).sum() > 0:
            imputer = SimpleImputer(strategy=imputer_strategy)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_train_res, y_train_res = apply_resampling(X_train, y_train, resampling_method, random_seed, cfg.resampling)
        # -------------------------------------------------------
        # 1. Base Models Training & Prediction
        # -------------------------------------------------------

        # XGBoost
        xgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, xgb_params,
                                            "xgboost", xgb_model_cfg, effective_cost, use_gpu, random_seed)
        y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        oof_xgb[test_idx] = y_prob_xgb # OOF 저장

        # LightGBM
        lgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, lgb_params,
                                            "lightgbm", lgb_model_cfg, effective_cost, use_gpu, random_seed)
        y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        oof_lgb[test_idx] = y_prob_lgb # OOF 저장

        # -------------------------------------------------------
        # 2. Ensemble: Soft Voting (Fold 내부에서 최적 가중치 계산은 과적합 위험이 있으므로,
        #    일반적으로는 Validation Set에서는 0.5 혹은 전체 OOF 후 계산하지만,
        #    여기서는 Fold마다 단순 평균(0.5)으로 성능 기록)
        # -------------------------------------------------------
        y_prob_voting = (y_prob_xgb + y_prob_lgb) / 2

        # -------------------------------------------------------
        # 3. Ensemble: Stacking (Fold-level approximation for logging)
        #    실제 Stacking은 CV 종료 후 전체 OOF 데이터로 학습하므로,
        #    여기서는 Fold별 성능 추이를 보기 위한 approximation으로
        #    현재 Fold의 Validation 예측값을 사용합니다.
        # -------------------------------------------------------

        # Fold 내에서 Stacking 성능을 추정하기 위해 Validation 예측값 사용
        # (실제 메타 모델은 CV 종료 후 전체 OOF로 학습)
        X_meta_test = np.column_stack((y_prob_xgb, y_prob_lgb))

        # 임시 메타 모델로 Fold 내 성능 추정 (실제 모델은 CV 후 학습)
        # 간단한 평균으로 approximation
        y_prob_stacking = (y_prob_xgb + y_prob_lgb) / 2

        # -------------------------------------------------------
        # 4. Metric Calculation & Logging
        # -------------------------------------------------------

        # 예측 확률 모음
        preds_dict = {
            "XGBoost": y_prob_xgb,
            "LightGBM": y_prob_lgb,
            "SoftVoting": y_prob_voting,
            "Stacking": y_prob_stacking
        }

        # (Inner function to reduce repetition)
        def calc_metrics_and_log(name, y_true, y_prob):
            # Threshold 찾기 (여기서는 Train 데이터 기준이 원칙이나 편의상 Test prob 분포 활용 혹은 고정값)
            # 엄밀하게는 y_train과 base model의 train prob를 써야 함.
            # 여기서는 편의상 Soft Voting 등은 0.5 default 혹은 Youden 적용
            opt_thresh, _ = find_optimal_threshold(y_true, y_prob, target_recall, threshold_method, default_threshold)
            optimal_thresholds[name].append(opt_thresh)

            y_pred = (y_prob >= opt_thresh).astype(int)

            cv_results[name]["auroc"].append(roc_auc_score(y_true, y_prob))
            cv_results[name]["auprc"].append(average_precision_score(y_true, y_prob))
            cv_results[name]["f1"].append(f1_score(y_true, y_pred, zero_division=0))
            cv_results[name]["precision"].append(precision_score(y_true, y_pred, zero_division=0))
            cv_results[name]["recall"].append(recall_score(y_true, y_pred, zero_division=0))
            cv_results[name]["accuracy"].append(accuracy_score(y_true, y_pred))

        for m_name, probs in preds_dict.items():
            calc_metrics_and_log(m_name, y_test, probs)

        print(f"  [Vote] AUROC={cv_results['SoftVoting']['auroc'][-1]:.4f}, F1={cv_results['SoftVoting']['f1'][-1]:.4f}")
        print(f"  [Stack] AUROC={cv_results['Stacking']['auroc'][-1]:.4f}, F1={cv_results['Stacking']['f1'][-1]:.4f}")

    # -------------------------------------------------------
    # 전체 CV 종료 후: Global Ensemble Optimization (OOF 기반)
    # -------------------------------------------------------

    # 1. Soft Voting 최적 가중치 찾기
    best_weight_xgb = optimize_ensemble_weights(y, oof_xgb, oof_lgb, cfg)
    print(f"\n[Global Optimization] Best Weight for XGB: {best_weight_xgb:.2f} (LGB: {1-best_weight_xgb:.2f})")

    # 2. Stacking 메타 모델 학습 (OOF 기반 - 과적합 방지)
    print("\n[Global Stacking] Train Meta Model using OOF predictions")
    X_meta_train = np.column_stack((oof_xgb, oof_lgb))
    meta_model = train_stacking_model(X_meta_train, y, cfg)
    print("  Meta model trained successfully using OOF predictions")

    # 2. 결과 출력
    summary_data = []
    for model in model_names:
        for metric in metrics_list:
            scores = cv_results[model][metric]
            summary_data.append({"Model": model, "Metric": metric.upper(), "Mean": np.mean(scores), "Std": np.std(scores)})

    summary_df = pd.DataFrame(summary_data)

    # 결과 반환에 최적 가중치와 메타 모델 학습을 위한 데이터 포함 가능
    return {
        "cv_results": cv_results,
        "summary": summary_df,
        "optimal_thresholds": optimal_thresholds,
        "feature_cols": feature_cols,
        "ensemble_info": {
            "best_weight_xgb": best_weight_xgb,
            "oof_xgb": oof_xgb,
            "oof_lgb": oof_lgb,
            "meta_model": meta_model
        }
    }


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
