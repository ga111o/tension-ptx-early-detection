"""
Feature importance and selection utilities
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from .model_utils import train_xgboost
from .preprocessing_utils import apply_preprocessing, apply_resampling


def get_feature_importance(model, feature_names: List[str], model_name: str, top_n: int) -> pd.DataFrame:
    """
    Extract feature importance from trained model

    Args:
        model: Trained model (XGBoost, LightGBM, etc.)
        feature_names: List of feature names
        model_name: Name of the model for logging
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importance scores
    """
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
    Calculate permutation importance

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: Feature names
        model_name: Model name for logging
        n_repeats: Number of permutation repeats
        random_state: Random state

    Returns:
        DataFrame with permutation importance scores
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
    Remove features with low permutation importance

    Args:
        model: Trained model
        X: Feature matrix
        y: Target vector
        feature_names: Feature names
        model_name: Model name
        importance_threshold: Threshold for feature removal
        min_features: Minimum number of features to keep
        n_repeats: Number of permutation repeats
        random_state: Random state

    Returns:
        Selected feature names and corresponding data
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
    cfg,
    xgb_model_cfg,
) -> List[str]:
    """
    Recursive Feature Elimination based feature selection

    Args:
        features_df: Full feature DataFrame
        feature_cols: Feature column names
        meta_cols: Metadata column names
        cfg: Configuration
        xgb_model_cfg: XGBoost model configuration

    Returns:
        Selected feature names
    """
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
            y_train, y_val = y[train_idx], y_val[val_idx]

            # 현재 피처만 선택
            train_feature_idx = [feature_cols.index(f) for f in current_features]
            val_feature_idx = train_feature_idx

            X_train_current = X_train[:, train_feature_idx]
            X_val_current = X_val[:, val_feature_idx]

            # 전처리
            X_train_current, X_val_current, _, _ = apply_preprocessing(X_train_current, X_val_current)

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
    cfg,
    xgb_model_cfg,
) -> List[str]:
    """
    Null Importance based feature selection

    Args:
        features_df: Full feature DataFrame
        feature_cols: Feature column names
        meta_cols: Metadata column names
        cfg: Configuration
        xgb_model_cfg: XGBoost model configuration

    Returns:
        Selected feature names
    """
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
        X_train, X_val, _, _ = apply_preprocessing(X_train, X_val)

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
            X_train, X_val, _, _ = apply_preprocessing(X_train, X_val)

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
    cfg,
    xgb_model_cfg,
) -> List[str]:
    """
    CV-based feature selection (data leakage prevention)

    Args:
        features_df: Full feature DataFrame
        feature_cols: Feature column names
        meta_cols: Metadata column names
        cfg: Configuration
        xgb_model_cfg: XGBoost model configuration

    Returns:
        Selected feature names
    """
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
        X_train, X_val, _, _ = apply_preprocessing(X_train, X_val)

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
