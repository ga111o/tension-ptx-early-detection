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
    if hasattr(model, 'feature_importance'):
        importance_values = model.feature_importance(importance_type='gain')

    elif hasattr(model, 'get_score'):
        importance_dict = model.get_score(importance_type='gain')

        importance_values = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            importance_values[i] = importance_dict.get(f'f{i}', 0.0)

    else:
        if hasattr(model, 'booster_'):
            importance_values = model.booster_.feature_importance(importance_type='gain')
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
    print(f"\nPermutation Importance {model_name} - calculating n_repeats={n_repeats}")

    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='roc_auc'
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
    }).sort_values('importance_mean', ascending=False)

    print(f"Complete - total {len(feature_names)} features")
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
    print(f"\nFeature Selection {model_name} - Permutation Importance based removal")
    print(f"Threshold: {importance_threshold}, Min Features: {min_features}")

    perm_df = calculate_permutation_importance(
        model, X, y, feature_names, model_name, n_repeats, random_state
    )

    low_importance_mask = perm_df['importance_mean'] <= importance_threshold
    n_low_importance = low_importance_mask.sum()

    if n_low_importance > 0:
        candidates_to_remove = perm_df[low_importance_mask].sort_values('importance_mean')['feature'].tolist()

        max_to_remove = len(feature_names) - min_features
        features_to_remove = candidates_to_remove[:max_to_remove]

        selected_features = [f for f in feature_names if f not in features_to_remove]
        selected_indices = [feature_names.index(f) for f in selected_features]
        X_selected = X[:, selected_indices]

        print(f"Removed features: {len(features_to_remove)}")
        if features_to_remove:
            print(f"  - {features_to_remove[:5]}{'...' if len(features_to_remove) > 5 else ''}")

        print(f"Selected features: {len(selected_features)} from {len(feature_names)}")
    else:
        print("No features to remove - no features meet threshold condition")
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
    pipeline_cfg = cfg.pipeline
    feature_cfg = cfg.feature_importance

    X_full = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_folds = min(pipeline_cfg.n_folds, 3)
    gkf = GroupKFold(n_splits=n_folds)

    rfe_step = feature_cfg.rfe.get("step", 5)
    min_features = feature_cfg.rfe.get("min_features", 10)
    performance_tolerance = feature_cfg.rfe.get("performance_tolerance", 0.01)

    current_features = feature_cols.copy()
    best_features = current_features.copy()
    best_score = 0.0

    print(f"Initial features: {len(current_features)}")
    print(f"Minimum features: {min_features}")
    print(f"Removal step: {rfe_step}")
    print(f"Performance tolerance: {performance_tolerance}")

    iteration = 0
    while len(current_features) > min_features:
        iteration += 1
        print(f"\nRFE Iteration {iteration} Current features: {len(current_features)}")

        fold_importances = []
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_full, y, groups=groups)):
            X_train, X_val = X_full[train_idx], X_full[val_idx]
            y_train, y_val = y[train_idx], y_val[val_idx]

            train_feature_idx = [feature_cols.index(f) for f in current_features]
            val_feature_idx = train_feature_idx

            X_train_current = X_train[:, train_feature_idx]
            X_val_current = X_val[:, val_feature_idx]

            X_train_current, X_val_current, _, _ = apply_preprocessing(X_train_current, X_val_current)

            X_train_res, y_train_res = apply_resampling(
                X_train_current, y_train, pipeline_cfg.resampling_method,
                pipeline_cfg.random_seed, cfg.resampling
            )

            xgb_model = train_xgboost(
                X_train_res, y_train_res, X_val_current, y_val, xgb_model_cfg,
                use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
                use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
            )

            y_prob = xgb_model.predict_proba(X_val_current)[:, 1]
            score = average_precision_score(y_val, y_prob)
            fold_scores.append(score)

            perm_df = calculate_permutation_importance(
                xgb_model, X_val_current, y_val, current_features,
                f"RFE_Fold_{fold+1}_Iter_{iteration}",
                n_repeats=feature_cfg.permutation.n_repeats,
                random_state=pipeline_cfg.random_seed
            )
            fold_importances.append(perm_df.set_index('feature')['importance_mean'])

        avg_score = np.mean(fold_scores)
        avg_importance_df = pd.concat(fold_importances, axis=1).mean(axis=1).reset_index()
        avg_importance_df.columns = ['feature', 'importance_mean']

        print(f"Average AUPRC: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_features = current_features.copy()
            print(f"Best performance updated: {best_score:.4f}")
        elif avg_score < best_score - performance_tolerance:
            print(f"Performance degradation detected decrease: {best_score - avg_score:.4f}, RFE stopped")
            break

        avg_importance_df = avg_importance_df.sort_values('importance_mean')
        n_to_remove = min(rfe_step, len(current_features) - min_features)

        if n_to_remove <= 0:
            break

        features_to_remove = avg_importance_df.head(n_to_remove)['feature'].tolist()
        current_features = [f for f in current_features if f not in features_to_remove]

        print(f"Removed features: {features_to_remove}")

    print(f"\nRFE complete - optimal features: {len(best_features)} AUPRC: {best_score:.4f}")
    print("Selected features Top 10:")
    for i, feature in enumerate(best_features[:10], 1):
        print(f"  {i}. {feature}")
    if len(best_features) > 10:
        print(f"  and {len(best_features) - 10} more")

    return best_features


def select_features_null_importance(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    cfg,
    xgb_model_cfg,
) -> List[str]:
    pipeline_cfg = cfg.pipeline
    feature_cfg = cfg.feature_importance

    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_folds = min(pipeline_cfg.n_folds, 3)
    gkf = GroupKFold(n_splits=n_folds)

    n_null_runs = feature_cfg.null_importance.get("n_runs", 10)
    null_importance_threshold = feature_cfg.null_importance.get("threshold_percentile", 95)

    print(f"Null Importance runs: {n_null_runs}")
    print(f"Removal threshold: {null_importance_threshold} percentile")

    real_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train, X_val, _, _ = apply_preprocessing(X_train, X_val)

        X_train_res, y_train_res = apply_resampling(
            X_train, y_train, pipeline_cfg.resampling_method,
            pipeline_cfg.random_seed, cfg.resampling
        )

        xgb_model = train_xgboost(
            X_train_res, y_train_res, X_val, y_val, xgb_model_cfg,
            use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
            use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
        )

        perm_df = calculate_permutation_importance(
            xgb_model, X_val, y_val, feature_cols, f"Real_Fold_{fold+1}",
            n_repeats=feature_cfg.permutation.n_repeats,
            random_state=pipeline_cfg.random_seed
        )
        real_importances.append(perm_df.set_index('feature')['importance_mean'])

    real_avg_importance = pd.concat(real_importances, axis=1).mean(axis=1).reset_index()
    real_avg_importance.columns = ['feature', 'importance_real']

    null_importances = []

    for null_run in range(n_null_runs):
        print(f"Null Importance Run {null_run + 1}/{n_null_runs}")

        fold_null_importances = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train_shuffled = y[train_idx].copy()
            np.random.shuffle(y_train_shuffled)
            y_val = y[val_idx]

            X_train, X_val, _, _ = apply_preprocessing(X_train, X_val)

            X_train_res, y_train_res_shuffled = apply_resampling(
                X_train, y_train_shuffled, pipeline_cfg.resampling_method,
                pipeline_cfg.random_seed, cfg.resampling
            )

            xgb_model = train_xgboost(
                X_train_res, y_train_res_shuffled, X_val, y_val, xgb_model_cfg,
                use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
                use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
            )

            perm_df = calculate_permutation_importance(
                xgb_model, X_val, y_val, feature_cols, f"Null_Run{null_run+1}_Fold_{fold+1}",
                n_repeats=feature_cfg.permutation.n_repeats,
                random_state=pipeline_cfg.random_seed
            )
            fold_null_importances.append(perm_df.set_index('feature')['importance_mean'])

        null_avg_importance = pd.concat(fold_null_importances, axis=1).mean(axis=1)
        null_importances.append(null_avg_importance)

    null_importance_df = pd.concat(null_importances, axis=1)
    null_importance_stats = pd.DataFrame({
        'feature': feature_cols,
        'null_importance_mean': null_importance_df.mean(axis=1),
        'null_importance_std': null_importance_df.std(axis=1),
        'null_importance_95p': null_importance_df.quantile(null_importance_threshold/100, axis=1)
    })

    comparison_df = pd.merge(real_avg_importance, null_importance_stats, on='feature')

    spurious_features = comparison_df[
        comparison_df['importance_real'] <= comparison_df[f'null_importance_{int(null_importance_threshold)}p']
    ]['feature'].tolist()

    selected_features = [f for f in feature_cols if f not in spurious_features]

    print(f"\nNull Importance results")
    print(f"Total features: {len(feature_cols)}")
    print(f"Spurious features: {len(spurious_features)}")
    print(f"Selected features: {len(selected_features)}")

    if spurious_features:
        print("Removed features:")
        for i, feature in enumerate(spurious_features[:10], 1):
            real_imp = comparison_df[comparison_df['feature'] == feature]['importance_real'].iloc[0]
            null_95p = comparison_df[comparison_df['feature'] == feature][f'null_importance_{int(null_importance_threshold)}p'].iloc[0]
            print(f"  {i}. {feature}: real={real_imp:.4f}, null_95p={null_95p:.4f}")
        if len(spurious_features) > 10:
            print(f"  and {len(spurious_features) - 10} more")

    return selected_features


def select_features_cv_based(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    meta_cols: List[str],
    cfg,
    xgb_model_cfg,
) -> List[str]:
    pipeline_cfg = cfg.pipeline
    feature_cfg = cfg.feature_importance

    print("\nCV based feature selection - prevent data leakage")

    X = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    n_folds = min(pipeline_cfg.n_folds, 3)
    gkf = GroupKFold(n_splits=n_folds)

    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"Fold {fold + 1}/{n_folds} calculating feature importance")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train, X_val, _, _ = apply_preprocessing(X_train, X_val)

        X_train_res, y_train_res = apply_resampling(
            X_train, y_train, pipeline_cfg.resampling_method,
            pipeline_cfg.random_seed, cfg.resampling
        )

        xgb_model = train_xgboost(
            X_train_res, y_train_res, X_val, y_val, xgb_model_cfg,
            use_cost_sensitive=pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none",
            use_gpu=pipeline_cfg.use_gpu, random_seed=pipeline_cfg.random_seed
        )

        perm_df = calculate_permutation_importance(
            xgb_model, X_val, y_val, feature_cols, f"XGBoost_Fold_{fold+1}",
            n_repeats=feature_cfg.permutation.n_repeats,
            random_state=pipeline_cfg.random_seed
        )

        fold_importances.append(perm_df.set_index('feature')['importance_mean'])

    avg_importance_df = pd.concat(fold_importances, axis=1).mean(axis=1).reset_index()
    avg_importance_df.columns = ['feature', 'importance_mean']

    print(avg_importance_df.sort_values('importance_mean', ascending=False).head(20).to_string(index=False))
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
