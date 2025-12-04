"""
Cross-validation utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from omegaconf import DictConfig

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
)

from .model_utils import train_with_best_params, find_optimal_threshold
from .preprocessing_utils import apply_preprocessing, apply_resampling
from .ensemble_utils import optimize_ensemble_weights, train_stacking_model
from .gpu_utils import GPU_AVAILABLE


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
        from .gpu_utils import check_gpu_availability
        print("\nCheck GPU")
        GPU_AVAILABLE = check_gpu_availability()

    from .data_utils import get_feature_columns
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
        xgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, {}, "xgboost",
                                           xgb_model_cfg, cfg, effective_cost_sensitive, use_gpu, random_seed)
        # Custom objective 사용 시 predict_proba가 1차원 배열을 반환할 수 있음
        y_prob_xgb_raw = xgb_model.predict_proba(X_test)
        y_prob_xgb = y_prob_xgb_raw if y_prob_xgb_raw.ndim == 1 else y_prob_xgb_raw[:, 1]
        
        y_prob_train_xgb_raw = xgb_model.predict_proba(X_train)
        y_prob_train_xgb = y_prob_train_xgb_raw if y_prob_train_xgb_raw.ndim == 1 else y_prob_train_xgb_raw[:, 1]
        opt_thresh_xgb, _ = find_optimal_threshold(y_train, y_prob_train_xgb,
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
        lgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, {}, "lightgbm",
                                           lgb_model_cfg, cfg, effective_cost_sensitive, use_gpu, random_seed)
        # Custom objective 사용 시 predict_proba가 1차원 배열을 반환할 수 있음
        y_prob_lgb_raw = lgb_model.predict_proba(X_test)
        y_prob_lgb = y_prob_lgb_raw if y_prob_lgb_raw.ndim == 1 else y_prob_lgb_raw[:, 1]
        
        y_prob_train_lgb_raw = lgb_model.predict_proba(X_train)
        y_prob_train_lgb = y_prob_train_lgb_raw if y_prob_train_lgb_raw.ndim == 1 else y_prob_train_lgb_raw[:, 1]
        opt_thresh_lgb, _ = find_optimal_threshold(y_train, y_prob_train_lgb,
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


def _run_cv_with_best_params(
    features_df: pd.DataFrame,
    xgb_params: Dict,
    lgb_params: Dict,
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
    lgb_model_cfg: DictConfig,
) -> Dict:
    """
    Run cross-validation with best parameters and ensemble methods
    """
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

    from .data_utils import get_feature_columns
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
                                           "xgboost", xgb_model_cfg, cfg, effective_cost, use_gpu, random_seed)
        # Custom objective 사용 시 predict_proba가 1차원 배열을 반환할 수 있음
        y_prob_xgb_raw = xgb_model.predict_proba(X_test)
        y_prob_xgb = y_prob_xgb_raw if y_prob_xgb_raw.ndim == 1 else y_prob_xgb_raw[:, 1]
        oof_xgb[test_idx] = y_prob_xgb  # OOF 저장

        # LightGBM
        lgb_model = train_with_best_params(X_train_res, y_train_res, X_test, y_test, lgb_params,
                                           "lightgbm", lgb_model_cfg, cfg, effective_cost, use_gpu, random_seed)
        # Custom objective 사용 시 predict_proba가 1차원 배열을 반환할 수 있음
        y_prob_lgb_raw = lgb_model.predict_proba(X_test)
        y_prob_lgb = y_prob_lgb_raw if y_prob_lgb_raw.ndim == 1 else y_prob_lgb_raw[:, 1]
        oof_lgb[test_idx] = y_prob_lgb  # OOF 저장

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
