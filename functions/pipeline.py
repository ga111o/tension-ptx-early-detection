"""
Main ML pipeline utilities
"""

import json
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from .gpu_utils import check_gpu_availability, set_gpu_status
from .data_utils import load_preprocessed_features, get_feature_columns
from .preprocessing_utils import apply_preprocessing, apply_resampling, preprocess_for_training
from .feature_utils import (
    select_features_rfe, select_features_null_importance, select_features_cv_based,
    get_feature_importance
)
from .optuna_utils import optimize_hyperparameters
from .model_utils import train_with_best_params
from .cv_utils import _run_cv_with_best_params


def run_optimization_pipeline(
    features_df: pd.DataFrame,
    cfg: DictConfig,
    xgb_model_cfg: DictConfig,
    lgb_model_cfg: DictConfig,
) -> dict:
    """
    Run the complete optimization pipeline including feature selection and hyperparameter optimization
    """
    pipeline_cfg = cfg.pipeline
    data_cfg = cfg.data
    feature_cfg = cfg.feature_importance
    meta_cols = list(data_cfg.meta_cols)

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
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=pipeline_cfg.random_seed, max_iter=cfg.imputer.iterative.max_iter)
        X_selected = imputer.fit_transform(X_selected)

    scaler_final = apply_preprocessing(X_selected, X_selected)[2]  # Only get scaler

    X_res, y_res = apply_resampling(X_selected, y, pipeline_cfg.resampling_method, pipeline_cfg.random_seed, cfg.resampling)
    effective_cost = pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none"

    print("\n최종 모델 학습")
    print(f"사용 피처: {len(selected_feature_cols)}개")

    print("Train XGBoost")
    xgb_model = train_with_best_params(X_res, y_res, X_selected, y, xgb_opt["best_params"], "xgboost",
                                       xgb_model_cfg, cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    print("Train LightGBM")
    lgb_model = train_with_best_params(X_res, y_res, X_selected, y, lgb_opt["best_params"], "lightgbm",
                                       lgb_model_cfg, cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

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


def run_pipeline(cfg: DictConfig) -> dict | None:
    """
    Run the complete ML pipeline
    """
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
