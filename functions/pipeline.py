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
    pipeline_cfg = cfg.pipeline
    data_cfg = cfg.data
    feature_cfg = cfg.feature_importance
    meta_cols = list(data_cfg.meta_cols)

    feature_cols = get_feature_columns(features_df, meta_cols)
    
    feature_cols_filtered = [col for col in feature_cols if '_count' not in col]
    n_excluded = len(feature_cols) - len(feature_cols_filtered)
    if n_excluded > 0:
        print(f"\nExcluded {n_excluded}")
        print(f"Features: {len(feature_cols)} -> {len(feature_cols_filtered)}")
    feature_cols = feature_cols_filtered
    
    X_full = features_df[feature_cols].values
    y = features_df["label"].values
    groups = features_df["subject_id"].values

    print(f"\n{'='*50}")
    print("PRE-OPTUNA FEATURE SELECTION")
    print(f"{'='*50}")

    selected_feature_cols = feature_cols.copy()

    if feature_cfg.rfe.enabled:
        print("RFE based feature selection enabled")
        selected_feature_cols = select_features_rfe(
            features_df, selected_feature_cols, meta_cols, cfg, xgb_model_cfg
        )

    elif feature_cfg.null_importance.enabled:
        print("Null Importance based feature selection enabled")
        selected_feature_cols = select_features_null_importance(
            features_df, selected_feature_cols, meta_cols, cfg, xgb_model_cfg
        )

    elif feature_cfg.permutation.enabled:
        print("Permutation Importance based feature selection enabled")
        selected_feature_cols = select_features_cv_based(
            features_df, selected_feature_cols, meta_cols, cfg, xgb_model_cfg
        )

    else:
        print("Feature selection disabled, using all features")

    if len(selected_feature_cols) != len(feature_cols):
        features_df_selected = features_df[selected_feature_cols + meta_cols].copy()
        print(f"\nFeature selection complete: {len(selected_feature_cols)}/{len(feature_cols)} features selected")
        print(f"Removal rate: {((len(feature_cols) - len(selected_feature_cols)) / len(feature_cols) * 100):.1f}%")
    else:
        features_df_selected = features_df.copy()

    results = {}

    print(f"Selected features: {len(selected_feature_cols)}")

    xgb_opt = optimize_hyperparameters(features_df_selected, "xgboost", cfg, xgb_model_cfg)
    results["xgboost_optimization"] = xgb_opt

    lgb_opt = optimize_hyperparameters(features_df_selected, "lightgbm", cfg, lgb_model_cfg)
    results["lightgbm_optimization"] = lgb_opt

    cv_results = _run_cv_with_best_params(
        features_df_selected, xgb_opt["best_params"], lgb_opt["best_params"],
        cfg, xgb_model_cfg, lgb_model_cfg,
    )
    results["cv_results"] = cv_results

    X_selected = features_df_selected[selected_feature_cols].values
    y = features_df_selected["label"].values

    imputer = None
    if np.isnan(X_selected).sum() > 0:
        print(f"\nFinal model training - Missing values: {np.isnan(X_selected).sum()} -> MICE")
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imputer = IterativeImputer(random_state=pipeline_cfg.random_seed, max_iter=cfg.imputer.iterative.max_iter)
        X_selected = imputer.fit_transform(X_selected)

    scaler_final = apply_preprocessing(X_selected, X_selected)[2]

    X_res, y_res = apply_resampling(X_selected, y, pipeline_cfg.resampling_method, pipeline_cfg.random_seed, cfg.resampling)
    effective_cost = pipeline_cfg.use_cost_sensitive and pipeline_cfg.resampling_method == "none"

    print("\nFinal model training")
    print(f"Features used: {len(selected_feature_cols)}")

    print("Train XGBoost")
    xgb_model = train_with_best_params(X_res, y_res, X_selected, y, xgb_opt["best_params"], "xgboost",
                                       xgb_model_cfg, cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    print("Train LightGBM")
    lgb_model = train_with_best_params(X_res, y_res, X_selected, y, lgb_opt["best_params"], "lightgbm",
                                       lgb_model_cfg, cfg, effective_cost, pipeline_cfg.use_gpu, pipeline_cfg.random_seed)

    results["models"] = {"xgboost": xgb_model, "lightgbm": lgb_model}
    results["feature_cols"] = selected_feature_cols
    results["original_feature_cols"] = feature_cols
    results["imputer"] = imputer
    results["scaler"] = scaler_final
    results["optimal_thresholds"] = cv_results["optimal_thresholds"]
    results["ensemble_info"] = cv_results["ensemble_info"]
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
    xgb_model_cfg = OmegaConf.load("conf/model/xgboost.yaml")
    lgb_model_cfg = OmegaConf.load("conf/model/lightgbm.yaml")

    pipeline_cfg = cfg.pipeline
    output_cfg = cfg.output
    data_cfg = cfg.data
    meta_cols = list(data_cfg.meta_cols)
    top_n = cfg.feature_importance.top_n

    print(f"{pipeline_cfg.n_folds}-Fold Cross-Validation")

    print(f"Data path: {data_cfg.features_path}")
    print(f"CV Folds: {pipeline_cfg.n_folds}")
    print(f"Resampling: {pipeline_cfg.resampling_method}")
    print(f"Cost-sensitive: {pipeline_cfg.use_cost_sensitive}")
    print(f"GPU: {pipeline_cfg.use_gpu}")
    print(f"Target Recall: {pipeline_cfg.target_recall}")
    print(f"Optuna trials: {cfg.optuna.n_trials}")
    print(f"Optimization metric: {cfg.optuna.metric}")

    if pipeline_cfg.use_gpu:
        gpu_status = check_gpu_availability()
        set_gpu_status(gpu_status)
    else:
        set_gpu_status({"xgboost": False, "lightgbm": False})

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
        print("\nModels saved:")
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
