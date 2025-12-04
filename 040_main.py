"""
Main ML pipeline execution script
Refactored version with modular components
"""

import warnings
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

# Import modular components
from functions.pipeline import run_pipeline

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    import os
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    logger.info(f"Original CWD: {original_cwd}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    results = run_pipeline(cfg)
        
    if "cv_results" in results and "summary" in results["cv_results"]:
        summary_df = results["cv_results"]["summary"]
        for model in summary_df["Model"].unique():
            model_metrics = summary_df[summary_df["Model"] == model]
            logger.info(f"\n{model}:")
            for _, row in model_metrics.iterrows():
                logger.info(f"  {row['Metric']}: {row['Mean']:.4f} (±{row['Std']:.4f})")
    
    if "best_params" in results:
        for model_name, params in results["best_params"].items():
            logger.info(f"\n{model_name}:")
            for param, value in params.items():
                if isinstance(value, float):
                    logger.info(f"  {param}: {value:.6f}")
                else:
                    logger.info(f"  {param}: {value}")
    
    if "feature_selection_info" in results:
        fs_info = results["feature_selection_info"]
        if fs_info.get("enabled"):
            logger.info(f"  Original features: {fs_info['original_n_features']}")
            logger.info(f"  Selected features: {fs_info['selected_n_features']}")
            logger.info(f"  Removal rate: {fs_info['removal_rate']:.1f}%")
    
    if "ensemble_info" in results:
        ens_info = results["ensemble_info"]
        if "best_weight_xgb" in ens_info:
            logger.info(f"  XGBoost: {ens_info['best_weight_xgb']:.2f}")
            logger.info(f"  LightGBM: {1 - ens_info['best_weight_xgb']:.2f}")
    
    if "feature_importance" in results:
        for model_name, importance_df in results["feature_importance"].items():
            if importance_df is not None and len(importance_df) > 0:
                logger.info(f"\n{model_name.upper()} Total {len(importance_df)} features")
                sorted_features = importance_df.sort_values("importance", ascending=False)
                for idx, row in sorted_features.iterrows():
                    logger.info(f"  {row['feature']:30s} {row['importance']:.6f}")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
