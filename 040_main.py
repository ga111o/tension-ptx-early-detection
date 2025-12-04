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
    
    # 깔끔한 결과 요약 출력
    logger.info("\n" + "="*60)
    logger.info("파이프라인 실행 완료")
    logger.info("="*60)
    
    # CV 결과 요약
    if "cv_results" in results and "summary" in results["cv_results"]:
        logger.info("\n[CV 결과 요약]")
        summary_df = results["cv_results"]["summary"]
        for model in summary_df["Model"].unique():
            model_metrics = summary_df[summary_df["Model"] == model]
            logger.info(f"\n{model}:")
            for _, row in model_metrics.iterrows():
                logger.info(f"  {row['Metric']}: {row['Mean']:.4f} (±{row['Std']:.4f})")
    
    # 최적 하이퍼파라미터
    if "best_params" in results:
        logger.info("\n[최적 하이퍼파라미터]")
        for model_name, params in results["best_params"].items():
            logger.info(f"\n{model_name}:")
            for param, value in params.items():
                if isinstance(value, float):
                    logger.info(f"  {param}: {value:.6f}")
                else:
                    logger.info(f"  {param}: {value}")
    
    # 피처 선택 정보
    if "feature_selection_info" in results:
        fs_info = results["feature_selection_info"]
        if fs_info.get("enabled"):
            logger.info("\n[피처 선택 정보]")
            logger.info(f"  원본 피처 수: {fs_info['original_n_features']}")
            logger.info(f"  선택된 피처 수: {fs_info['selected_n_features']}")
            logger.info(f"  제거 비율: {fs_info['removal_rate']:.1f}%")
    
    # 앙상블 정보
    if "ensemble_info" in results:
        ens_info = results["ensemble_info"]
        if "best_weight_xgb" in ens_info:
            logger.info("\n[앙상블 가중치]")
            logger.info(f"  XGBoost: {ens_info['best_weight_xgb']:.2f}")
            logger.info(f"  LightGBM: {1 - ens_info['best_weight_xgb']:.2f}")
    
    # 피처 중요도 (전체)
    if "feature_importance" in results:
        logger.info("\n[피처 중요도]")
        for model_name, importance_df in results["feature_importance"].items():
            if importance_df is not None and len(importance_df) > 0:
                logger.info(f"\n{model_name.upper()} (총 {len(importance_df)}개 피처):")
                sorted_features = importance_df.sort_values("importance", ascending=False)
                for idx, row in sorted_features.iterrows():
                    logger.info(f"  {row['feature']:30s} {row['importance']:.6f}")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
