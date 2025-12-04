"""
GPU availability checking utilities for XGBoost and LightGBM
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb

GPU_AVAILABLE: dict[str, bool] | None = None


def check_gpu_availability() -> dict[str, bool]:
    """
    Check GPU availability for XGBoost and LightGBM

    Returns:
        Dictionary with GPU availability status for each library
    """
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


def set_gpu_status(status: dict[str, bool]) -> None:
    """
    Set the global GPU availability status

    Args:
        status: GPU availability dictionary
    """
    global GPU_AVAILABLE
    GPU_AVAILABLE = status


def get_gpu_status() -> dict[str, bool] | None:
    """
    Get the current GPU availability status

    Returns:
        Current GPU status dictionary or None if not set
    """
    return GPU_AVAILABLE
