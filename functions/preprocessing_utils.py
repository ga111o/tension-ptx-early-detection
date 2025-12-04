import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from imblearn.over_sampling import ADASYN, SMOTE
from omegaconf import DictConfig


def apply_resampling(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    random_seed: int,
    resampling_cfg: DictConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "none":
        return X, y

    print(f"Resampling: {method.upper()} before: Pos={sum(y)}, Neg={len(y) - sum(y)}")

    if method == "smote":
        k_neighbors = resampling_cfg.smote.k_neighbors
        sampler = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
    elif method == "adasyn":
        n_neighbors = resampling_cfg.adasyn.n_neighbors
        sampler = ADASYN(random_state=random_seed, n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Unknown resampling method: {method}")

    X_res, y_res = sampler.fit_resample(X, y)
    print(f"after: Pos={sum(y_res)}, Neg={len(y_res) - sum(y_res)}")
    return X_res, y_res


def apply_preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    imputer_strategy: str = "mean",
    use_iterative_imputer: bool = False,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, SimpleImputer | IterativeImputer]:
    if use_iterative_imputer:
        imputer = IterativeImputer(random_state=42, max_iter=10)
    else:
        imputer = SimpleImputer(strategy=imputer_strategy)

    if np.isnan(X_train).sum() > 0:
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler, imputer


def preprocess_for_training(
    X: np.ndarray,
    y: np.ndarray,
    imputer_strategy: str = "mean",
    use_iterative_imputer: bool = False,
    resampling_method: str = "none",
    random_seed: int = 42,
    resampling_cfg: DictConfig = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, SimpleImputer | IterativeImputer]:
    X_proc, _, scaler, imputer = apply_preprocessing(
        X, X, imputer_strategy, use_iterative_imputer
    )
    if resampling_cfg is not None:
        X_proc, y_proc = apply_resampling(X_proc, y, resampling_method, random_seed, resampling_cfg)
    else:
        y_proc = y

    return X_proc, y_proc, scaler, imputer
