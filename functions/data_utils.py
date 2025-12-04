"""
Data loading and feature extraction utilities
"""

import pandas as pd
from typing import List


def load_preprocessed_features(path: str, meta_cols: List[str]) -> pd.DataFrame:
    """
    Load preprocessed features from CSV file

    Args:
        path: Path to the features CSV file
        meta_cols: List of metadata column names to exclude from features

    Returns:
        DataFrame containing the loaded features
    """
    print(f"\n[전처리된 Features 로드] {path}")
    df = pd.read_csv(path)

    feature_cols = get_feature_columns(df, meta_cols)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()

    print(f"  샘플: {len(df)}개 (Positive={n_pos}, Negative={n_neg})")
    print(f"  Features: {len(feature_cols)}개")

    missing_count = df[feature_cols].isna().sum().sum()
    if missing_count > 0:
        print(f"  결측치: {missing_count}개")
    else:
        print("  결측치 없음")

    return df


def get_feature_columns(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    """
    Extract feature column names from DataFrame, excluding metadata columns

    Args:
        df: DataFrame containing features and metadata
        meta_cols: List of metadata column names to exclude

    Returns:
        List of feature column names
    """
    return [c for c in df.columns if c not in meta_cols]
