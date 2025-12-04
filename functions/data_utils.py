import pandas as pd
from typing import List


def load_preprocessed_features(path: str, meta_cols: List[str]) -> pd.DataFrame:
    print(f"\nLoading preprocessed features: {path}")
    df = pd.read_csv(path)

    feature_cols = get_feature_columns(df, meta_cols)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()

    print(f"Samples: {len(df)} Positive={n_pos}, Negative={n_neg}")
    print(f"Features: {len(feature_cols)}")

    missing_count = df[feature_cols].isna().sum().sum()
    if missing_count > 0:
        print(f"Missing values: {missing_count}")
    else:
        print("No missing values")

    return df


def get_feature_columns(df: pd.DataFrame, meta_cols: List[str]) -> List[str]:
    return [c for c in df.columns if c not in meta_cols]
