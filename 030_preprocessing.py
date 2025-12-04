"""
Preprocessing Module
Load Engineered Feature CSV
Handle missing values with Greedy sample removal and MICE imputation
Save preprocessed Feature CSV
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

warnings.filterwarnings("ignore")


ENGINEERED_FEATURES_PATH = "data/features_engineered.csv"
PREPROCESSED_FEATURES_PATH = "data/features_preprocessed.csv"

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns only"""
    meta_cols = {"subject_id", "hadm_id", "label", "cohort_type", "group_label", "ref_time", "window_start"}
    return [c for c in df.columns if c not in meta_cols]

def _get_group_label(label: int) -> str:
    """Convert label to group_label"""
    return "experimental" if label == 1 else "control"

def analyze_missing_by_group(
    df: pd.DataFrame,
    feature_cols: List[str],
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    verbose: bool = False,
) -> Dict:
    """Analyze missing rate by group"""
    thresholds = {"control": control_threshold, "experimental": experimental_threshold}
    result = {}

    for group in ["control", "experimental"]:
        group_df = df[df["label"] == 1] if group == "experimental" else df[df["label"] != 1]
        threshold = thresholds[group]
        exceeding = []

        for col in feature_cols:
            if len(group_df) > 0:
                missing_rate = group_df[col].isna().mean() * 100
                if missing_rate > threshold:
                    exceeding.append((col, missing_rate))

        result[group] = {"count": len(group_df), "threshold": threshold, "exceeding_features": exceeding}

        if verbose:
            print(f"\n{group.upper()} Samples: {len(group_df)}, Features exceeding threshold: {len(exceeding)}")

    return result


def print_missing_summary(df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
    """Print missing value summary"""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    print("\nMissing value summary")
    
    missing_rates = df[feature_cols].isna().mean() * 100
    print(f"Overall average missing rate: {missing_rates.mean():.2f}%")
    print(f"Missing rate range: {missing_rates.min():.2f}% ~ {missing_rates.max():.2f}%")
    
    for group_name, label_filter in [("Experimental label=1", df["label"] == 1), ("Control label!=1", df["label"] != 1)]:
        group_df = df[label_filter]
        if len(group_df) > 0:
            group_missing = group_df[feature_cols].isna().mean() * 100
            print(f"\n{group_name} Samples: {len(group_df)}")
            print(f"Average missing rate: {group_missing.mean():.2f}%")
            
            top_missing = group_missing.sort_values(ascending=False).head(5)
            print(f"Top 5 missing features:")
            for feat, rate in top_missing.items():
                print(f"  - {feat}: {rate:.2f}%")


def greedy_sample_removal(
    df: pd.DataFrame,
    feature_cols: List[str],
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    min_control_samples: int = 100,
    batch_size: int = 100,
) -> pd.DataFrame:
    df = df.copy()
    if "group_label" not in df.columns:
        df["group_label"] = df["label"].apply(_get_group_label)

    exp_df = df[df["group_label"] == "experimental"].reset_index(drop=True)
    control_df = df[df["group_label"] == "control"].reset_index(drop=True)

    print(f"\n[Greedy Sample Removal]")
    print(f"  Experimental: {len(exp_df)}, Control: {len(control_df)}")

    iteration = 0
    removed_count = 0

    while True:
        iteration += 1

        exceeding = []
        for col in feature_cols:
            if len(control_df) > 0:
                rate = control_df[col].isna().mean() * 100
                if rate > control_threshold:
                    exceeding.append((col, rate - control_threshold))

        if not exceeding or len(control_df) <= min_control_samples:
            break

        scores = pd.Series(0.0, index=control_df.index)
        for col, excess in exceeding:
            scores[control_df[col].isna()] += excess

        scores = scores.sort_values(ascending=False)
        n_remove = min(batch_size, len(control_df) - min_control_samples)
        if n_remove <= 0:
            break

        indices_to_remove = scores.head(n_remove).index.tolist()
        control_df = control_df.drop(indices_to_remove).reset_index(drop=True)
        removed_count += n_remove

    print(f"Iterations: {iteration}, Removed: {removed_count} -> Control: {len(control_df)}")

    return pd.concat([exp_df, control_df], ignore_index=True)


def get_usable_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    verbose: bool = True,
) -> List[str]:
    """Filter features meeting group-specific thresholds"""
    control_df = df[df["label"] != 1]
    exp_df = df[df["label"] == 1]

    usable = []
    feature_missing_rates = []
    
    for col in feature_cols:
        ctrl_rate = control_df[col].isna().mean() * 100 if len(control_df) > 0 else 0
        exp_rate = exp_df[col].isna().mean() * 100 if len(exp_df) > 0 else 0
        if ctrl_rate <= control_threshold and exp_rate <= experimental_threshold:
            usable.append(col)
            feature_missing_rates.append((col, ctrl_rate, exp_rate))

    print(f"\nUsable features: {len(usable)}/{len(feature_cols)}")
    
    if verbose and usable:
        feature_missing_rates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n  {'Feature':<40} {'Control(%)':<12} {'Exp(%)':<12}")
        print(f"  {'-'*40} {'-'*12} {'-'*12}")
        for feat, ctrl_rate, exp_rate in feature_missing_rates:
            print(f"  {feat:<40} {ctrl_rate:>10.2f}% {exp_rate:>10.2f}%")
    
    return usable


def apply_multiple_imputation(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Multiple imputation with MICE"""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    result_df = df.copy()
    X = df[feature_cols].values
    missing_count = np.isnan(X).sum()

    if missing_count == 0:
        print("No missing values")
        return df

    print(f"\nMultiple imputation n_imputations={n_imputations}, missing={missing_count}")

    imputed_arrays = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_imputations):
            imputer = IterativeImputer(
                max_iter=max_iter,
                random_state=random_state + i,
                initial_strategy="mean",
                skip_complete=True,
                min_value=0,
                verbose=0,
            )
            imputed_arrays.append(imputer.fit_transform(X))

    X_final = np.mean(imputed_arrays, axis=0)
    for idx, col in enumerate(feature_cols):
        result_df[col] = X_final[:, idx]

    print(f"Imputation complete: {missing_count} missing values filled")
    return result_df


def handle_missing_values(
    df: pd.DataFrame,
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    min_control_samples: int = 100,
    batch_size: int = 100,
    n_imputations: int = 5,
    mice_max_iter: int = 10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Group-specific missing value handling pipeline
    1 Greedy sample removal
    2 Feature filtering
    3 MICE multiple imputation
    """
    print("\n" + "=" * 60)
    print("Missing Value Handling Pipeline")
    print("=" * 60)

    feature_cols = get_feature_columns(df)
    df = df.copy()
    if "group_label" not in df.columns:
        df["group_label"] = df["label"].apply(_get_group_label)

    n_exp_init = (df["group_label"] == "experimental").sum()
    n_ctrl_init = (df["group_label"] == "control").sum()

    print(f"Initial: {len(df)} samples Exp={n_exp_init}, Ctrl={n_ctrl_init}, Features={len(feature_cols)}")

    df = greedy_sample_removal(
        df, feature_cols, control_threshold, experimental_threshold, min_control_samples, batch_size
    )

    usable_features = get_usable_features(df, feature_cols, control_threshold, experimental_threshold)

    if not usable_features:
        return df, {"usable_features": [], "excluded_features": feature_cols}

    df = apply_multiple_imputation(df, usable_features, n_imputations, mice_max_iter, random_state)

    n_exp_final = (df["group_label"] == "experimental").sum()
    n_ctrl_final = (df["group_label"] == "control").sum()

    info = {
        "initial_samples": n_exp_init + n_ctrl_init,
        "final_samples": len(df),
        "removed_control": n_ctrl_init - n_ctrl_final,
        "removed_experimental": n_exp_init - n_exp_final,
        "usable_features": usable_features,
        "excluded_features": [f for f in feature_cols if f not in usable_features],
        "final_missing_rate": df[usable_features].isna().mean().mean(),
    }

    print(f"\nComplete: {len(df)} samples Exp={n_exp_final}, Ctrl={n_ctrl_final}, Features={len(usable_features)}")
    return df, info


def load_engineered_features(input_path: str = ENGINEERED_FEATURES_PATH) -> pd.DataFrame:
    """Load Engineered Feature CSV"""
    print(f"\nLoading Engineered Features: {input_path}")
    df = pd.read_csv(input_path)
    feature_cols = get_feature_columns(df)
    print(f"Samples: {len(df)}, Features: {len(feature_cols)}")
    return df


def save_preprocessed_features(
    df: pd.DataFrame,
    output_path: str = PREPROCESSED_FEATURES_PATH,
    usable_features: Optional[List[str]] = None,
):
    """Save preprocessed Feature CSV"""
    meta_cols = ["subject_id", "hadm_id", "label"]
    if usable_features:
        save_cols = meta_cols + usable_features
        save_df = df[save_cols]
    else:
        save_df = df
    
    save_df.to_csv(output_path, index=False)
    
    feature_cols = [c for c in save_df.columns if c not in meta_cols]
    print(f"\nSaving Preprocessed Features: {output_path}")
    print(f"Samples: {len(save_df)}, Features: {len(feature_cols)}")
    
    final_missing = save_df[feature_cols].isna().sum().sum()
    print(f"Final missing values: {final_missing}")


def run_preprocessing(
    input_path: str = ENGINEERED_FEATURES_PATH,
    output_path: str = PREPROCESSED_FEATURES_PATH,
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    min_control_samples: int = 100,
    batch_size: int = 100,
    n_imputations: int = 5,
    mice_max_iter: int = 10,
    random_state: int = 42,
    exclude_silver: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run preprocessing pipeline

    Parameters
    input_path: Engineered Feature CSV path
    output_path: Output CSV path
    control_threshold: Control group missing rate threshold
    experimental_threshold: Experimental group missing rate threshold
    min_control_samples: Minimum Control sample count
    batch_size: Greedy removal batch size
    n_imputations: MICE multiple imputation count
    mice_max_iter: MICE max iterations
    random_state: Random seed
    exclude_silver: Exclude silver cohort

    Returns
    Preprocessed DataFrame and processing info
    """
    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    df = load_engineered_features(input_path)
    
    print_missing_summary(df)
    
    if exclude_silver and "label" in df.columns:
        silver_count = (df["label"] == -1).sum()
        if silver_count > 0:
            print(f"\nExcluding Silver cohort: {silver_count} samples")
            df = df[df["label"] != -1].copy()
    
    df, info = handle_missing_values(
        df,
        control_threshold=control_threshold,
        experimental_threshold=experimental_threshold,
        min_control_samples=min_control_samples,
        batch_size=batch_size,
        n_imputations=n_imputations,
        mice_max_iter=mice_max_iter,
        random_state=random_state,
    )
    
    save_preprocessed_features(df, output_path, info.get("usable_features"))
    
    print("\n" + "=" * 60)
    print("Preprocessing Complete Summary")
    print("=" * 60)
    print(f"Initial samples: {info['initial_samples']}")
    print(f"Final samples: {info['final_samples']}")
    print(f"Removed Control samples: {info['removed_control']}")
    print(f"Removed Experimental samples: {info['removed_experimental']}")
    print(f"Usable features: {len(info['usable_features'])}")
    print(f"Excluded features: {len(info['excluded_features'])}")
    if info['excluded_features']:
        print(f"Excluded list: {info['excluded_features'][:10]}{'...' if len(info['excluded_features']) > 10 else ''}")
    
    return df, info


if __name__ == "__main__":
    config = {
        "input_path": "data/features_engineered.csv",
        "output_path": "data/features_preprocessed.csv",
        "control_threshold": 50.0,
        "experimental_threshold": 85.0,
        "min_control_samples": 750,
        "batch_size": 50,
        "n_imputations": 5,
        "mice_max_iter": 10,
        "random_state": 42,
        "exclude_silver": True,
    }
    
    print("Preprocessing configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    df, info = run_preprocessing(**config)

