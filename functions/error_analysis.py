import os
import argparse
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple

def analyze_model_errors(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    features_df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float,
    model_name: str,
    output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    
    row_missing_rate = features_df[feature_cols].isnull().mean(axis=1)
    
    stats_rows = []
    
    with open(os.path.join(output_dir, f"error_analysis_{model_name}.txt"), "w") as f:
        f.write(f"Error Analysis for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"TN: {tn_mask.sum()}, FP: {fp_mask.sum()}\n")
        f.write(f"FN: {fn_mask.sum()}, TP: {tp_mask.sum()}\n\n")
        
        f.write("1. False Negative Analysis (Missed Cases)\n")
        f.write("-" * 40 + "\n")
        if fn_mask.sum() > 0 and tp_mask.sum() > 0:
            f.write("Comparing FN (Missed) vs TP (Detected).\n\n")
            
            # 1. Missingness
            miss_fn = row_missing_rate[fn_mask]
            miss_tp = row_missing_rate[tp_mask]
            t_stat, p_miss = stats.ttest_ind(miss_fn, miss_tp, equal_var=False)
            f.write(f"[Missing Data Rate] FN Mean: {miss_fn.mean():.4f}, TP Mean: {miss_tp.mean():.4f}, p={p_miss:.4e}\n")
            if p_miss < 0.05 and miss_fn.mean() > miss_tp.mean():
                f.write(">> FN cases have significantly higher missing data rates.\n")
            f.write("\n")
            
            # 2. Features
            f.write("Features with significant differences (p < 0.05):\n\n")
            fn_diffs = _compare_groups(features_df, feature_cols, fn_mask, tp_mask, "FN", "TP")
            for item in fn_diffs:
                f.write(f"- {item['feature']}:\n")
                f.write(f"  FN Mean: {item['mean1']:.4f}, TP Mean: {item['mean2']:.4f}\n")
                f.write(f"  Diff: {item['diff']:.4f} ({item['direction']})\n")
                f.write(f"  P-value: {item['p_value']:.4e}\n")
                
                stats_rows.append({
                    "Model": model_name,
                    "Analysis": "False Negative (FN vs TP)",
                    "Feature": item['feature'],
                    "Mean_FN": item['mean1'],
                    "Mean_TP": item['mean2'],
                    "P_Value": item['p_value']
                })
        else:
            f.write("Not enough samples for FN analysis.\n")
        f.write("\n")

        # --- False Positive Analysis ---
        f.write("2. False Positive Analysis (False Alarms)\n")
        f.write("-" * 40 + "\n")
        if fp_mask.sum() > 0 and tn_mask.sum() > 0:
            f.write("Comparing FP (False Alarm) vs TN (Correct Rejection).\n\n")

            # 1. Missingness
            miss_fp = row_missing_rate[fp_mask]
            miss_tn = row_missing_rate[tn_mask]
            t_stat, p_miss = stats.ttest_ind(miss_fp, miss_tn, equal_var=False)
            f.write(f"[Missing Data Rate] FP Mean: {miss_fp.mean():.4f}, TN Mean: {miss_tn.mean():.4f}, p={p_miss:.4e}\n")
            f.write("\n")

            # 2. Features
            f.write("Features with significant differences (p < 0.05):\n\n")
            
            fp_diffs = _compare_groups(features_df, feature_cols, fp_mask, tn_mask, "FP", "TN")
            for item in fp_diffs:
                f.write(f"- {item['feature']}:\n")
                f.write(f"  FP Mean: {item['mean1']:.4f}, TN Mean: {item['mean2']:.4f}\n")
                f.write(f"  Diff: {item['diff']:.4f} ({item['direction']})\n")
                f.write(f"  P-value: {item['p_value']:.4e}\n")

                stats_rows.append({
                    "Model": model_name,
                    "Analysis": "False Positive (FP vs TN)",
                    "Feature": item['feature'],
                    "Mean_FP": item['mean1'],
                    "Mean_TN": item['mean2'],
                    "P_Value": item['p_value']
                })
        else:
            f.write("Not enough samples for FP analysis.\n")
        f.write("\n")
        
    # Save CSV summary
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(os.path.join(output_dir, f"error_analysis_stats_{model_name}.csv"), index=False)
    
    print(f"[{model_name}] Analysis saved to {output_dir}")

def _compare_groups(df, feature_cols, mask1, mask2, name1, name2):
    """
    Compares features between two groups using t-test (or similar).
    Returns list of features with significant differences.
    """
    results = []
    
    group1 = df.loc[mask1, feature_cols]
    group2 = df.loc[mask2, feature_cols]
    
    for col in feature_cols:
        # Check if numerical
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        vals1 = group1[col].dropna()
        vals2 = group2[col].dropna()
        
        if len(vals1) < 2 or len(vals2) < 2:
            continue
            
        # T-test
        try:
            stat, p_val = stats.ttest_ind(vals1, vals2, equal_var=False)
        except Exception:
            continue
            
        if p_val < 0.05:
            mean1 = vals1.mean()
            mean2 = vals2.mean()
            diff = mean1 - mean2
            direction = "Higher" if diff > 0 else "Lower"
            
            results.append({
                "feature": col,
                "mean1": mean1,
                "mean2": mean2,
                "diff": diff,
                "direction": direction,
                "p_value": p_val
            })
            
    # Sort by p-value
    results.sort(key=lambda x: x['p_value'])
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run qualitative error analysis for ML models.")
    parser.add_argument("--features", type=str, default="data/features_preprocessed.csv", help="Path to preprocessed features CSV")
    parser.add_argument("--oof", type=str, default="data/oof_predictions.csv", help="Path to OOF predictions CSV")
    parser.add_argument("--output_dir", type=str, default="data/error_analysis", help="Directory to save output")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--meta_cols", type=str, nargs='+', default=["subject_id", "hadm_id", "label"], help="Metadata columns to exclude from features")

    args = parser.parse_args()

    # Load data
    print(f"Loading features from {args.features}...")
    features_df = pd.read_csv(args.features)
    
    print(f"Loading OOF predictions from {args.oof}...")
    oof_df = pd.read_csv(args.oof)
    
    # Ensure alignment
    if len(features_df) != len(oof_df):
        print("Error: Features and OOF predictions must have the same number of rows.")
        exit(1)
        
    y_true = oof_df["label"].values
    
    # Get feature columns
    feature_cols = [c for c in features_df.columns if c not in args.meta_cols]
    
    # Models to analyze (infer from OOF columns)
    model_cols = [c for c in oof_df.columns if c.endswith("_pred")]
    
    for model_col in model_cols:
        model_name = model_col.replace("_pred", "").capitalize() # e.g., xgboost_pred -> Xgboost (or XGBoost if we manually map)
        if "xgboost" in model_col: model_name = "XGBoost"
        elif "lightgbm" in model_col: model_name = "LightGBM"
            
        print(f"Analyzing {model_name}...")
        y_prob = oof_df[model_col].values
        
        analyze_model_errors(
            y_true=y_true,
            y_pred_prob=y_prob,
            features_df=features_df,
            feature_cols=feature_cols,
            threshold=args.threshold,
            model_name=model_name,
            output_dir=args.output_dir
        )
