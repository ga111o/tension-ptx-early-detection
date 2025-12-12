import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(features_path, oof_path):
    features_df = pd.read_csv(features_path)
    oof_df = pd.read_csv(oof_path)
    return features_df, oof_df

def get_significant_features(stats_path, analysis_type="False Negative", top_n=5):
    if not os.path.exists(stats_path):
        return []
    
    df = pd.read_csv(stats_path)
    df = df[df['Analysis'].str.contains(analysis_type, case=False, regex=False)]
    
    df = df.sort_values("P_Value")
    
    return df['Feature'].head(top_n).tolist()

def plot_feature_distributions(features_df, y_true, y_pred_prob, feature_cols, output_dir, model_name, group_name):
    if not feature_cols:
        print(f"No significant features found for {group_name}.")
        return
        
    threshold = 0.5
    y_pred = (y_pred_prob >= threshold).astype(int)

    if group_name == "FN":
        error_mask = (y_true == 1) & (y_pred == 0)
        correct_mask = (y_true == 1) & (y_pred == 1)
        labels = ["Missed (FN)", "Detected (TP)"]
        palette = ["#e74c3c", "#2ecc71"] # Red, Green
    else: 
        error_mask = (y_true == 0) & (y_pred == 1)
        correct_mask = (y_true == 0) & (y_pred == 0)
        labels = ["False Alarm (FP)", "Correct Rejection (TN)"]
        palette = ["#e67e22", "#3498db"] # Orange, Blue

    if error_mask.sum() == 0 or correct_mask.sum() == 0:
        print(f"Not enough samples for {group_name} visualization.")
        return

    plot_data = []
    
    error_data = features_df.loc[error_mask, feature_cols].copy()
    error_data['Group'] = labels[0]
    plot_data.append(error_data)
    
    correct_data = features_df.loc[correct_mask, feature_cols].copy()
    correct_data['Group'] = labels[1]
    plot_data.append(correct_data)
    
    combined_df = pd.concat(plot_data)
    
    melted_df = combined_df.melt(id_vars="Group", value_vars=feature_cols, var_name="Feature", value_name="Value")
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df, x="Feature", y="Value", hue="Group", palette=palette)
    plt.title(f"{model_name}: Top Features distinguishing {labels[0]} from {labels[1]}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"dist_{model_name}_{group_name}_top{len(feature_cols)}.png")
    plt.savefig(output_path)
    print(f"Saved distribution plot to {output_path}")
    plt.close()

def find_representative_cases(features_df, oof_df, y_true, y_pred_prob, feature_cols, model_name, group_name, top_k=5):
    """
    Identifies specific cases that are 'most typical' of the error type based on the features.
    For FN (Missed): Finds cases with feature values most deviant from TP mean (in the direction of FN mean).
    Basically, looks for the 'most flat' signals if flatness is the distinguishing factor.
    """
    threshold = 0.5
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    if group_name == "FN":
        mask = (y_true == 1) & (y_pred == 0)
        direction_df = pd.read_csv(f"data/error_analysis/error_analysis_stats_{model_name}.csv")
        direction_df = direction_df[direction_df['Analysis'].str.contains("False Negative")]
    else:
        mask = (y_true == 0) & (y_pred == 1)
        direction_df = pd.read_csv(f"data/error_analysis/error_analysis_stats_{model_name}.csv")
        direction_df = direction_df[direction_df['Analysis'].str.contains("False Positive")]

    if mask.sum() == 0:
        return

    candidates = features_df[mask].copy()
    
    # Calculate a "Deviance Score"
    # We want cases that are EXTREME in the direction of the error.
    # E.g. if FN mean < TP mean for a feature, we want the FN cases with the LOWEST values.
    
    scores = np.zeros(len(candidates))
    
    for feat in feature_cols:
        feat_row = direction_df[direction_df['Feature'] == feat]
        if feat_row.empty:
            continue
            
        mean_error = feat_row.iloc[0][f'Mean_{group_name}']
        mean_correct = feat_row.iloc[0][f'Mean_{"TP" if group_name == "FN" else "TN"}']
        
        # Standardize
        vals = candidates[feat].values
        std = vals.std() if vals.std() > 0 else 1.0
        
        if mean_error < mean_correct:
            # We want smaller values (more negative z-score)
            scores -= (vals - mean_error) / std
        else:
            # We want larger values
            scores += (vals - mean_error) / std
            
    # Add predicted probability info
    # For FN: Lower probability = "More confident it's negative" = "Worse miss"
    # For FP: Higher probability = "More confident it's positive" = "Worse alarm"
    prob_vals = y_pred_prob[mask]
    
    candidates['error_score'] = scores
    candidates['pred_prob'] = prob_vals
    
    # Combine feature deviance and probability confidence?
    # Let's just list them.
    
    print(f"\n--- Top {top_k} Representative {group_name} Cases ({model_name}) ---")
    print("Sorted by feature 'extremeness' (most typical of the error pattern):")
    
    sorted_idx = np.argsort(scores)[::-1] # Higher score = more aligned with error direction
    top_indices = candidates.index[sorted_idx[:top_k]]
    
    for idx in top_indices:
        subj = candidates.loc[idx, 'subject_id'] if 'subject_id' in candidates else "N/A"
        hadm = candidates.loc[idx, 'hadm_id'] if 'hadm_id' in candidates else "N/A"
        prob = candidates.loc[idx, 'pred_prob']
        print(f"Index: {idx}, Subject: {subj}, Hadm: {hadm}, Prob: {prob:.4f}")
        
        # Print feature values
        print("  Key Features:")
        for feat in feature_cols:
            val = candidates.loc[idx, feat]
            print(f"    {feat}: {val:.4f}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LightGBM")
    parser.add_argument("--output_dir", type=str, default="data/error_analysis/plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    features_df, oof_df = load_data("data/features_preprocessed.csv", "data/oof_predictions.csv")
    
    # Infer model column
    model_col = f"{args.model.lower()}_pred"
    if model_col not in oof_df.columns:
        # try search
        found = [c for c in oof_df.columns if args.model.lower() in c]
        if found:
            model_col = found[0]
        else:
            print(f"Column for {args.model} not found.")
            exit()
            
    y_true = oof_df['label'].values
    y_pred_prob = oof_df[model_col].values
    
    stats_file = f"data/error_analysis/error_analysis_stats_{args.model}.csv"
    
    # --- FN Analysis ---
    fn_feats = get_significant_features(stats_file, "False Negative")
    if fn_feats:
        print(f"Top distinguishing features for FN: {fn_feats}")
        plot_feature_distributions(features_df, y_true, y_pred_prob, fn_feats, args.output_dir, args.model, "FN")
        find_representative_cases(features_df, oof_df, y_true, y_pred_prob, fn_feats, args.model, "FN")
        
    # --- FP Analysis ---
    fp_feats = get_significant_features(stats_file, "False Positive")
    if fp_feats:
        print(f"Top distinguishing features for FP: {fp_feats}")
        plot_feature_distributions(features_df, y_true, y_pred_prob, fp_feats, args.output_dir, args.model, "FP")
        find_representative_cases(features_df, oof_df, y_true, y_pred_prob, fp_feats, args.model, "FP")
