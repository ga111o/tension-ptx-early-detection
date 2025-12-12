import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def plot_calibration_curve_with_brier(
    y_true: np.ndarray,
    y_probs_dict: dict,
    output_path: str,
    n_bins: int = 10
):
    plt.figure(figsize=(8, 8))
    
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated", color="black")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (model_name, y_prob) in enumerate(y_probs_dict.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        brier = brier_score_loss(y_true, y_prob)
        
        color = colors[i % len(colors)]
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                 label=f"{model_name} (Brier: {brier:.3f})", color=color)

    plt.xlabel("Mean Predicted Probability", fontsize=14)
    plt.ylabel("Fraction of Positives", fontsize=14)
    plt.title("Calibration Curve", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration curve to {output_path}")


def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefits = []
    n = len(y_true)
    
    for thresh in thresholds:
        if thresh == 1.0:
            net_benefits.append(0.0)
            continue
            
        y_pred = (y_prob >= thresh).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        # Net Benefit formula
        # NB = (TP/N) - (FP/N) * (thresh / (1 - thresh))
        w = thresh / (1 - thresh)
        nb = (tp / n) - (fp / n) * w
        net_benefits.append(nb)
        
    return np.array(net_benefits)


def plot_decision_curve_analysis(
    y_true: np.ndarray,
    y_probs_dict: dict,
    output_path: str,
    thresholds: np.ndarray = np.linspace(0.01, 0.99, 99)
):
    """
    Plots Decision Curve Analysis (DCA).
    """
    plt.figure(figsize=(8, 8))
    
    # Calculate Treat All and Treat None
    prevalence = np.mean(y_true)
    
    # Treat All: TP=All Positives, FP=All Negatives
    # NB_all = Prevalence - (1-Prevalence)*w
    # Treat None: NB = 0
    
    treat_all_nb = []
    treat_none_nb = np.zeros(len(thresholds))
    
    for thresh in thresholds:
        if thresh == 1.0:
            treat_all_nb.append(0.0) # Should conceptually handle limit, but for DCA plot usually we stop before 1
        else:
            w = thresh / (1 - thresh)
            nb_all = prevalence - (1 - prevalence) * w
            treat_all_nb.append(nb_all)
            
    treat_all_nb = np.array(treat_all_nb)
    
    # Plot Treat All / Treat None
    plt.plot(thresholds, treat_all_nb, linestyle="--", color="gray", label="Treat All")
    plt.plot(thresholds, treat_none_nb, linestyle="-", color="black", label="Treat None")
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Clip Y-axis to reasonable range (e.g., -0.05 to max prevalence + buffer)
    y_min = -0.05
    y_max = prevalence + 0.05
    
    for i, (model_name, y_prob) in enumerate(y_probs_dict.items()):
        nb = calculate_net_benefit(y_true, y_prob, thresholds)
        
        color = colors[i % len(colors)]
        plt.plot(thresholds, nb, linewidth=2, label=model_name, color=color)
        
    plt.ylim(y_min, y_max)
    plt.xlim(0, 1)
    plt.xlabel("Threshold Probability", fontsize=14)
    plt.ylabel("Net Benefit", fontsize=14)
    plt.title("Decision Curve Analysis", fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved DCA plot to {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation plots from OOF predictions.")
    parser.add_argument("--input", type=str, default="data/oof_predictions.csv", help="Path to OOF predictions CSV")
    parser.add_argument("--output_dir", type=str, default="report/figures", help="Directory to save plots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        # Fallback for running from functions/ directory or other locations
        if os.path.exists(os.path.join("..", args.input)):
            args.input = os.path.join("..", args.input)
            
            # Adjust output dir as well if using relative paths from subdir
            if args.output_dir == "report/figures":
                args.output_dir = "../report/figures"
        else:
            print(f"Error: Input file '{args.input}' not found.")
            exit(1)
            
    print(f"Loading predictions from {args.input}...")
    df = pd.read_csv(args.input)
    
    y_true = df["label"].values
    y_probs = {}
    
    if "xgboost_pred" in df.columns:
        y_probs["XGBoost"] = df["xgboost_pred"].values
    if "lightgbm_pred" in df.columns:
        y_probs["LightGBM"] = df["lightgbm_pred"].values
        
    if not y_probs:
        print("Error: No prediction columns found (xgboost_pred, lightgbm_pred).")
        exit(1)

    print(f"Generating plots in {args.output_dir}...")
    
    plot_calibration_curve_with_brier(
        y_true, 
        y_probs, 
        os.path.join(args.output_dir, "calibration_curve.png")
    )
    
    plot_decision_curve_analysis(
        y_true, 
        y_probs, 
        os.path.join(args.output_dir, "decision_curve_analysis.png")
    )
    
    print("Done.")
