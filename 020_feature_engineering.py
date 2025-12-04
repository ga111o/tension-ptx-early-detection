"""
Medical Feature Engineering Module Advanced
Convert Raw Feature CSV long format to Wide format
Domain knowledge derived variables MAP SHOCK_INDEX ROX etc
Advanced Time-series Analysis Entropy RMSSD Kurtosis Statistical Tests
Generate final Feature DataFrame
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RAW_FEATURES_PATH = "data/features_raw.csv"
ENGINEERED_FEATURES_PATH = "data/features_engineered.csv"

def load_raw_features(input_path: str = RAW_FEATURES_PATH) -> pd.DataFrame:
    """Load Raw Feature CSV"""
    print(f"\nLoading Raw Features: {input_path}")
    df = pd.read_csv(input_path)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["reference_time"] = pd.to_datetime(df["reference_time"])
    return df


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Long format to Wide format"""
    print("\nConverting Long to Wide Format")
    df = df.copy()
    df["patient_key"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
    
    pivot_df = df.pivot_table(
        index=["patient_key", "subject_id", "hadm_id", "reference_time", "label", "minutes_before_ref"],
        columns="var_name",
        values="valuenum",
        aggfunc="mean"
    ).reset_index()
    
    pivot_df.columns.name = None
    print(f"Records after conversion: {len(pivot_df)}")
    return pivot_df


def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite indicators using medical domain knowledge
    """
    print("\nCalculating derived features")
    df = df.copy()
    
    if "SBP" in df.columns and "DBP" in df.columns:
        df["MAP"] = df["DBP"] + (df["SBP"] - df["DBP"]) / 3
        df["PP"] = df["SBP"] - df["DBP"]
    
    if "HR" in df.columns and "SBP" in df.columns:
        df["SHOCK_INDEX"] = df["HR"] / df["SBP"].replace(0, np.nan)
        
    if "HR" in df.columns and "MAP" in df.columns:
        df["MSI"] = df["HR"] / df["MAP"].replace(0, np.nan)

    for col in ["FiO2", "SpO2"]:
        if col in df.columns:
            mask = df[col] > 1
            df.loc[mask, col] = df.loc[mask, col] / 100

    if "PaO2" in df.columns and "FiO2" in df.columns:
        df["PF_RATIO"] = df["PaO2"] / df["FiO2"].replace(0, np.nan)

    if all(col in df.columns for col in ["PaO2", "PaCO2", "FiO2"]):
        Pb, PH2O, RQ = 760, 47, 0.8
        pao2_alveolar = df["FiO2"] * (Pb - PH2O) - (df["PaCO2"] / RQ)
        df["AA_GRADIENT"] = pao2_alveolar - df["PaO2"]

    if all(col in df.columns for col in ["SpO2", "FiO2", "RR"]):
        sf_ratio = df["SpO2"] / df["FiO2"].replace(0, np.nan)
        df["ROX"] = sf_ratio / df["RR"].replace(0, np.nan)
    
    if "RR" in df.columns and "SpO2" in df.columns:
        df["RDI"] = df["RR"] / df["SpO2"].replace(0, np.nan) * 100

    return df


def calculate_sample_entropy(L: np.array, m: int = 2, r: float = 0.2) -> float:
    """
    Calculate Sample Entropy SampEn
    Measure complexity of biological signals
    Low values near 0: highly regular repetitive signals possible pathological state
    Moderately high values: healthy physiological chaos state
    """
    L = np.array(L)
    N = len(L)
    if N < m + 1:
        return np.nan
    
    if np.std(L) == 0: return 0
    L = (L - np.mean(L)) / np.std(L)

    def _phi(m):
        x = np.array([L[i : i + m] for i in range(N - m + 1)])
        C = np.sum(np.abs(x[:, np.newaxis] - x) <= r, axis=2)
        return np.sum(np.all(C, axis=1)) / (N - m + 1)**2

    A = _phi(m + 1)
    B = _phi(m)
    
    if A == 0 or B == 0:
        return 0.0
        
    return -np.log(A / B)


def calculate_advanced_ts_patterns(
    patient_df: pd.DataFrame,
    vital_cols: List[str],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Extract time-series patterns and statistical test statistics
    
    1. Distribution Shape: Skewness, Kurtosis
    2. Variability Dynamics: RMSSD, MAD
    3. Complexity: Sample Entropy
    4. Statistical Trend: Mann-Kendall Tau, Shapiro-Wilk P-value
    """
    features = {}
    
    for col in vital_cols:
        if col not in patient_df.columns:
            continue
            
        ts_data = patient_df.sort_values("minutes_before_ref", ascending=False)[col].dropna()
        values = ts_data.values
        
        if len(values) < 3:
            continue
            
        col_prefix = f"{prefix}{col}_"
        
        features[f"{col_prefix}skew"] = stats.skew(values)
        features[f"{col_prefix}kurt"] = stats.kurtosis(values)
        
        diff = np.diff(values)
        features[f"{col_prefix}rmssd"] = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0
        features[f"{col_prefix}mad"] = stats.median_abs_deviation(values)
        
        if len(values) >= 10:
            features[f"{col_prefix}sampen"] = calculate_sample_entropy(values)
        else:
            features[f"{col_prefix}sampen"] = 0
            
        try:
            _, p_norm = stats.shapiro(values)
            features[f"{col_prefix}normality_p"] = p_norm
        except:
            features[f"{col_prefix}normality_p"] = 1.0
            
        try:
            tau, p_trend = stats.kendalltau(np.arange(len(values)), values)
            features[f"{col_prefix}trend_tau"] = tau if not np.isnan(tau) else 0
            features[f"{col_prefix}trend_p"] = p_trend if not np.isnan(p_trend) else 1.0
        except:
            features[f"{col_prefix}trend_tau"] = 0
            features[f"{col_prefix}trend_p"] = 1.0

    return features


def calculate_time_window_features(
    patient_df: pd.DataFrame,
    vital_cols: List[str],
    windows_minutes: List[int] = [30, 60, 120]
) -> Dict[str, float]:
    """Generate advanced features based on time windows"""
    window_dict = {}
    
    if "minutes_before_ref" not in patient_df.columns:
        return window_dict
    
    for window in windows_minutes:
        window_df = patient_df[patient_df["minutes_before_ref"] <= window]
        
        if len(window_df) == 0:
            continue
        
        prefix = f"w{window}_"
        
        stats_features = calculate_advanced_ts_patterns(window_df, vital_cols, prefix)
        window_dict.update(stats_features)
    
    return window_dict


def aggregate_patient_features(
    wide_df: pd.DataFrame,
    vital_cols: Optional[List[str]] = None,
    windows_minutes: List[int] = [30, 60, 120],
) -> pd.DataFrame:
    """Aggregate all features per patient with advanced analysis"""
    print("\nAggregating patient features")
    
    if vital_cols is None:
        vital_cols = [
            "HR", "RR", "SpO2", "SBP", "DBP", "MAP", "PP", 
            "SHOCK_INDEX", "PF_RATIO", "Temperature"
        ]
        vital_cols = [c for c in vital_cols if c in wide_df.columns]
    
    patient_features = []
    patient_keys = wide_df["patient_key"].unique()
    
    print(f"Patients: {len(patient_keys)}")
    print(f"Variables to analyze: {vital_cols}")
    
    for i, patient_key in enumerate(patient_keys):
        if (i + 1) % 100 == 0:
            print(f"Processing: {i + 1}/{len(patient_keys)}")
        
        patient_df = wide_df[wide_df["patient_key"] == patient_key].copy()
        
        features = {
            "subject_id": patient_df["subject_id"].iloc[0],
            "hadm_id": patient_df["hadm_id"].iloc[0],
            "label": patient_df["label"].iloc[0],
        }
        
        features.update(calculate_advanced_ts_patterns(patient_df, vital_cols, "all_"))
        
        features.update(calculate_time_window_features(patient_df, vital_cols, windows_minutes))
        
        patient_features.append(features)
    
    result_df = pd.DataFrame(patient_features)
    print(f"\nFinal features generated: {len(result_df.columns) - 3}")
    return result_df


def run_feature_engineering(
    input_path: str = RAW_FEATURES_PATH,
    output_path: str = ENGINEERED_FEATURES_PATH,
    windows_minutes: List[int] = [30, 60, 120],
) -> pd.DataFrame:
    
    print("=" * 60)
    print("Advanced Medical Feature Engineering Pipeline")
    print("=" * 60)
    
    raw_df = load_raw_features(input_path)
    wide_df = pivot_to_wide_format(raw_df)
    
    wide_df = calculate_derived_features(wide_df)
    
    feature_df = aggregate_patient_features(wide_df, windows_minutes=windows_minutes)
    
    feature_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Feature Engineering Complete")
    print("=" * 60)
    
    cols = [c for c in feature_df.columns if c not in ["subject_id", "hadm_id", "label"]]
    
    print("\nGenerated feature groups:")
    print(f"  - Shape (Skew/Kurt): {[c for c in cols if 'skew' in c or 'kurt' in c][:3]} ...")
    print(f"  - Variability (RMSSD/MAD): {[c for c in cols if 'rmssd' in c or 'mad' in c][:3]} ...")
    print(f"  - Complexity (SampEn): {[c for c in cols if 'sampen' in c][:3]} ...")
    print(f"  - Trend (Kendall): {[c for c in cols if 'trend' in c][:3]} ...")
    
    return feature_df


if __name__ == "__main__":
    run_feature_engineering()