"""
Medical Feature Engineering Module (Advanced)
=============================================
- Raw Feature CSV (long format) → Wide format 변환
- 도메인 지식 기반 파생변수 (MAP, SHOCK_INDEX, ROX 등)
- Advanced Time-series Analysis (Entropy, RMSSD, Kurtosis, Statistical Tests)
- 최종 Feature DataFrame 생성
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

RAW_FEATURES_PATH = "data/features_raw.csv"
ENGINEERED_FEATURES_PATH = "data/features_engineered.csv"

# =============================================================================
# 1. Long → Wide Format 변환
# =============================================================================

def load_raw_features(input_path: str = RAW_FEATURES_PATH) -> pd.DataFrame:
    """Raw Feature CSV 로드"""
    print(f"\n[Raw Features 로드] {input_path}")
    df = pd.read_csv(input_path)
    df["charttime"] = pd.to_datetime(df["charttime"])
    df["reference_time"] = pd.to_datetime(df["reference_time"])
    return df


def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Long format → Wide format 변환"""
    print("\n[Long → Wide Format 변환]")
    df = df.copy()
    df["patient_key"] = df["subject_id"].astype(str) + "_" + df["hadm_id"].astype(str)
    
    pivot_df = df.pivot_table(
        index=["patient_key", "subject_id", "hadm_id", "reference_time", "label", "minutes_before_ref"],
        columns="var_name",
        values="valuenum",
        aggfunc="mean"
    ).reset_index()
    
    pivot_df.columns.name = None
    print(f"  변환 후 레코드: {len(pivot_df)}개")
    return pivot_df


# =============================================================================
# 2. 도메인 지식 기반 파생변수 (Hemodynamic & Respiratory)
# =============================================================================

def calculate_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    의료 도메인 지식을 활용한 복합 지표 계산
    """
    print("\n[파생변수 계산]")
    df = df.copy()
    
    # 1. 혈역학적 지표 (Hemodynamic)
    if "SBP" in df.columns and "DBP" in df.columns:
        # MAP: 장기 관류압의 핵심 지표
        df["MAP"] = df["DBP"] + (df["SBP"] - df["DBP"]) / 3
        # Pulse Pressure: 맥압 (심혈관 경직도 반영)
        df["PP"] = df["SBP"] - df["DBP"]
    
    if "HR" in df.columns and "SBP" in df.columns:
        # Shock Index: 쇼크의 조기 경고 지표 (SI > 0.7 주의, > 1.0 위험)
        df["SHOCK_INDEX"] = df["HR"] / df["SBP"].replace(0, np.nan)
        
    if "HR" in df.columns and "MAP" in df.columns:
        # Modified Shock Index
        df["MSI"] = df["HR"] / df["MAP"].replace(0, np.nan)

    # 2. 호흡기계 지표 (Respiratory)
    # FiO2, SpO2 단위 보정 (1보다 크면 %로 간주하여 100으로 나눔)
    for col in ["FiO2", "SpO2"]:
        if col in df.columns:
            mask = df[col] > 1
            df.loc[mask, col] = df.loc[mask, col] / 100

    if "PaO2" in df.columns and "FiO2" in df.columns:
        # P/F Ratio: 급성 호흡 곤란 증후군(ARDS) 진단 기준
        df["PF_RATIO"] = df["PaO2"] / df["FiO2"].replace(0, np.nan)

    if all(col in df.columns for col in ["PaO2", "PaCO2", "FiO2"]):
        # A-a Gradient: 폐포-동맥혈 산소 분압 차 (가스 교환 효율성)
        Pb, PH2O, RQ = 760, 47, 0.8
        pao2_alveolar = df["FiO2"] * (Pb - PH2O) - (df["PaCO2"] / RQ)
        df["AA_GRADIENT"] = pao2_alveolar - df["PaO2"]

    if all(col in df.columns for col in ["SpO2", "FiO2", "RR"]):
        # ROX Index: 고유량 비강 캐뉼라 실패 예측 (< 4.88 위험)
        sf_ratio = df["SpO2"] / df["FiO2"].replace(0, np.nan)
        df["ROX"] = sf_ratio / df["RR"].replace(0, np.nan)
    
    # RDI: 호흡 곤란 지수
    if "RR" in df.columns and "SpO2" in df.columns:
        df["RDI"] = df["RR"] / df["SpO2"].replace(0, np.nan) * 100

    return df


# =============================================================================
# 3. [핵심 수정] Advanced Time-series Patterns & Statistical Tests
# =============================================================================

def calculate_sample_entropy(L: np.array, m: int = 2, r: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) 계산
    - 생체 신호의 복잡성(Complexity) 측정
    - 값이 낮음(0에 근접): 매우 규칙적이고 반복적인 신호 (병적인 상태 가능성)
    - 값이 적당히 높음: 건강한 카오스(Physiological Chaos) 상태
    """
    L = np.array(L)
    N = len(L)
    if N < m + 1:
        return np.nan
    
    # 데이터 표준화 (Scale dependency 제거)
    if np.std(L) == 0: return 0
    L = (L - np.mean(L)) / np.std(L)

    def _phi(m):
        # 벡터화된 패턴 매칭
        x = np.array([L[i : i + m] for i in range(N - m + 1)])
        # 자신과의 거리 계산 (Chebyshev distance)
        C = np.sum(np.abs(x[:, np.newaxis] - x) <= r, axis=2)
        # Self-match 제외하고 확률 계산 (N-m+1) * (N-m) 로 나누는 것이 정석이나 근사치 사용
        return np.sum(np.all(C, axis=1)) / (N - m + 1)**2

    # 로그 계산 시 0 방지
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
    [대체 함수] 단순 통계량 대신 시계열 패턴 및 검정 통계량 추출
    
    1. Distribution Shape: Skewness, Kurtosis
    2. Variability Dynamics: RMSSD, MAD
    3. Complexity: Sample Entropy
    4. Statistical Trend: Mann-Kendall Tau, Shapiro-Wilk P-value
    """
    features = {}
    
    for col in vital_cols:
        if col not in patient_df.columns:
            continue
            
        # 시간순 정렬 (과거 -> 현재)
        ts_data = patient_df.sort_values("minutes_before_ref", ascending=False)[col].dropna()
        values = ts_data.values
        
        # 데이터가 너무 적으면 계산 불가
        if len(values) < 3:
            continue
            
        col_prefix = f"{prefix}{col}_"
        
        # --- 1. 분포 형태 (Distribution Shape) ---
        # 왜도 (Skewness): 급격한 악화(꼬리) 감지. 0이면 대칭.
        features[f"{col_prefix}skew"] = stats.skew(values)
        # 첨도 (Kurtosis): 이상치(Extreme values)의 빈도. 높으면 이상치가 많음.
        features[f"{col_prefix}kurt"] = stats.kurtosis(values)
        
        # --- 2. 변동성 역학 (Variability Dynamics) ---
        # RMSSD: 인접 값 간 차이의 제곱 평균 제곱근. 단기 변동성(Short-term variability) 측정.
        diff = np.diff(values)
        features[f"{col_prefix}rmssd"] = np.sqrt(np.mean(diff**2)) if len(diff) > 0 else 0
        # MAD: 중앙값 절대 편차. 이상치에 강건한 변동성 지표.
        features[f"{col_prefix}mad"] = stats.median_abs_deviation(values)
        
        # --- 3. 복잡성 (Complexity) ---
        # Sample Entropy: 신호의 불규칙성. (데이터 10개 이상일 때 권장)
        if len(values) >= 10:
            features[f"{col_prefix}sampen"] = calculate_sample_entropy(values)
        else:
            features[f"{col_prefix}sampen"] = 0
            
        # --- 4. 통계적 검정 (Statistical Tests) ---
        # Shapiro-Wilk: 정규성 검정 P-value. (낮으면 비정상 분포일 확률 높음)
        try:
            _, p_norm = stats.shapiro(values)
            features[f"{col_prefix}normality_p"] = p_norm
        except:
            features[f"{col_prefix}normality_p"] = 1.0
            
        # Kendall's Tau: 비모수적 추세 검정 (선형회귀보다 이상치에 강함)
        # 시간 순서(index)와 값 사이의 상관관계
        try:
            # x: 시간 순서 (0, 1, 2...), y: 값
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
    """시간대별 윈도우 기반 고급 피쳐 생성"""
    window_dict = {}
    
    if "minutes_before_ref" not in patient_df.columns:
        return window_dict
    
    for window in windows_minutes:
        window_df = patient_df[patient_df["minutes_before_ref"] <= window]
        
        if len(window_df) == 0:
            continue
        
        prefix = f"w{window}_"
        
        # [수정] 단순 통계량 함수 호출 제거 -> 고급 분석 함수 호출
        stats_features = calculate_advanced_ts_patterns(window_df, vital_cols, prefix)
        window_dict.update(stats_features)
    
    return window_dict


# =============================================================================
# 4. 환자별 피쳐 집계
# =============================================================================

def aggregate_patient_features(
    wide_df: pd.DataFrame,
    vital_cols: Optional[List[str]] = None,
    windows_minutes: List[int] = [30, 60, 120],
) -> pd.DataFrame:
    """환자별 모든 피쳐 집계 (고급 분석 적용)"""
    print("\n[환자별 피쳐 집계]")
    
    if vital_cols is None:
        vital_cols = [
            "HR", "RR", "SpO2", "SBP", "DBP", "MAP", "PP", 
            "SHOCK_INDEX", "PF_RATIO", "Temperature"
        ]
        # 실제 데이터프레임에 존재하는 컬럼만 선택
        vital_cols = [c for c in vital_cols if c in wide_df.columns]
    
    patient_features = []
    patient_keys = wide_df["patient_key"].unique()
    
    print(f"  환자 수: {len(patient_keys)}명")
    print(f"  분석 대상 변수: {vital_cols}")
    
    for i, patient_key in enumerate(patient_keys):
        if (i + 1) % 100 == 0:
            print(f"  처리 중: {i + 1}/{len(patient_keys)}")
        
        patient_df = wide_df[wide_df["patient_key"] == patient_key].copy()
        
        # 메타 정보
        features = {
            "subject_id": patient_df["subject_id"].iloc[0],
            "hadm_id": patient_df["hadm_id"].iloc[0],
            "label": patient_df["label"].iloc[0],
        }
        
        # 1. 전체 기간 고급 패턴 분석 [변경됨]
        features.update(calculate_advanced_ts_patterns(patient_df, vital_cols, "all_"))
        
        # 2. 시간대별 윈도우 패턴 분석 [변경됨]
        features.update(calculate_time_window_features(patient_df, vital_cols, windows_minutes))
        
        patient_features.append(features)
    
    result_df = pd.DataFrame(patient_features)
    print(f"\n  최종 생성된 피쳐 수: {len(result_df.columns) - 3}개")
    return result_df


# =============================================================================
# 5. 메인 파이프라인
# =============================================================================

def run_feature_engineering(
    input_path: str = RAW_FEATURES_PATH,
    output_path: str = ENGINEERED_FEATURES_PATH,
    windows_minutes: List[int] = [30, 60, 120],
) -> pd.DataFrame:
    
    print("=" * 60)
    print("Advanced Medical Feature Engineering Pipeline")
    print("=" * 60)
    
    # 1. 로드 & 변환
    raw_df = load_raw_features(input_path)
    wide_df = pivot_to_wide_format(raw_df)
    
    # 2. 도메인 파생변수 (SI, MAP, ROX 등)
    wide_df = calculate_derived_features(wide_df)
    
    # 3. 고급 시계열 패턴 추출 및 집계
    feature_df = aggregate_patient_features(wide_df, windows_minutes=windows_minutes)
    
    # 4. 저장
    feature_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("Feature Engineering 완료")
    print("=" * 60)
    
    # 생성된 피쳐 확인
    cols = [c for c in feature_df.columns if c not in ["subject_id", "hadm_id", "label"]]
    
    print("\n  [생성된 주요 피쳐 그룹]")
    print(f"  - Shape (Skew/Kurt): {[c for c in cols if 'skew' in c or 'kurt' in c][:3]} ...")
    print(f"  - Variability (RMSSD/MAD): {[c for c in cols if 'rmssd' in c or 'mad' in c][:3]} ...")
    print(f"  - Complexity (SampEn): {[c for c in cols if 'sampen' in c][:3]} ...")
    print(f"  - Trend (Kendall): {[c for c in cols if 'trend' in c][:3]} ...")
    
    return feature_df


if __name__ == "__main__":
    run_feature_engineering()