"""
Preprocessing Module
====================
- Engineered Feature CSV 로드
- 결측치 처리 (그룹별 Greedy 샘플 제거 + MICE 다중 대체)
- 전처리된 Feature CSV 저장
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
    """Feature 컬럼만 추출"""
    meta_cols = {"subject_id", "hadm_id", "label", "cohort_type", "group_label", "ref_time", "window_start"}
    return [c for c in df.columns if c not in meta_cols]

def _get_group_label(label: int) -> str:
    """label → group_label 변환"""
    return "experimental" if label == 1 else "control"

def analyze_missing_by_group(
    df: pd.DataFrame,
    feature_cols: List[str],
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    verbose: bool = False,
) -> Dict:
    """그룹별 결측률 분석"""
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
            print(f"\n[{group.upper()}] 샘플: {len(group_df)}, 임계값 초과 Feature: {len(exceeding)}개")

    return result


def print_missing_summary(df: pd.DataFrame, feature_cols: Optional[List[str]] = None):
    """결측치 현황 요약 출력"""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)
    
    print("\n[결측치 현황]")
    
    # 전체 통계
    missing_rates = df[feature_cols].isna().mean() * 100
    print(f"  전체 평균 결측률: {missing_rates.mean():.2f}%")
    print(f"  결측률 범위: {missing_rates.min():.2f}% ~ {missing_rates.max():.2f}%")
    
    # 그룹별 통계
    for group_name, label_filter in [("Experimental (label=1)", df["label"] == 1), ("Control (label!=1)", df["label"] != 1)]:
        group_df = df[label_filter]
        if len(group_df) > 0:
            group_missing = group_df[feature_cols].isna().mean() * 100
            print(f"\n  [{group_name}] 샘플: {len(group_df)}개")
            print(f"    평균 결측률: {group_missing.mean():.2f}%")
            
            # 결측률 높은 상위 5개 feature
            top_missing = group_missing.sort_values(ascending=False).head(5)
            print(f"    상위 결측 Feature:")
            for feat, rate in top_missing.items():
                print(f"      - {feat}: {rate:.2f}%")


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

    print(f"  반복: {iteration}, 제거: {removed_count}개 -> Control: {len(control_df)}개")

    return pd.concat([exp_df, control_df], ignore_index=True)


def get_usable_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    control_threshold: float = 50.0,
    experimental_threshold: float = 85.0,
    verbose: bool = True,
) -> List[str]:
    """그룹별 임계값 만족 Feature 필터링"""
    control_df = df[df["label"] != 1]
    exp_df = df[df["label"] == 1]

    usable = []
    feature_missing_rates = []  # (feature_name, ctrl_rate, exp_rate)
    
    for col in feature_cols:
        ctrl_rate = control_df[col].isna().mean() * 100 if len(control_df) > 0 else 0
        exp_rate = exp_df[col].isna().mean() * 100 if len(exp_df) > 0 else 0
        if ctrl_rate <= control_threshold and exp_rate <= experimental_threshold:
            usable.append(col)
            feature_missing_rates.append((col, ctrl_rate, exp_rate))

    print(f"\n  사용 가능 Feature: {len(usable)}/{len(feature_cols)}개")
    
    if verbose and usable:
        # 결측률 기준 정렬 (Control 결측률 높은 순)
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
    """다중 대체법 (MICE × n_imputations 평균)"""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    result_df = df.copy()
    X = df[feature_cols].values
    missing_count = np.isnan(X).sum()

    if missing_count == 0:
        print("  결측치 없음")
        return df

    print(f"\n[다중 대체법] n_imputations={n_imputations}, 결측치={missing_count}")

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

    print(f"  {missing_count}개 결측치 대체 완료")
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
    그룹별 결측치 처리 파이프라인
    1) Greedy 샘플 제거
    2) Feature 필터링
    3) MICE 다중 대체
    """
    print("\n" + "=" * 60)
    print("결측치 처리 파이프라인")
    print("=" * 60)

    feature_cols = get_feature_columns(df)
    df = df.copy()
    if "group_label" not in df.columns:
        df["group_label"] = df["label"].apply(_get_group_label)

    n_exp_init = (df["group_label"] == "experimental").sum()
    n_ctrl_init = (df["group_label"] == "control").sum()

    print(f"초기: {len(df)}개 (Exp={n_exp_init}, Ctrl={n_ctrl_init}), Features={len(feature_cols)}")

    # Step 1: Greedy 샘플 제거
    df = greedy_sample_removal(
        df, feature_cols, control_threshold, experimental_threshold, min_control_samples, batch_size
    )

    # Step 2: Feature 필터링
    usable_features = get_usable_features(df, feature_cols, control_threshold, experimental_threshold)

    if not usable_features:
        return df, {"usable_features": [], "excluded_features": feature_cols}

    # Step 3: MICE 다중 대체
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

    print(f"\n완료: {len(df)}개 (Exp={n_exp_final}, Ctrl={n_ctrl_final}), Features={len(usable_features)}")
    return df, info


def load_engineered_features(input_path: str = ENGINEERED_FEATURES_PATH) -> pd.DataFrame:
    """Engineered Feature CSV 로드"""
    print(f"\n[Engineered Features 로드] {input_path}")
    df = pd.read_csv(input_path)
    feature_cols = get_feature_columns(df)
    print(f"  샘플: {len(df)}개, Features: {len(feature_cols)}개")
    return df


def save_preprocessed_features(
    df: pd.DataFrame,
    output_path: str = PREPROCESSED_FEATURES_PATH,
    usable_features: Optional[List[str]] = None,
):
    """전처리된 Feature CSV 저장"""
    # 메타 컬럼 + 사용 가능 feature만 저장
    meta_cols = ["subject_id", "hadm_id", "label"]
    if usable_features:
        save_cols = meta_cols + usable_features
        save_df = df[save_cols]
    else:
        save_df = df
    
    save_df.to_csv(output_path, index=False)
    
    feature_cols = [c for c in save_df.columns if c not in meta_cols]
    print(f"\nPreprocessed Features 저장: {output_path}")
    print(f"  샘플: {len(save_df)}개, Features: {len(feature_cols)}개")
    
    # 최종 결측치 확인
    final_missing = save_df[feature_cols].isna().sum().sum()
    print(f"  최종 결측치: {final_missing}개")


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
    전처리 파이프라인 실행

    Parameters
    ----------
    input_path : str
        Engineered Feature CSV 경로
    output_path : str
        출력 CSV 경로
    control_threshold : float
        Control 그룹 결측률 임계값 (%)
    experimental_threshold : float
        Experimental 그룹 결측률 임계값 (%)
    min_control_samples : int
        최소 Control 샘플 수
    batch_size : int
        Greedy 제거 배치 크기
    n_imputations : int
        MICE 다중 대체 횟수
    mice_max_iter : int
        MICE 최대 반복 횟수
    random_state : int
        랜덤 시드
    exclude_silver : bool
        Silver 코호트 제외 여부

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        전처리된 DataFrame과 처리 정보
    """
    print("=" * 60)
    print("Preprocessing Pipeline")
    print("=" * 60)

    # Engineered Features 로드
    df = load_engineered_features(input_path)
    
    # 결측치 현황 출력
    print_missing_summary(df)
    
    # Silver 제외 (학습에서 제외되는 샘플)
    if exclude_silver and "label" in df.columns:
        silver_count = (df["label"] == -1).sum()
        if silver_count > 0:
            print(f"\n[Silver 코호트 제외] {silver_count}개")
            df = df[df["label"] != -1].copy()
    
    # 결측치 처리
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
    
    # 전처리된 CSV 저장
    save_preprocessed_features(df, output_path, info.get("usable_features"))
    
    # 처리 정보 출력
    print("\n" + "=" * 60)
    print("전처리 완료 요약")
    print("=" * 60)
    print(f"  초기 샘플: {info['initial_samples']}개")
    print(f"  최종 샘플: {info['final_samples']}개")
    print(f"  제거된 Control 샘플: {info['removed_control']}개")
    print(f"  제거된 Experimental 샘플: {info['removed_experimental']}개")
    print(f"  사용 가능 Feature: {len(info['usable_features'])}개")
    print(f"  제외된 Feature: {len(info['excluded_features'])}개")
    if info['excluded_features']:
        print(f"    제외 목록: {info['excluded_features'][:10]}{'...' if len(info['excluded_features']) > 10 else ''}")
    
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
    
    print("Preprocessing 설정:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    df, info = run_preprocessing(**config)

