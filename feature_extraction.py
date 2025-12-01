import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

VITAL_ITEMIDS = {
    "HR": [220045],
    "RR": [220210, 224690],
    "SpO2": [220277],
    "SBP": [220179, 220050],
    "DBP": [220180, 220051],
    "FiO2": [223835, 227009, 227010],
}

ABG_ITEMIDS = {
    "pH": [50820],
    "PaO2": [50821],
    "PaCO2": [50818],
    "HCO3": [50882],
}

TIME_WINDOWS = [
    ("t_180", 150, 180),
    ("t_150", 120, 150),
    ("t_120", 90, 120),
    ("t_90", 60, 90),
    ("t_60", 30, 60),
    ("t_30", 0, 30),
]

RAW_FEATURES_PATH = "data/features_raw.csv"


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def _build_vital_query(subject_ids: List[int], hadm_ids: List[int]) -> str:
    all_itemids = [id_ for ids in VITAL_ITEMIDS.values() for id_ in ids]
    return f"""
    SELECT ce.subject_id, ce.hadm_id, ce.charttime, ce.itemid, ce.valuenum
    FROM public.chartevents ce
    WHERE ce.subject_id IN ({','.join(map(str, subject_ids))})
      AND ce.hadm_id IN ({','.join(map(str, hadm_ids))})
      AND ce.itemid IN ({','.join(map(str, all_itemids))})
      AND ce.valuenum IS NOT NULL
    ORDER BY ce.subject_id, ce.hadm_id, ce.charttime
    """


def _build_abg_query(subject_ids: List[int], hadm_ids: List[int]) -> str:
    all_itemids = [id_ for ids in ABG_ITEMIDS.values() for id_ in ids]
    return f"""
    SELECT le.subject_id, le.hadm_id, le.charttime, le.itemid, le.valuenum
    FROM public.labevents le
    WHERE le.subject_id IN ({','.join(map(str, subject_ids))})
      AND le.hadm_id IN ({','.join(map(str, hadm_ids))})
      AND le.itemid IN ({','.join(map(str, all_itemids))})
      AND le.valuenum IS NOT NULL
    ORDER BY le.subject_id, le.hadm_id, le.charttime
    """


def _build_admission_query(subject_ids: List[int], hadm_ids: List[int]) -> str:
    return f"""
    SELECT subject_id, hadm_id, admittime, dischtime
    FROM public.admissions
    WHERE subject_id IN ({','.join(map(str, subject_ids))})
      AND hadm_id IN ({','.join(map(str, hadm_ids))})
    """


def _build_procedure_query(subject_ids: List[int], hadm_ids: List[int]) -> str:
    return f"""
    WITH chest_tube_admissions AS (
        SELECT DISTINCT subject_id, hadm_id
        FROM public.procedures_icd
        WHERE icd_version = 9 AND icd_code IN ('3491', '3404')
    ),
    chest_tube_items AS (
        SELECT itemid FROM public.d_items
        WHERE LOWER(label) LIKE '%chest tube%' AND LOWER(label) LIKE '%placed%'
    )
    SELECT pe.subject_id, pe.hadm_id, MIN(pe.starttime) as procedure_time
    FROM public.procedureevents pe
    INNER JOIN chest_tube_admissions cta
        ON pe.subject_id = cta.subject_id AND pe.hadm_id = cta.hadm_id
    LEFT JOIN chest_tube_items cti ON pe.itemid = cti.itemid
    WHERE pe.subject_id IN ({','.join(map(str, subject_ids))})
      AND pe.hadm_id IN ({','.join(map(str, hadm_ids))})
      AND (pe.itemid = 225433 OR cti.itemid IS NOT NULL)
    GROUP BY pe.subject_id, pe.hadm_id
    """


def load_cohort_data(
    gold_path: str = "data/cohort_gold.csv",
    clean_path: str = "data/cohort_clean.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("코호트 데이터 로딩...")
    
    gold_df = pd.read_csv(gold_path)
    clean_df = pd.read_csv(clean_path)
    
    gold_df["label"] = 1
    clean_df["label"] = 0
    
    print(f"  GOLD (Positive): {len(gold_df)}개")
    print(f"  CLEAN (Negative): {len(clean_df)}개")
    
    return gold_df, clean_df


def load_silver_cohort(path: str = "data/cohort_silver.csv") -> pd.DataFrame:
    silver_df = pd.read_csv(path)
    silver_df["label"] = -1
    print(f"  SILVER (Weak Label): {len(silver_df)}개")
    return silver_df


def _itemid_to_name(itemid: int) -> str:
    for name, ids in {**VITAL_ITEMIDS, **ABG_ITEMIDS}.items():
        if itemid in ids:
            return name
    return f"unknown_{itemid}"


def _calculate_slope(values: np.ndarray, times: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    try:
        if hasattr(times[0], "total_seconds"):
            time_mins = np.array([t.total_seconds() / 60 for t in times])
        else:
            time_mins = np.arange(len(values))
        return np.polyfit(time_mins, values, 1)[0]
    except Exception:
        return np.nan


def _calculate_cv(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    mean_val = np.nanmean(values)
    if mean_val == 0 or np.isnan(mean_val):
        return np.nan
    return np.nanstd(values) / abs(mean_val)


def _calculate_iqr(values: np.ndarray) -> float:
    if len(values) < 4:
        return np.nan
    try:
        q1 = np.nanpercentile(values, 25)
        q3 = np.nanpercentile(values, 75)
        return q3 - q1
    except Exception:
        return np.nan


def _calculate_range(values: np.ndarray) -> float:
    if len(values) < 1:
        return np.nan
    return np.nanmax(values) - np.nanmin(values)


def _calculate_aa_gradient(pao2: float, paco2: float, fio2: float) -> float:
    if np.isnan(pao2) or np.isnan(paco2) or np.isnan(fio2):
        return np.nan
    
    if fio2 > 1:
        fio2 = fio2 / 100
    
    Patm = 760
    PH2O = 47
    R = 0.8
    
    pao2_alveolar = fio2 * (Patm - PH2O) - (paco2 / R)
    
    return pao2_alveolar - pao2


def _calculate_baseline_change(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 2:
        return np.nan
    return valid_values[-1] - valid_values[0]


def _calculate_baseline_change_pct(values: np.ndarray) -> float:
    if len(values) < 2:
        return np.nan
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 2:
        return np.nan
    baseline = valid_values[0]
    if baseline == 0:
        return np.nan
    return (valid_values[-1] - baseline) / abs(baseline)


def extract_summary_features(
    vital_df: pd.DataFrame,
    abg_df: pd.DataFrame,
    reference_times: pd.DataFrame,
    window_hours: float = 3.0,
) -> pd.DataFrame:
    features_list = []
    window_delta = timedelta(hours=window_hours)

    for _, row in reference_times.iterrows():
        subject_id = row["subject_id"]
        hadm_id = row["hadm_id"]
        ref_time = pd.to_datetime(row["reference_time"])
        start_time = ref_time - window_delta

        features = {"subject_id": subject_id, "hadm_id": hadm_id}
        vital_agg = {}
        abg_agg = {}

        mask = (
            (vital_df["subject_id"] == subject_id)
            & (vital_df["hadm_id"] == hadm_id)
            & (vital_df["charttime"] >= start_time)
            & (vital_df["charttime"] <= ref_time)
        )
        patient_vitals = vital_df[mask]

        for var_name in VITAL_ITEMIDS:
            var_data = patient_vitals[patient_vitals["var_name"] == var_name].sort_values("charttime")
            values = var_data["valuenum"].values

            features[f"{var_name}_mean"] = np.nanmean(values) if len(values) > 0 else np.nan
            features[f"{var_name}_std"] = np.nanstd(values) if len(values) > 1 else np.nan
            features[f"{var_name}_min"] = np.nanmin(values) if len(values) > 0 else np.nan
            features[f"{var_name}_max"] = np.nanmax(values) if len(values) > 0 else np.nan

            features[f"{var_name}_cv"] = _calculate_cv(values)
            features[f"{var_name}_iqr"] = _calculate_iqr(values)
            features[f"{var_name}_range"] = _calculate_range(values)

            features[f"{var_name}_count"] = len(values)

            features[f"{var_name}_baseline_change"] = _calculate_baseline_change(values)
            features[f"{var_name}_baseline_change_pct"] = _calculate_baseline_change_pct(values)

            if len(var_data) >= 2:
                times = var_data["charttime"].values
                time_diff = [(t - times[0]) / np.timedelta64(1, "m") for t in times]
                features[f"{var_name}_slope"] = _calculate_slope(values, np.array(time_diff))
            else:
                features[f"{var_name}_slope"] = np.nan

            vital_agg[var_name] = features[f"{var_name}_mean"]

        mask = (
            (abg_df["subject_id"] == subject_id)
            & (abg_df["hadm_id"] == hadm_id)
            & (abg_df["charttime"] >= start_time)
            & (abg_df["charttime"] <= ref_time)
        )
        patient_abg = abg_df[mask]

        for var_name in ABG_ITEMIDS:
            var_data = patient_abg[patient_abg["var_name"] == var_name].sort_values("charttime")
            values = var_data["valuenum"].values

            features[f"{var_name}_mean"] = np.nanmean(values) if len(values) > 0 else np.nan
            features[f"{var_name}_min"] = np.nanmin(values) if len(values) > 0 else np.nan
            features[f"{var_name}_max"] = np.nanmax(values) if len(values) > 0 else np.nan
            features[f"{var_name}_std"] = np.nanstd(values) if len(values) > 1 else np.nan

            features[f"{var_name}_cv"] = _calculate_cv(values)
            features[f"{var_name}_iqr"] = _calculate_iqr(values)
            features[f"{var_name}_range"] = _calculate_range(values)

            features[f"{var_name}_count"] = len(values)

            features[f"{var_name}_baseline_change"] = _calculate_baseline_change(values)
            features[f"{var_name}_baseline_change_pct"] = _calculate_baseline_change_pct(values)

            abg_agg[var_name] = features[f"{var_name}_mean"]

        sbp, dbp = vital_agg.get("SBP", np.nan), vital_agg.get("DBP", np.nan)
        hr = vital_agg.get("HR", np.nan)
        pao2 = abg_agg.get("PaO2", np.nan)
        paco2 = abg_agg.get("PaCO2", np.nan)
        fio2 = vital_agg.get("FiO2", np.nan)

        features["MAP_mean"] = dbp + (sbp - dbp) / 3 if not (np.isnan(sbp) or np.isnan(dbp)) else np.nan
        features["SHOCK_INDEX"] = hr / sbp if not (np.isnan(hr) or np.isnan(sbp)) and sbp > 0 else np.nan
        
        if not (np.isnan(pao2) or np.isnan(fio2)) and fio2 > 0:
            fio2_adj = fio2 / 100 if fio2 > 1 else fio2
            features["PF_RATIO"] = pao2 / fio2_adj
        else:
            features["PF_RATIO"] = np.nan

        features["AA_GRADIENT"] = _calculate_aa_gradient(pao2, paco2, fio2)

        features_list.append(features)

    return pd.DataFrame(features_list)


DELTA_PAIRS = [
    ("t_30", "t_60"),
    ("t_60", "t_90"),
    ("t_90", "t_120"),
    ("t_120", "t_150"),
    ("t_150", "t_180"),
]


def extract_sequence_features(
    vital_df: pd.DataFrame,
    abg_df: pd.DataFrame,
    reference_times: pd.DataFrame,
) -> pd.DataFrame:
    features_list = []

    for _, row in reference_times.iterrows():
        subject_id = row["subject_id"]
        hadm_id = row["hadm_id"]
        ref_time = pd.to_datetime(row["reference_time"])

        features = {"subject_id": subject_id, "hadm_id": hadm_id}
        
        window_values = {var: {} for var in list(VITAL_ITEMIDS.keys()) + list(ABG_ITEMIDS.keys())}

        for window_name, start_min, end_min in TIME_WINDOWS:
            start_time = ref_time - timedelta(minutes=end_min)
            end_time = ref_time - timedelta(minutes=start_min)

            mask = (
                (vital_df["subject_id"] == subject_id)
                & (vital_df["hadm_id"] == hadm_id)
                & (vital_df["charttime"] >= start_time)
                & (vital_df["charttime"] < end_time)
            )
            window_vitals = vital_df[mask]

            for var_name in VITAL_ITEMIDS:
                var_data = window_vitals[window_vitals["var_name"] == var_name]
                values = var_data["valuenum"].values
                
                mean_val = np.nanmean(values) if len(values) > 0 else np.nan
                features[f"{var_name}_{window_name}"] = mean_val
                
                features[f"{var_name}_{window_name}_min"] = np.nanmin(values) if len(values) > 0 else np.nan
                features[f"{var_name}_{window_name}_max"] = np.nanmax(values) if len(values) > 0 else np.nan
                
                features[f"{var_name}_{window_name}_count"] = len(values)
                
                window_values[var_name][window_name] = mean_val

            mask = (
                (abg_df["subject_id"] == subject_id)
                & (abg_df["hadm_id"] == hadm_id)
                & (abg_df["charttime"] >= start_time)
                & (abg_df["charttime"] < end_time)
            )
            window_abg = abg_df[mask]

            for var_name in ABG_ITEMIDS:
                var_data = window_abg[window_abg["var_name"] == var_name]
                values = var_data["valuenum"].values
                
                mean_val = np.nanmean(values) if len(values) > 0 else np.nan
                features[f"{var_name}_{window_name}"] = mean_val
                
                features[f"{var_name}_{window_name}_min"] = np.nanmin(values) if len(values) > 0 else np.nan
                features[f"{var_name}_{window_name}_max"] = np.nanmax(values) if len(values) > 0 else np.nan
                
                features[f"{var_name}_{window_name}_count"] = len(values)
                
                window_values[var_name][window_name] = mean_val

        all_vars = list(VITAL_ITEMIDS.keys()) + list(ABG_ITEMIDS.keys())
        for var_name in all_vars:
            for recent_window, prev_window in DELTA_PAIRS:
                recent_val = window_values[var_name].get(recent_window, np.nan)
                prev_val = window_values[var_name].get(prev_window, np.nan)
                
                delta = recent_val - prev_val if not (np.isnan(recent_val) or np.isnan(prev_val)) else np.nan
                features[f"{var_name}_delta_{recent_window}_{prev_window}"] = delta
                
                if not np.isnan(delta) and prev_val != 0 and not np.isnan(prev_val):
                    features[f"{var_name}_delta_pct_{recent_window}_{prev_window}"] = delta / abs(prev_val)
                else:
                    features[f"{var_name}_delta_pct_{recent_window}_{prev_window}"] = np.nan

        features_list.append(features)

    return pd.DataFrame(features_list)


def _process_batch(args: Tuple) -> Optional[pd.DataFrame]:
    batch_df, batch_idx, n_batches, use_sequence, db_config = args

    try:
        conn = psycopg2.connect(**db_config)
    except Exception as e:
        print(f"  배치 {batch_idx + 1}/{n_batches} DB 연결 실패: {e}")
        return None

    try:
        subject_ids = batch_df["subject_id"].tolist()
        hadm_ids = batch_df["hadm_id"].tolist()

        admit_df = pd.read_sql_query(_build_admission_query(subject_ids, hadm_ids), conn)
        try:
            proc_df = pd.read_sql_query(_build_procedure_query(subject_ids, hadm_ids), conn)
        except Exception:
            proc_df = pd.DataFrame(columns=["subject_id", "hadm_id", "procedure_time"])

        ref_times = batch_df[["subject_id", "hadm_id", "label"]].copy()
        ref_times = ref_times.merge(admit_df[["subject_id", "hadm_id", "admittime"]], on=["subject_id", "hadm_id"], how="left")
        ref_times = ref_times.merge(proc_df, on=["subject_id", "hadm_id"], how="left")

        def _get_ref_time(row):
            if row["label"] == 1:
                return row["procedure_time"] if pd.notna(row.get("procedure_time")) else None
            return pd.to_datetime(row["admittime"]) + timedelta(hours=24)

        ref_times["reference_time"] = ref_times.apply(_get_ref_time, axis=1)
        ref_times = ref_times.dropna(subset=["reference_time"])

        if ref_times.empty:
            return None

        vital_df = pd.read_sql_query(_build_vital_query(subject_ids, hadm_ids), conn)
        vital_df["charttime"] = pd.to_datetime(vital_df["charttime"])
        vital_df["var_name"] = vital_df["itemid"].apply(_itemid_to_name)

        abg_df = pd.read_sql_query(_build_abg_query(subject_ids, hadm_ids), conn)
        abg_df["charttime"] = pd.to_datetime(abg_df["charttime"])
        abg_df["var_name"] = abg_df["itemid"].apply(_itemid_to_name)

        summary_feat = extract_summary_features(vital_df, abg_df, ref_times)

        if use_sequence:
            seq_feat = extract_sequence_features(vital_df, abg_df, ref_times)
            batch_feat = summary_feat.merge(seq_feat, on=["subject_id", "hadm_id"])
        else:
            batch_feat = summary_feat

        batch_feat = batch_feat.merge(ref_times[["subject_id", "hadm_id", "label"]], on=["subject_id", "hadm_id"])
        return batch_feat

    except Exception as e:
        print(f"  배치 {batch_idx + 1}/{n_batches} 처리 오류: {e}")
        return None
    finally:
        conn.close()


def fetch_all_features(
    cohort_df: pd.DataFrame,
    use_sequence: bool = True,
    batch_size: int = 500,
    n_workers: Optional[int] = None,
) -> pd.DataFrame:
    print(f"\n총 {len(cohort_df)}개 샘플 Feature 추출 시작...")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 18)

    n_batches = (len(cohort_df) + batch_size - 1) // batch_size
    print(f"  배치: {n_batches}개, 워커: {n_workers}개")

    batch_args = [
        (cohort_df.iloc[i * batch_size : (i + 1) * batch_size].copy(), i, n_batches, use_sequence, DB_CONFIG)
        for i in range(n_batches)
    ]

    all_features = []

    if n_workers > 1 and n_batches > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_process_batch, args): args[1] for args in batch_args}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result is not None and len(result) > 0:
                    all_features.append(result)
                if completed % max(1, n_batches // 10) == 0 or completed == n_batches:
                    print(f"  진행: {completed}/{n_batches} ({completed / n_batches * 100:.0f}%)")
    else:
        for args in batch_args:
            result = _process_batch(args)
            if result is not None:
                all_features.append(result)

    if not all_features:
        print("Feature 추출 실패")
        return pd.DataFrame()

    final_df = pd.concat(all_features, ignore_index=True)
    print(f"총 {len(final_df)}개 샘플 Feature 추출 완료")
    return final_df


def save_raw_features(features_df: pd.DataFrame, output_path: str = RAW_FEATURES_PATH):
    features_df.to_csv(output_path, index=False)
    feature_cols = [c for c in features_df.columns if c not in ["subject_id", "hadm_id", "label"]]
    print(f"Raw Features 저장: {output_path}")
    print(f"  샘플: {len(features_df)}개, Features: {len(feature_cols)}개")
    
    missing_rates = features_df[feature_cols].isna().mean() * 100
    print(f"  평균 결측률: {missing_rates.mean():.2f}%")
    print(f"  결측률 범위: {missing_rates.min():.2f}% ~ {missing_rates.max():.2f}%")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    meta_cols = {"subject_id", "hadm_id", "label", "cohort_type", "group_label", "ref_time", "window_start"}
    return [c for c in df.columns if c not in meta_cols]


def run_feature_extraction(
    gold_path: str = "data/cohort_gold.csv",
    clean_path: str = "data/cohort_clean.csv",
    silver_path: Optional[str] = "data/cohort_silver.csv",
    output_path: str = RAW_FEATURES_PATH,
    use_sequence: bool = True,
    clean_sample_ratio: float = 0.1,
    batch_size: int = 500,
    n_workers: Optional[int] = None,
    use_silver: bool = False,
) -> pd.DataFrame:
    
    gold_df, clean_df = load_cohort_data(gold_path, clean_path)
    
    silver_df = None
    if use_silver and silver_path:
        try:
            silver_df = load_silver_cohort(silver_path)
        except FileNotFoundError:
            print(f"  {silver_path} 없음")
    
    n_clean = int(len(clean_df) * clean_sample_ratio)
    n_clean = max(n_clean, len(gold_df) * 10)
    n_clean = min(n_clean, len(clean_df))
    clean_sampled = clean_df.sample(n=n_clean, random_state=42)
    print(f"\nCLEAN 샘플링: {len(clean_df)} -> {len(clean_sampled)}")
    
    cohorts = [gold_df, clean_sampled]
    if silver_df is not None:
        cohorts.append(silver_df)
    cohort_df = pd.concat(cohorts, ignore_index=True)
    print(f"cohort_df: {len(cohort_df)}개")
    
    features_df = fetch_all_features(cohort_df, use_sequence, batch_size, n_workers)
    
    if len(features_df) == 0:
        print("features_df == 0")
        return pd.DataFrame()
    
    save_raw_features(features_df, output_path)
    
    return features_df


if __name__ == "__main__":
    config = {
        "gold_path": "data/cohort_gold.csv",
        "clean_path": "data/cohort_clean.csv",
        "silver_path": "data/cohort_silver.csv",
        "output_path": "data/features_raw.csv",
        "use_sequence": True,
        "clean_sample_ratio": 0.05,
        "batch_size": 500,
        "n_workers": None,
        "use_silver": True,
    }
    
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    features_df = run_feature_extraction(**config)
