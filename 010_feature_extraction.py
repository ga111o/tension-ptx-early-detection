import os
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import timedelta
from typing import List, Optional, Tuple

import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

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

RAW_DATA_PATH = "data/features_raw.csv"


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
    print("Loading cohort data...")
    
    gold_df = pd.read_csv(gold_path)
    clean_df = pd.read_csv(clean_path)
    
    gold_df["label"] = 1
    clean_df["label"] = 0
    
    print(f"GOLD Positive: {len(gold_df)}")
    print(f"CLEAN Negative: {len(clean_df)}")
    
    return gold_df, clean_df


def load_silver_cohort(path: str = "data/cohort_silver.csv") -> pd.DataFrame:
    silver_df = pd.read_csv(path)
    silver_df["label"] = -1
    print(f"SILVER Weak Label: {len(silver_df)}")
    return silver_df


def _itemid_to_name(itemid: int) -> str:
    for name, ids in {**VITAL_ITEMIDS, **ABG_ITEMIDS}.items():
        if itemid in ids:
            return name
    return f"unknown_{itemid}"


def _filter_by_reference_time(
    df: pd.DataFrame,
    ref_times: pd.DataFrame,
    window_hours: float = 3.0,
) -> pd.DataFrame:
    """Filter data within window_hours from reference_time"""
    window_delta = timedelta(hours=window_hours)
    
    filtered_rows = []
    for _, ref_row in ref_times.iterrows():
        subject_id = ref_row["subject_id"]
        hadm_id = ref_row["hadm_id"]
        ref_time = pd.to_datetime(ref_row["reference_time"])
        start_time = ref_time - window_delta
        
        mask = (
            (df["subject_id"] == subject_id)
            & (df["hadm_id"] == hadm_id)
            & (df["charttime"] >= start_time)
            & (df["charttime"] <= ref_time)
        )
        patient_data = df[mask].copy()
        
        if len(patient_data) > 0:
            patient_data["reference_time"] = ref_time
            patient_data["minutes_before_ref"] = (ref_time - patient_data["charttime"]).dt.total_seconds() / 60
            filtered_rows.append(patient_data)
    
    if filtered_rows:
        return pd.concat(filtered_rows, ignore_index=True)
    return pd.DataFrame()


def _process_batch(args: Tuple) -> Optional[pd.DataFrame]:
    batch_df, batch_idx, n_batches, window_hours, db_config = args

    try:
        conn = psycopg2.connect(**db_config)
    except Exception as e:
        print(f"Batch {batch_idx + 1}/{n_batches} DB connection failed: {e}")
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
        vital_df["data_type"] = "vital"

        abg_df = pd.read_sql_query(_build_abg_query(subject_ids, hadm_ids), conn)
        abg_df["charttime"] = pd.to_datetime(abg_df["charttime"])
        abg_df["var_name"] = abg_df["itemid"].apply(_itemid_to_name)
        abg_df["data_type"] = "abg"

        vital_filtered = _filter_by_reference_time(vital_df, ref_times, window_hours)
        abg_filtered = _filter_by_reference_time(abg_df, ref_times, window_hours)

        all_data = pd.concat([vital_filtered, abg_filtered], ignore_index=True)
        
        if len(all_data) == 0:
            return None

        all_data = all_data.merge(
            ref_times[["subject_id", "hadm_id", "label"]], 
            on=["subject_id", "hadm_id"], 
            how="left"
        )

        return all_data

    except Exception as e:
        print(f"Batch {batch_idx + 1}/{n_batches} processing error: {e}")
        return None
    finally:
        conn.close()


def fetch_raw_data(
    cohort_df: pd.DataFrame,
    window_hours: float = 3.0,
    batch_size: int = 500,
    n_workers: Optional[int] = None,
) -> pd.DataFrame: 
    print(f"\nExtracting raw data from {len(cohort_df)} samples...")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 12)

    n_batches = (len(cohort_df) + batch_size - 1) // batch_size
    print(f"Batches: {n_batches}, Workers: {n_workers}")

    batch_args = [
        (cohort_df.iloc[i * batch_size : (i + 1) * batch_size].copy(), i, n_batches, window_hours, DB_CONFIG)
        for i in range(n_batches)
    ]

    all_data = []

    if n_workers > 1 and n_batches > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_process_batch, args): args[1] for args in batch_args}
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()
                if result is not None and len(result) > 0:
                    all_data.append(result)
                if completed % max(1, n_batches // 10) == 0 or completed == n_batches:
                    print(f"Progress: {completed}/{n_batches} {completed / n_batches * 100:.0f}%")
    else:
        for args in batch_args:
            result = _process_batch(args)
            if result is not None:
                all_data.append(result)

    if not all_data:
        print("Raw data extraction failed")
        return pd.DataFrame()

    final_df = pd.concat(all_data, ignore_index=True)
    print(f"Extraction complete: {len(final_df)} records")
    return final_df


def save_raw_data(raw_df: pd.DataFrame, output_path: str = RAW_DATA_PATH):
    raw_df.to_csv(output_path, index=False)
    
    n_subjects = raw_df[["subject_id", "hadm_id"]].drop_duplicates().shape[0]
    n_records = len(raw_df)
    
    print(f"Raw data saved: {output_path}")
    print(f"Patients: {n_subjects}")
    print(f"Total records: {n_records}")
    
    if "var_name" in raw_df.columns:
        print(f"Records per variable:")
        for var_name, count in raw_df["var_name"].value_counts().items():
            print(f"  {var_name}: {count}")


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    meta_cols = {"subject_id", "hadm_id", "label", "cohort_type", "group_label", "ref_time", "window_start"}
    return [c for c in df.columns if c not in meta_cols]


def run_feature_extraction(
    gold_path: str = "data/cohort_gold.csv",
    clean_path: str = "data/cohort_clean.csv",
    silver_path: Optional[str] = "data/cohort_silver.csv",
    output_path: str = RAW_DATA_PATH,
    window_hours: float = 3.0,
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
            print(f"{silver_path} not found")
    
    n_clean = int(len(clean_df) * clean_sample_ratio)
    n_clean = max(n_clean, len(gold_df) * 10)
    n_clean = min(n_clean, len(clean_df))
    clean_sampled = clean_df.sample(n=n_clean, random_state=42)
    print(f"\nCLEAN sampling: {len(clean_df)} -> {len(clean_sampled)}")
    
    cohorts = [gold_df, clean_sampled]
    if silver_df is not None:
        cohorts.append(silver_df)
    cohort_df = pd.concat(cohorts, ignore_index=True)
    print(f"cohort_df: {len(cohort_df)} samples")
    
    raw_df = fetch_raw_data(cohort_df, window_hours, batch_size, n_workers)
    
    if len(raw_df) == 0:
        print("raw_df == 0")
        return pd.DataFrame()
    
    save_raw_data(raw_df, output_path)
    
    return raw_df


if __name__ == "__main__":
    config = {
        "gold_path": "data/cohort_gold.csv",
        "clean_path": "data/cohort_clean.csv",
        "silver_path": "data/cohort_silver.csv",
        "output_path": "data/features_raw.csv",
        "window_hours": 3.0,
        "clean_sample_ratio": 0.05,
        "batch_size": 500,
        "n_workers": None,
        "use_silver": True,
    }
    
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    raw_df = run_feature_extraction(**config)
