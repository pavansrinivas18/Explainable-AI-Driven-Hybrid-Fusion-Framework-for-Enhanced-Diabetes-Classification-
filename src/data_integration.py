"""
Data Integration Module.
Unifies PIMA Indian Dataset with Local Clinical Dataset into a harmonized cohort.
Performs unit harmonization (mmol/L to mg/dL for glucose, tenths-of-mm to mm for skin fold)
and clinical biomarker feature engineering (MAP, Glucose/Insulin, CardioMetabolic Index).
"""

import pandas as pd
import numpy as np
from src.config import (
    LOCAL_DATA_PATH,
    PIMA_DATA_PATH,
    INTEGRATED_DATA_PATH,
    BIOLOGICAL_NONZERO_COLS
)


def load_and_harmonize_local_data(filepath=LOCAL_DATA_PATH) -> pd.DataFrame:
    """
    Ingests Local Clinical Dataset and harmonizes units to clinical standards:
    - Glucose Consumption (recorded in mmol/L) -> converted to mg/dL (factor = 18.0182)
    - Skin Thickness (recorded in tenths of mm > 100) -> scaled to standard mm
    - BP(Systolic) and BP(Diastolic) -> Mean Arterial Pressure (MAP) & Diastolic BP
    - Genetic score -> Diabetes Pedigree Function equivalent
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    harmonized = pd.DataFrame()
    harmonized["Pregnancies"] = df["No. of Pregnancy"].astype(float)
    harmonized["Age"] = df["Age"].astype(float)
    harmonized["BMI"] = df["BMI"].astype(float)
    
    # Blood Pressure handling
    diastolic = df["BP(Diastolic)"].astype(float)
    systolic = df["BP(Systolic)"].astype(float)
    harmonized["BloodPressure"] = diastolic  # Standard diastolic pressure
    # Mean Arterial Pressure: DBP + (SBP - DBP) / 3
    harmonized["MeanArterialPressure"] = diastolic + (systolic - diastolic) / 3.0

    # Glucose conversion: Local data is in mmol/L -> convert to standard mg/dL
    glucose_mmol = df["Glucose Consumption"].astype(float)
    harmonized["Glucose"] = glucose_mmol * 18.0182

    # Skin thickness: values > 100 are recorded in tenths of mm (e.g. 317.5 = 31.75 mm)
    skin = df["Skin Thickness(mm)"].astype(float)
    harmonized["SkinThickness"] = skin.apply(lambda x: x / 10.0 if x > 100 else x)

    # Insulin
    harmonized["Insulin"] = df["Insulin"].astype(float)

    # Genetic factor: scale count (0-8) to pedigree score equivalent (~0.1 to 1.8)
    genetic = df["Genetic"].astype(float)
    harmonized["DiabetesPedigreeFunction"] = 0.15 + (genetic * 0.18)

    # Target outcome
    harmonized["Outcome"] = df["Outcome"].astype(int)
    harmonized["Cohort_Source"] = "Local_Cohort"

    return harmonized


def load_and_harmonize_pima_data(filepath=PIMA_DATA_PATH) -> pd.DataFrame:
    """
    Ingests benchmark PIMA dataset and calculates derived clinical biomarkers:
    - Mean Arterial Pressure (estimated from Diastolic BP: MAP ≈ DBP + 40/3 ≈ DBP + 13.33)
    """
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]

    harmonized = df.copy()
    for col in harmonized.columns:
        harmonized[col] = pd.to_numeric(harmonized[col], errors="coerce")

    # Estimated MAP for PIMA (DBP + 13.33 mmHg average pulse pressure contribution)
    harmonized["MeanArterialPressure"] = harmonized["BloodPressure"].apply(
        lambda dbp: (dbp + 13.33) if dbp > 0 else np.nan
    )
    harmonized["Cohort_Source"] = "PIMA_Benchmark"

    return harmonized


def engineer_biomarkers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes composite clinical indices to assist deep models (FT-Transformer & KCCAM-DNN):
    1. Glucose-to-Insulin Ratio: proxy for pancreatic beta-cell insulin resistance
    2. CardioMetabolic Index: (BMI * Glucose) / 100
    """
    engineered = df.copy()

    # Glucose-to-Insulin Ratio (safe handling of 0 or missing insulin)
    engineered["Glucose_Insulin_Ratio"] = engineered.apply(
        lambda r: (r["Glucose"] / (r["Insulin"] + 1.0)) if r["Insulin"] > 0 else (r["Glucose"] / 10.0),
        axis=1
    )

    # CardioMetabolic Index
    engineered["CardioMetabolic_Index"] = (engineered["BMI"] * engineered["Glucose"]) / 100.0

    return engineered


def integrate_datasets(save_output=True) -> pd.DataFrame:
    """
    Executes the full Data Integration pipeline.
    Combines Local Clinical Cohort and PIMA Benchmark into a single harmonized dataset.
    """
    local_df = load_and_harmonize_local_data()
    pima_df = load_and_harmonize_pima_data()

    # Align columns
    common_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        "MeanArterialPressure", "Outcome", "Cohort_Source"
    ]
    
    merged = pd.concat([local_df[common_cols], pima_df[common_cols]], ignore_index=True)
    
    # Feature Engineering
    integrated = engineer_biomarkers(merged)

    if save_output:
        integrated.to_csv(INTEGRATED_DATA_PATH, index=False)
        print(f"[DATA INTEGRATION] Unified dataset created at {INTEGRATED_DATA_PATH}")
        print(f"[DATA INTEGRATION] Total patient records: {len(integrated)} "
              f"({len(local_df)} Local + {len(pima_df)} PIMA)")
        print(f"[DATA INTEGRATION] Positive cases (Diabetic): {integrated['Outcome'].sum()} "
              f"({integrated['Outcome'].mean():.2%})")

    return integrated


if __name__ == "__main__":
    integrate_datasets()
