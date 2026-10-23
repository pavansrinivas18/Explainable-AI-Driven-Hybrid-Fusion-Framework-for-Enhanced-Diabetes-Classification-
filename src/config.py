import os
from pathlib import Path
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for d in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Datasets
LOCAL_DATA_PATH = DATA_DIR / "local_diabetes_data.csv"
PIMA_DATA_PATH = DATA_DIR / "pima_diabetes_data.csv"
INTEGRATED_DATA_PATH = DATA_DIR / "integrated_diabetes_data.csv"

# Reproducibility
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clinical Feature Definitions
CORE_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

ENGINEERED_FEATURES = [
    "MeanArterialPressure",
    "Glucose_Insulin_Ratio",
    "CardioMetabolic_Index"
]

ALL_FEATURES = CORE_FEATURES + ENGINEERED_FEATURES
TARGET_COL = "Outcome"

# Biological Zero Sanity Checks (Features where 0 is physiologically invalid in living humans)
BIOLOGICAL_NONZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "BMI"]

# Clinical Reference Ranges & Units (for doctor interface & triage)
CLINICAL_RANGES = {
    "Pregnancies": {"min": 0, "max": 17, "unit": "count", "normal": "0-3"},
    "Glucose": {"min": 40.0, "max": 250.0, "unit": "mg/dL", "normal": "70-99 (Fasting)"},
    "BloodPressure": {"min": 50.0, "max": 180.0, "unit": "mmHg", "normal": "60-80 (Diastolic)"},
    "SkinThickness": {"min": 7.0, "max": 65.0, "unit": "mm", "normal": "10-30"},
    "Insulin": {"min": 0.0, "max": 800.0, "unit": "μU/mL", "normal": "15-160"},
    "BMI": {"min": 15.0, "max": 60.0, "unit": "kg/m²", "normal": "18.5-24.9"},
    "DiabetesPedigreeFunction": {"min": 0.05, "max": 2.50, "unit": "score", "normal": "< 0.50"},
    "Age": {"min": 18, "max": 90, "unit": "years", "normal": "-"}
}

# Clinical Triage Risk Tiers
RISK_TIERS = {
    "LOW": {"threshold": 0.30, "label": "Low Risk", "color": "#10B981"},
    "MODERATE": {"threshold": 0.70, "label": "Moderate Risk", "color": "#F59E0B"},
    "HIGH": {"threshold": 1.00, "label": "High Risk", "color": "#EF4444"}
}

# Model Hyperparameters
KCCAM_PARAMS = {
    "hidden_dims": [128, 64, 32],
    "dropout_rate": 0.25,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "epochs": 120,
    "batch_size": 32,
    "lambda_kendall": 0.5
}

FT_TRANSFORMER_PARAMS = {
    "d_token": 64,
    "n_blocks": 3,
    "n_heads": 4,
    "dropout": 0.2,
    "lr": 0.0008,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 32
}

TABPFN_PARAMS = {
    "device": "cpu",
    "N_ensemble_configurations": 4
}
