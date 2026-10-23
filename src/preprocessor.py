"""
Data Preprocessing Pipeline.
Implements:
1. Data Cleaning & Biological Sanity Check (0s in Glucose, BP, BMI -> NaNs)
2. Leak-Free Stratified Train-Test Split (80/20)
3. Multivariate Missing Value Imputation (IterativeImputer fit strictly on train)
4. Outlier Handling (IQR-based clipping)
5. Class Balancing (SMOTE on training partition)
6. Feature Scaling (StandardScaler)
7. Feature Selection (Kendall's Tau & Mutual Information)
8. Pipeline serialization for production CDSS inference
"""

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

from src.config import (
    INTEGRATED_DATA_PATH,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    ALL_FEATURES,
    TARGET_COL,
    BIOLOGICAL_NONZERO_COLS,
    RANDOM_SEED
)


class ClinicalPreprocessor:
    def __init__(self, random_state=RANDOM_SEED):
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self.outlier_bounds = {}
        self.kendall_scores = {}
        self.mi_scores = {}
        self.feature_names = list(ALL_FEATURES)

    def clean_biological_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts clinically impossible zero values (e.g. Glucose=0, BP=0, BMI=0) into NaNs.
        """
        cleaned = df.copy()
        for col in BIOLOGICAL_NONZERO_COLS:
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].replace(0, np.nan)
        if "MeanArterialPressure" in cleaned.columns:
            cleaned["MeanArterialPressure"] = cleaned["MeanArterialPressure"].replace(0, np.nan)
        return cleaned

    def fit_transform_train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits all preprocessing components STRICTLY on the training fold to prevent data leakage.
        """
        X = X_train.copy()

        # 1. Clean biological zeros
        X = self.clean_biological_zeros(X)

        # 2. Fit and apply Multivariate Imputer
        self.imputer = IterativeImputer(
            max_iter=15,
            random_state=self.random_state,
            initial_strategy="median"
        )
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X[self.feature_names]),
            columns=self.feature_names,
            index=X.index
        )

        # 3. Fit outlier bounds (IQR 1.5x) on training data
        X_clipped = X_imputed.copy()
        for col in self.feature_names:
            q1 = X_clipped[col].quantile(0.01)
            q99 = X_clipped[col].quantile(0.99)
            self.outlier_bounds[col] = (q1, q99)
            X_clipped[col] = X_clipped[col].clip(q1, q99)

        # 4. Feature Selection: Kendall's Tau & Mutual Information
        for col in self.feature_names:
            tau, p_val = kendalltau(X_clipped[col], y_train)
            self.kendall_scores[col] = 0.0 if np.isnan(tau) else float(tau)

        mi = mutual_info_classif(
            X_clipped[self.feature_names],
            y_train,
            random_state=self.random_state
        )
        for col, score in zip(self.feature_names, mi):
            self.mi_scores[col] = float(score)

        # 5. Fit and apply Standard Scaler
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_clipped[self.feature_names]),
            columns=self.feature_names,
            index=X.index
        )

        return X_scaled

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms unseen test or live patient data using fitted parameters.
        """
        if self.imputer is None or self.scaler is None:
            raise ValueError("Preprocessor has not been fitted yet!")

        X_curr = self.clean_biological_zeros(X)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_curr[self.feature_names]),
            columns=self.feature_names,
            index=X_curr.index
        )

        # Apply training outlier bounds
        X_clipped = X_imputed.copy()
        for col in self.feature_names:
            if col in self.outlier_bounds:
                q1, q99 = self.outlier_bounds[col]
                X_clipped[col] = X_clipped[col].clip(q1, q99)

        # Apply training standard scaling
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_clipped[self.feature_names]),
            columns=self.feature_names,
            index=X_curr.index
        )
        return X_scaled

    def get_kendall_vector(self) -> np.ndarray:
        """
        Returns the absolute Kendall Tau vector for guiding KCCAM-DNN attention.
        """
        return np.array([abs(self.kendall_scores.get(col, 0.0)) for col in self.feature_names], dtype=np.float32)

    def save(self, filepath):
        data = {
            "imputer": self.imputer,
            "scaler": self.scaler,
            "outlier_bounds": self.outlier_bounds,
            "kendall_scores": self.kendall_scores,
            "mi_scores": self.mi_scores,
            "feature_names": self.feature_names,
            "random_state": self.random_state
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath):
        data = joblib.load(filepath)
        obj = cls(random_state=data.get("random_state", RANDOM_SEED))
        obj.imputer = data["imputer"]
        obj.scaler = data["scaler"]
        obj.outlier_bounds = data["outlier_bounds"]
        obj.kendall_scores = data["kendall_scores"]
        obj.mi_scores = data["mi_scores"]
        obj.feature_names = data["feature_names"]
        return obj


def run_preprocessing_pipeline(save_artifacts=True):
    """
    Orchestrates full leak-free data preprocessing:
    - Loads integrated dataset
    - Stratified train-test split (80% train, 20% test)
    - Fits preprocessor on train, transforms test
    - Performs SMOTE class balancing on training split
    - Saves processed files and serialized preprocessor
    """
    df = pd.read_csv(INTEGRATED_DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET_COL]

    # Stratified 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = ClinicalPreprocessor(random_state=RANDOM_SEED)
    X_train_scaled = preprocessor.fit_transform_train(X_train, y_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Class balancing with SMOTE strictly on training fold
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_SEED)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    except ImportError:
        # Fallback if imblearn is not installed: training weights will handle class balance
        X_train_balanced, y_train_balanced = X_train_scaled, y_train

    if save_artifacts:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        X_train_balanced.to_csv(PROCESSED_DATA_DIR / "X_train_processed.csv", index=False)
        y_train_balanced.to_csv(PROCESSED_DATA_DIR / "y_train_processed.csv", index=False)
        X_test_scaled.to_csv(PROCESSED_DATA_DIR / "X_test_processed.csv", index=False)
        y_test.to_csv(PROCESSED_DATA_DIR / "y_test_processed.csv", index=False)

        # Also save unscaled train for TabPFN / tree models
        X_train.to_csv(PROCESSED_DATA_DIR / "X_train_raw.csv", index=False)
        X_test.to_csv(PROCESSED_DATA_DIR / "X_test_raw.csv", index=False)

        # Save preprocessor object via robust dict serialization
        preprocessor_path = MODELS_DIR / "clinical_preprocessor.joblib"
        preprocessor.save(preprocessor_path)

        print("[PREPROCESSING] Completed successfully.")
        print(f"[PREPROCESSING] Training samples: {len(X_train_balanced)}, Test samples: {len(X_test_scaled)}")
        print(f"[PREPROCESSING] Serialized preprocessor saved to: {preprocessor_path}")
        print("\n[PREPROCESSING] Feature Importance ranking via Kendall's Tau (monotonic association):")
        sorted_kendall = sorted(preprocessor.kendall_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        for rank, (feat, score) in enumerate(sorted_kendall, 1):
            print(f"  {rank}. {feat:<26}: tau = {score:+.4f} (MI = {preprocessor.mi_scores.get(feat, 0.0):.4f})")

    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, preprocessor


if __name__ == "__main__":
    run_preprocessing_pipeline()
