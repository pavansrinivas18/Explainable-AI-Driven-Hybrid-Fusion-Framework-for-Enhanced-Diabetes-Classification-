"""
TabPFN (Tabular Prior-Data Fitted Network) Model Wrapper.
Implements the third parallel model from the research blueprint.
Leverages TabPFN's Bayesian transformer foundation model for tabular data,
with a robust fallback to Bayesian-calibrated gradient boosting in offline environments.
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from src.config import RANDOM_SEED


class TabPFNModelWrapper(BaseEstimator, ClassifierMixin):
    """
    Robust wrapper for TabPFN Classifier.
    Guarantees standard scikit-learn API: fit, predict, predict_proba.
    Also provides Bayesian uncertainty estimation (predictive entropy).
    """
    def __init__(self, device: str = "cpu", N_ensemble_configurations: int = 4, random_state: int = RANDOM_SEED):
        self.device = device
        self.N_ensemble_configurations = N_ensemble_configurations
        self.random_state = random_state
        self.model = None
        self.using_fallback = False

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y)

        try:
            from tabpfn import TabPFNClassifier
            # TabPFN attention is O(N^2); subsample 400 representative samples for optimal fast CPU inference
            if len(X_arr) > 400:
                indices = np.random.RandomState(self.random_state).choice(len(X_arr), size=400, replace=False)
                X_fit, y_fit = X_arr[indices], y_arr[indices]
            else:
                X_fit, y_fit = X_arr, y_arr

            tabpfn_clf = TabPFNClassifier(
                device=self.device,
                N_ensemble_configurations=self.N_ensemble_configurations
            )
            tabpfn_clf.fit(X_fit, y_fit)
            self.model = tabpfn_clf
            self.using_fallback = False
            print("[TabPFN] Successfully initialized native TabPFN Transformer foundation model.", flush=True)

        except Exception as e:
            warnings.warn(f"[TabPFN Notice] TabPFN native weights unavailable ({e}). "
                          f"Engaging Bayesian Calibrated Tabular surrogate.")
            # High-performance Bayesian Calibrated Ensemble surrogate
            base_estimator = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.08,
                random_state=self.random_state,
                class_weight="balanced"
            )
            calibrated = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=3)
            calibrated.fit(X_arr, y_arr)
            self.model = calibrated
            self.using_fallback = True

        return self

    def predict_proba(self, X):
        X_arr = np.array(X)
        if self.model is None:
            raise ValueError("TabPFN model is not fitted yet!")

        probs = self.model.predict_proba(X_arr)
        # Ensure proper shape (n_samples, 2)
        if probs.shape[1] == 1:
            probs = np.column_stack([1.0 - probs[:, 0], probs[:, 0]])
        return probs

    def predict(self, X, threshold: float = 0.5):
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)

    def get_uncertainty(self, X) -> np.ndarray:
        """
        Computes Shannon entropy as a measure of model prediction uncertainty:
        H(p) = - sum(p * log(p))
        Used by the Adaptive Ensemble for certainty-weighted meta-learning.
        """
        probs = np.clip(self.predict_proba(X), 1e-7, 1.0 - 1e-7)
        entropy = -np.sum(probs * np.log2(probs), axis=1)
        return entropy
