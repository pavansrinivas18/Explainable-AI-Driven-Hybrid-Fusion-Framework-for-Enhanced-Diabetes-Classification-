"""
Adaptive Ensemble Learning Module.
Synergizes predictions from the three parallel models:
1. KCCAM-DNN (Kendall Attention Deep Neural Network)
2. FT-Transformer (Tabular Feature Tokenizer Self-Attention)
3. TabPFN (Bayesian Prior-Data Fitted Foundation Network)

Uses an uncertainty-aware dynamic meta-weighting mechanism to produce
the final calibrated diabetes risk probability.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from src.config import RANDOM_SEED


class AdaptiveEnsemble:
    """
    Adaptive Meta-Learner combining KCCAM-DNN, FT-Transformer, and TabPFN.
    """
    def __init__(self, kccam_model, ft_model, tabpfn_model, random_state=RANDOM_SEED):
        self.kccam = kccam_model
        self.ft = ft_model
        self.tabpfn = tabpfn_model
        self.random_state = random_state
        self.meta_learner = None
        self.static_weights = np.array([0.35, 0.35, 0.30])  # Default baseline weights

    def fit_meta_learner(self, X_val, y_val):
        """
        Fits a meta-classifier on validation predictions to learn optimal blending weights.
        """
        p_kccam = self.kccam.predict_proba(X_val)[:, 1]
        p_ft = self.ft.predict_proba(X_val)[:, 1]
        p_tabpfn = self.tabpfn.predict_proba(X_val)[:, 1]

        # Stack into meta-feature matrix: (N, 3)
        meta_features = np.column_stack([p_kccam, p_ft, p_tabpfn])

        meta_clf = LogisticRegression(
            C=1.0,
            penalty="l2",
            random_state=self.random_state,
            solver="lbfgs"
        )
        meta_clf.fit(meta_features, y_val)
        self.meta_learner = meta_clf

        # Extract normalized positive coefficients as baseline weights
        coefs = np.maximum(meta_clf.coef_[0], 0.01)
        self.static_weights = coefs / np.sum(coefs)
        print(f"[ADAPTIVE ENSEMBLE] Fitted meta-learner. Baseline weights:")
        print(f"  - KCCAM-DNN      : {self.static_weights[0]:.3f}")
        print(f"  - FT-Transformer : {self.static_weights[1]:.3f}")
        print(f"  - TabPFN         : {self.static_weights[2]:.3f}")
        return self

    def predict_components(self, X):
        """
        Returns predictions from each parallel model individually.
        """
        p_kccam = self.kccam.predict_proba(X)[:, 1]
        p_ft = self.ft.predict_proba(X)[:, 1]
        p_tabpfn = self.tabpfn.predict_proba(X)[:, 1]
        return {
            "KCCAM-DNN": p_kccam,
            "FT-Transformer": p_ft,
            "TabPFN": p_tabpfn
        }

    def predict_proba(self, X) -> np.ndarray:
        """
        Generates ensemble probability with dynamic uncertainty-aware weighting:
        More confident models receive higher weight for any given patient.
        """
        preds = self.predict_components(X)
        p_kccam = preds["KCCAM-DNN"]
        p_ft = preds["FT-Transformer"]
        p_tabpfn = preds["TabPFN"]

        meta_features = np.column_stack([p_kccam, p_ft, p_tabpfn])

        if self.meta_learner is not None:
            # Calibrated probability from logistic meta-learner
            ensemble_p1 = self.meta_learner.predict_proba(meta_features)[:, 1]
        else:
            # Dynamic uncertainty-aware certainty weighting
            # Certainty c_m = 2 * |p_m - 0.5|  (0 = max entropy/uncertain, 1 = absolute certainty)
            c1 = 2.0 * np.abs(p_kccam - 0.5)
            c2 = 2.0 * np.abs(p_ft - 0.5)
            c3 = 2.0 * np.abs(p_tabpfn - 0.5)

            # Modulate baseline weights by sample certainty
            w1 = self.static_weights[0] * (1.0 + 0.5 * c1)
            w2 = self.static_weights[1] * (1.0 + 0.5 * c2)
            w3 = self.static_weights[2] * (1.0 + 0.5 * c3)
            sum_w = w1 + w2 + w3

            ensemble_p1 = (w1 * p_kccam + w2 * p_ft + w3 * p_tabpfn) / sum_w

        ensemble_p1 = np.clip(ensemble_p1, 0.0, 1.0)
        return np.column_stack([1.0 - ensemble_p1, ensemble_p1])

    def predict(self, X, threshold=0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= threshold).astype(int)
