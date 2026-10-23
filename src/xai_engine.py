"""
Explainable AI (XAI) Engine.
Integrates SHAP (Shapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations)
over the Adaptive Ensemble to provide clinical transparency for disease diagnosis.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread-safe plot generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.config import ALL_FEATURES, RANDOM_SEED


class XAIEngine:
    """
    Unified Explainability Engine for CDSS.
    Provides global feature importance, local patient-level waterfall plots,
    and LIME rule-based risk factors.
    """
    def __init__(self, ensemble_model, background_data: pd.DataFrame, feature_names=None):
        self.model = ensemble_model
        self.feature_names = feature_names or list(ALL_FEATURES)
        self.background_data = background_data[self.feature_names].values

        # Use sampled background for fast, responsive clinical SHAP kernel evaluations
        if len(self.background_data) > 12:
            self.shap_background = shap.sample(self.background_data, 12, random_state=RANDOM_SEED)
        else:
            self.shap_background = self.background_data

        # SHAP KernelExplainer over ensemble probability
        def predict_fn(x):
            return self.model.predict_proba(x)[:, 1]

        self.predict_fn = predict_fn
        self.shap_explainer = shap.KernelExplainer(self.predict_fn, self.shap_background)

        # LIME Tabular Explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=self.background_data,
            feature_names=self.feature_names,
            class_names=["Non-Diabetic", "Diabetic"],
            mode="classification",
            random_state=RANDOM_SEED
        )

    def explain_patient_shap(self, patient_features: pd.DataFrame, nsamples=100):
        """
        Computes SHAP attribution values for a single patient instance.
        Returns:
            shap_values: array of feature attributions
            base_value: baseline population risk
            patient_dict: feature name -> (value, shap_attribution)
        """
        x_patient = patient_features[self.feature_names].values
        shap_vals = self.shap_explainer.shap_values(x_patient, nsamples=nsamples)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[0]

        base_val = float(self.shap_explainer.expected_value)
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[0])

        explanation_data = []
        for feat, val, sv in zip(self.feature_names, x_patient[0], shap_vals):
            explanation_data.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(sv),
                "impact": "Increases Risk" if sv > 0 else "Decreases Risk",
                "abs_impact": abs(float(sv))
            })

        # Sort by absolute impact
        explanation_data.sort(key=lambda x: x["abs_impact"], reverse=True)

        return {
            "shap_values": shap_vals,
            "base_value": base_val,
            "patient_risk": float(self.predict_fn(x_patient)[0]),
            "feature_impacts": explanation_data
        }

    def generate_shap_waterfall_plot(self, patient_features: pd.DataFrame, save_path=None):
        """
        Renders a medical-grade visual Waterfall chart showing how patient vitals
        deviated their risk from the population baseline.
        """
        explanation = self.explain_patient_shap(patient_features)
        impacts = explanation["feature_impacts"][:8]  # Top 8 most influential vitals

        features = [i["feature"] for i in impacts][::-1]
        values = [i["shap_value"] * 100.0 for i in impacts][::-1]  # Percentage delta
        colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
        bars = ax.barh(features, values, color=colors, height=0.55, edgecolor="none", alpha=0.9)

        # Baseline & labels
        ax.axvline(0, color="#64748B", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Contribution to Predicted Risk (Percentage Points %)", fontsize=11, fontweight="bold", color="#1E293B")
        ax.set_title(
            f"Patient Risk Factor Attribution (SHAP Analysis)\nPredicted Risk: {explanation['patient_risk']:.1%} | Baseline: {explanation['base_value']:.1%}",
            fontsize=12, fontweight="bold", color="#0F172A", pad=12
        )

        # Add value annotations
        for bar in bars:
            width = bar.get_width()
            align = "left" if width >= 0 else "right"
            offset = 0.5 if width >= 0 else -0.5
            ax.text(
                width + offset,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:+.1f}%",
                va="center",
                ha=align,
                fontsize=9.5,
                fontweight="bold",
                color="#0F172A"
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        return fig, explanation

    def explain_patient_lime(self, patient_features: pd.DataFrame, num_features=6):
        """
        Generates LIME local surrogate explanations.
        """
        x_patient = patient_features[self.feature_names].values[0]

        def lime_predict_proba(x):
            return self.model.predict_proba(x)

        exp = self.lime_explainer.explain_instance(
            x_patient,
            lime_predict_proba,
            num_features=num_features
        )
        return exp
