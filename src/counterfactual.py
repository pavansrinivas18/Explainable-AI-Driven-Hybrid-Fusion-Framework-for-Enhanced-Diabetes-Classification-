"""
Counterfactual & What-If Clinical Simulator.
Allows physicians to simulate actionable lifestyle and biomarker improvements
(e.g., reducing BMI, lowering glucose, optimizing blood pressure)
and quantify the resultant reduction in diabetes risk.
"""

import copy
import pandas as pd
import numpy as np


class CounterfactualSimulator:
    def __init__(self, ensemble_model, preprocessor):
        self.model = ensemble_model
        self.preprocessor = preprocessor

    def simulate_what_if(self, original_patient_df: pd.DataFrame, modifications: dict) -> dict:
        """
        Applies hypothetical clinical adjustments (e.g. {'Glucose': 105.0, 'BMI': 24.5}),
        recomputes derived biomarkers (MAP, Glucose/Insulin ratio, CardioMetabolic Index),
        transforms features, and computes the new risk probability.
        """
        modified_df = original_patient_df.copy()

        # Apply biomarker adjustments
        for feat, val in modifications.items():
            if feat in modified_df.columns:
                modified_df[feat] = float(val)

        # Re-derive composite biomarkers
        if "Glucose" in modified_df.columns and "BMI" in modified_df.columns:
            modified_df["CardioMetabolic_Index"] = (modified_df["BMI"] * modified_df["Glucose"]) / 100.0

        if "Glucose" in modified_df.columns and "Insulin" in modified_df.columns:
            ins = modified_df["Insulin"].values[0]
            glu = modified_df["Glucose"].values[0]
            modified_df["Glucose_Insulin_Ratio"] = (glu / (ins + 1.0)) if ins > 0 else (glu / 10.0)

        if "BloodPressure" in modified_df.columns:
            dbp = modified_df["BloodPressure"].values[0]
            modified_df["MeanArterialPressure"] = dbp + 13.33

        # Transform both baseline and modified
        baseline_scaled = self.preprocessor.transform(original_patient_df)
        modified_scaled = self.preprocessor.transform(modified_df)

        baseline_prob = float(self.model.predict_proba(baseline_scaled)[0, 1])
        modified_prob = float(self.model.predict_proba(modified_scaled)[0, 1])
        risk_reduction = baseline_prob - modified_prob

        return {
            "baseline_risk": baseline_prob,
            "simulated_risk": modified_prob,
            "absolute_reduction": risk_reduction,
            "relative_reduction_pct": (risk_reduction / max(baseline_prob, 1e-5)) * 100.0,
            "modified_features": modified_df.to_dict(orient="records")[0]
        }

    def compute_safe_target_trajectory(self, original_patient_df: pd.DataFrame, target_risk: float = 0.30) -> dict:
        """
        Iteratively determines the minimal lifestyle/biomarker adjustment
        needed to bring the patient's risk below the target threshold (default 30% - Low Risk).
        """
        curr_df = original_patient_df.copy()
        current_glucose = float(curr_df["Glucose"].values[0])
        current_bmi = float(curr_df["BMI"].values[0])

        trajectory = []
        # Simulate step-wise reductions in Glucose and BMI (5 steps)
        for step in range(1, 6):
            factor = step / 5.0
            target_glu = current_glucose - (current_glucose - 99.0) * factor if current_glucose > 99 else current_glucose
            target_bmi = current_bmi - (current_bmi - 24.0) * factor if current_bmi > 24 else current_bmi

            sim_result = self.simulate_what_if(curr_df, {"Glucose": target_glu, "BMI": target_bmi})
            trajectory.append({
                "step": step,
                "target_glucose": round(target_glu, 1),
                "target_bmi": round(target_bmi, 1),
                "resulting_risk": round(sim_result["simulated_risk"], 3),
                "risk_tier": "Low" if sim_result["simulated_risk"] < 0.30 else ("Moderate" if sim_result["simulated_risk"] < 0.70 else "High")
            })

        return trajectory
