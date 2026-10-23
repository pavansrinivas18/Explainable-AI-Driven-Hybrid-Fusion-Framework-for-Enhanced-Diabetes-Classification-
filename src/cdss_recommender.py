"""
Clinical Decision Support System (CDSS) Recommendation Engine.
Translates statistical predictions and SHAP biomarker attributions
into structured medical triage guidance adhering to clinical diabetes care guidelines.
"""

from src.config import RISK_TIERS, CLINICAL_RANGES


class CDSSRecommender:
    @staticmethod
    def generate_clinical_assessment(risk_prob: float, shap_impacts: list, patient_data: dict) -> dict:
        """
        Generates clinical assessment, triage category, and actionable guidance.
        """
        if risk_prob < RISK_TIERS["LOW"]["threshold"]:
            tier = "LOW"
            severity = "Low Risk / Non-Diabetic Profile"
            urgency = "Routine / Preventative"
            color = RISK_TIERS["LOW"]["color"]
            primary_rec = "Patient demonstrates low likelihood of diabetes. Continue standard preventative primary care."
            action_items = [
                "Schedule routine annual preventive health screening.",
                "Maintain healthy dietary patterns (rich in fiber, whole grains, low in processed sugars).",
                "Recommend minimum 150 minutes of moderate-intensity physical activity per week."
            ]
        elif risk_prob < RISK_TIERS["MODERATE"]["threshold"]:
            tier = "MODERATE"
            severity = "Moderate Risk / Pre-Diabetes Suspected"
            urgency = "Semi-Urgent / 3-Month Follow-Up"
            color = RISK_TIERS["MODERATE"]["color"]
            primary_rec = "Biomarkers indicate elevated risk of pre-diabetes or impaired glucose tolerance. Confirmatory testing warranted."
            action_items = [
                "Order laboratory Fasting Plasma Glucose (FPG) and Glycated Hemoglobin (HbA1c) test.",
                "Implement structured Medical Nutrition Therapy (MNT) with a registered dietitian.",
                "Target 5% to 7% intentional body weight reduction if BMI >= 25 kg/m².",
                "Repeat diagnostic evaluation in 3 to 6 months."
            ]
        else:
            tier = "HIGH"
            severity = "High Risk / Probable Diabetes Mellitus"
            urgency = "Urgent / Clinical Evaluation Required"
            color = RISK_TIERS["HIGH"]["color"]
            primary_rec = "High diagnostic probability of clinical diabetes. Immediate clinical workup and physician evaluation recommended."
            action_items = [
                "Urgent diagnostic confirmation with formal 2-hour Oral Glucose Tolerance Test (OGTT) or duplicate HbA1c.",
                "Referral to Endocrinology / Diabetes Care Team for comprehensive metabolic evaluation.",
                "Screen for microvascular complications: Urine albumin-to-creatinine ratio (uACR) and comprehensive dilated eye exam.",
                "Evaluate initiation of guideline-directed pharmacological therapy (e.g., Metformin or SGLT2i/GLP-1 RA if indicated).",
                "Educate patient on self-monitoring of blood glucose (SMBG) and symptoms of hyper/hypoglycemia."
            ]

        # Targeted biomarker advice based on top SHAP drivers
        targeted_insights = []
        for item in shap_impacts[:4]:
            feat = item["feature"]
            val = patient_data.get(feat, None)
            impact = item["impact"]

            if impact == "Increases Risk":
                if feat == "Glucose" and val and val > 125:
                    targeted_insights.append(f"Elevated Plasma Glucose ({val:.1f} mg/dL) is the primary driver of risk; fasting hyperglycemia suspected.")
                elif feat == "BMI" and val and val > 27.5:
                    targeted_insights.append(f"Elevated BMI ({val:.1f} kg/m²) significantly increases insulin resistance.")
                elif feat == "BloodPressure" and val and val > 85:
                    targeted_insights.append(f"Diastolic Blood Pressure ({val:.1f} mmHg) exceeds optimal targets; cardiovascular co-management advised.")
                elif feat == "Age" and val and val > 45:
                    targeted_insights.append(f"Age demographic ({int(val)} years) represents an unmodifiable metabolic risk multiplier.")
                elif feat == "CardioMetabolic_Index":
                    targeted_insights.append(f"Combined Cardiometabolic Index indicates heightened systemic metabolic burden.")

        return {
            "tier": tier,
            "risk_score_pct": round(risk_prob * 100.0, 1),
            "severity": severity,
            "urgency": urgency,
            "badge_color": color,
            "primary_recommendation": primary_rec,
            "clinical_action_items": action_items,
            "targeted_biomarker_insights": targeted_insights
        }
