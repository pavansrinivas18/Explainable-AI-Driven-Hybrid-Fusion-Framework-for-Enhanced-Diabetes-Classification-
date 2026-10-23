"""
Clinician-Centric Explainable AI (XAI) Clinical Decision Support System (CDSS).
Provides an interactive medical dashboard for disease diagnosis (Diabetes Prediction).
Features:
- Patient Vitals & Lab Biomarkers Intake Form (with preset clinical patient cases)
- Parallel Model Consensus (KCCAM-DNN, FT-Transformer, TabPFN, Adaptive Ensemble)
- Diagnostic Triage Risk Meter (Low / Moderate / High)
- SHAP Patient-Level Waterfall Attribution & LIME Local Rules
- Counterfactual 'What-If' Lifestyle & Biomarker Simulator
- Comparative Model Validation Benchmark Table
- Exportable Clinical EHR Consultation Summary Note
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    ALL_FEATURES,
    CLINICAL_RANGES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    RISK_TIERS
)
from src.preprocessor import ClinicalPreprocessor
from src.counterfactual import CounterfactualSimulator
from src.cdss_recommender import CDSSRecommender
from src.xai_engine import XAIEngine


# Streamlit Page Setup
st.set_page_config(
    page_title="XAI Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Dashboard CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: #F8FAFC;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    .main-header h1 {
        color: #38BDF8;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    .main-header p {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .clinical-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0EA5E9;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_artifacts():
    """
    Loads serialized models, preprocessors, and background data.
    """
    preprocessor = ClinicalPreprocessor.load(MODELS_DIR / "clinical_preprocessor.joblib")
    ensemble = joblib.load(MODELS_DIR / "adaptive_ensemble.joblib")
    background_df = pd.read_csv(PROCESSED_DATA_DIR / "X_train_processed.csv")

    # Initialize XAI & Counterfactual engines
    xai_engine = XAIEngine(ensemble, background_data=background_df)
    cf_simulator = CounterfactualSimulator(ensemble, preprocessor)

    # Load comparative metrics
    metrics_df = None
    metrics_path = REPORTS_DIR / "model_comparison_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path, index_col=0)

    return preprocessor, ensemble, xai_engine, cf_simulator, metrics_df


# Header Section
st.markdown("""
<div class="main-header">
    <h1>🩺 Explainable AI (XAI) Clinical Decision Support System</h1>
    <p>Predictive Disease Diagnosis using <b>KCCAM-DNN</b>, <b>FT-Transformer</b>, <b>TabPFN</b>, and <b>SHAP</b></p>
</div>
""", unsafe_allow_html=True)

try:
    preprocessor, ensemble, xai_engine, cf_simulator, metrics_df = load_system_artifacts()
except Exception as e:
    st.error(f"Error loading system models: {e}. Please ensure `python -m src.train_and_evaluate` has been executed.")
    st.stop()


# -------------------------------------------------------------
# SIDEBAR: Patient Intake & Clinical Presets
# -------------------------------------------------------------
with st.sidebar:
    st.header("📋 Patient Clinical Intake")
    st.caption("Input lab biomarkers or choose a diagnostic preset:")

    preset = st.selectbox(
        "Load Clinical Case Study:",
        [
            "Custom Input",
            "Case 1: High-Risk Diabetic Crisis (Elevated Glucose & BMI)",
            "Case 2: Healthy Control (Normal Glycemic Profile)",
            "Case 3: Borderline Pre-Diabetic (Impaired Fasting Glucose)"
        ]
    )

    # Preset configurations
    if preset == "Case 1: High-Risk Diabetic Crisis (Elevated Glucose & BMI)":
        defaults = {"preg": 4, "glu": 178.0, "bp": 92.0, "skin": 36.0, "ins": 190.0, "bmi": 38.4, "dpf": 0.85, "age": 52}
    elif preset == "Case 2: Healthy Control (Normal Glycemic Profile)":
        defaults = {"preg": 1, "glu": 88.0, "bp": 68.0, "skin": 22.0, "ins": 50.0, "bmi": 22.5, "dpf": 0.25, "age": 27}
    elif preset == "Case 3: Borderline Pre-Diabetic (Impaired Fasting Glucose)":
        defaults = {"preg": 2, "glu": 118.0, "bp": 78.0, "skin": 28.0, "ins": 95.0, "bmi": 28.2, "dpf": 0.45, "age": 41}
    else:
        defaults = {"preg": 2, "glu": 135.0, "bp": 80.0, "skin": 25.0, "ins": 85.0, "bmi": 29.5, "dpf": 0.50, "age": 38}

    with st.form("patient_form"):
        st.subheader("Patient Vitals")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(defaults["age"]))
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=int(defaults["preg"]))
            glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=float(defaults["glu"]), step=1.0)
            bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=180.0, value=float(defaults["bp"]), step=1.0)
        with col_s2:
            bmi = st.number_input("BMI (kg/m²)", min_value=12.0, max_value=70.0, value=float(defaults["bmi"]), step=0.1)
            insulin = st.number_input("Serum Insulin (μU/mL)", min_value=0.0, max_value=900.0, value=float(defaults["ins"]), step=1.0)
            skin = st.number_input("Triceps Skinfold (mm)", min_value=5.0, max_value=80.0, value=float(defaults["skin"]), step=0.5)
            dpf = st.number_input("Diabetes Pedigree", min_value=0.05, max_value=3.0, value=float(defaults["dpf"]), step=0.01)

        submit_btn = st.form_submit_button("Run CDSS Diagnostic Evaluation ➔", use_container_width=True)

# Build Patient Record
patient_dict = {
    "Pregnancies": float(pregnancies),
    "Glucose": float(glucose),
    "BloodPressure": float(bp),
    "SkinThickness": float(skin),
    "Insulin": float(insulin),
    "BMI": float(bmi),
    "DiabetesPedigreeFunction": float(dpf),
    "Age": float(age),
    "MeanArterialPressure": float(bp) + 13.33,
    "Glucose_Insulin_Ratio": (float(glucose) / (float(insulin) + 1.0)) if float(insulin) > 0 else (float(glucose) / 10.0),
    "CardioMetabolic_Index": (float(bmi) * float(glucose)) / 100.0
}
patient_df = pd.DataFrame([patient_dict])
patient_scaled = preprocessor.transform(patient_df)


# -------------------------------------------------------------
# MAIN APP TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 Diagnostic Assessment & Consensus",
    "🔍 Explainable AI (SHAP & LIME)",
    "🔄 What-If Clinical Simulator",
    "📊 Population Model Validation",
    "📄 EHR Clinical Summary Note"
])

# Inference on Parallel Models
model_preds = ensemble.predict_components(patient_scaled)
p_kccam = float(model_preds["KCCAM-DNN"][0])
p_ft = float(model_preds["FT-Transformer"][0])
p_tabpfn = float(model_preds["TabPFN"][0])
ensemble_prob = float(ensemble.predict_proba(patient_scaled)[0, 1])

# XAI Explanation
explanation = xai_engine.explain_patient_shap(patient_scaled)
assessment = CDSSRecommender.generate_clinical_assessment(
    ensemble_prob, explanation["feature_impacts"], patient_dict
)


# -------------------------------------------------------------
# TAB 1: DIAGNOSTIC ASSESSMENT & CONSENSUS
# -------------------------------------------------------------
with tab1:
    col_m1, col_m2, col_m3 = st.columns([1.2, 1, 1])

    with col_m1:
        st.subheader("Diagnostic Risk Score")
        st.markdown(f"""
        <div class="metric-card">
            <h1 style="font-size: 3rem; margin: 0; color: {assessment['badge_color']};">
                {assessment['risk_score_pct']}%
            </h1>
            <p style="margin: 0; font-weight: 600; color: #475569;">Calibrated Probability of Diabetes</p>
            <div style="margin-top: 10px;">
                <span class="badge-{assessment['tier'].lower()}">{assessment['severity']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.subheader("Triage Category")
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0F172A; margin: 0;">Urgency Tier</h3>
            <p style="font-size: 1.15rem; font-weight: 700; color: {assessment['badge_color']}; margin-top: 0.5rem;">
                {assessment['urgency']}
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">ADA Clinical Guideline Benchmark</p>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.subheader("Top Clinical Driver")
        top_driver = explanation["feature_impacts"][0]
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0F172A; margin: 0;">Primary Risk Factor</h3>
            <p style="font-size: 1.15rem; font-weight: 700; color: #DC2626; margin-top: 0.5rem;">
                {top_driver['feature']}
            </p>
            <p style="font-size: 0.85rem; color: #64748B;">Impact: {top_driver['shap_value']*100:+.1f}% risk shift</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Parallel Models Consensus Panel
    st.subheader("🤖 Parallel Deep Learning Consensus View")
    st.caption("Cross-paradigm diagnosis from Kendall Attention DNN, Tabular Self-Attention Transformer, and Bayesian TabPFN:")

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        st.metric("1. KCCAM-DNN", f"{p_kccam:.1%}", help="Kendall Correlation Attention Deep Neural Network")
    with col_p2:
        st.metric("2. FT-Transformer", f"{p_ft:.1%}", help="Feature Tokenizer Multi-Head Self-Attention")
    with col_p3:
        st.metric("3. TabPFN", f"{p_tabpfn:.1%}", help="Prior-Data Fitted Bayesian Foundation Network")
    with col_p4:
        st.metric("4. Adaptive Ensemble", f"{ensemble_prob:.1%}", delta=f"{ensemble_prob - 0.50:+.1%}", help="Uncertainty-Weighted Meta-Learner")

    # Clinical Guidance Box
    st.markdown(f"""
    <div class="clinical-box">
        <h4 style="margin: 0; color: #0369A1;">🩺 Clinical Action Recommendations</h4>
        <p style="margin-top: 0.5rem; font-weight: 500;">{assessment['primary_recommendation']}</p>
        <ul>
            {''.join(f'<li>{item}</li>' for item in assessment['clinical_action_items'])}
        </ul>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: EXPLAINABLE AI (SHAP & LIME)
# -------------------------------------------------------------
with tab2:
    st.subheader("🔍 Patient-Specific Feature Attribution (SHAP Waterfall Analysis)")
    st.caption("How each biometric measurement pushed this patient's risk higher (Red) or lower (Green) from the population baseline:")

    fig_waterfall, _ = xai_engine.generate_shap_waterfall_plot(patient_scaled)
    st.pyplot(fig_waterfall, use_container_width=True)

    col_x1, col_x2 = st.columns(2)
    with col_x1:
        st.subheader("Detailed Factor Breakdown")
        df_impacts = pd.DataFrame(explanation["feature_impacts"])[["feature", "impact", "shap_value"]]
        df_impacts["shap_value"] = df_impacts["shap_value"].apply(lambda v: f"{v*100:+.2f}%")
        st.dataframe(df_impacts, use_container_width=True, hide_index=True)

    with col_x2:
        st.subheader("Targeted Clinical Biomarker Insights")
        if assessment["targeted_biomarker_insights"]:
            for insight in assessment["targeted_biomarker_insights"]:
                st.info(f"📌 {insight}")
        else:
            st.success("No abnormal elevated risk biomarkers detected for this patient.")


# -------------------------------------------------------------
# TAB 3: WHAT-IF CLINICAL SIMULATOR
# -------------------------------------------------------------
with tab3:
    st.subheader("🔄 Counterfactual Lifestyle Intervention Sandbox")
    st.caption("Simulate how target lifestyle or pharmaceutical improvements would reduce this patient's disease risk:")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        target_glucose = st.slider("Target Fasting Glucose (mg/dL)", 70.0, 250.0, float(patient_dict["Glucose"]), step=2.0)
        target_bmi = st.slider("Target BMI (kg/m²)", 18.0, 50.0, float(patient_dict["BMI"]), step=0.5)
        target_bp = st.slider("Target Diastolic BP (mmHg)", 50.0, 120.0, float(patient_dict["BloodPressure"]), step=1.0)

    sim_result = cf_simulator.simulate_what_if(
        patient_df,
        {"Glucose": target_glucose, "BMI": target_bmi, "BloodPressure": target_bp}
    )

    with col_sim2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Intervention Impact Summary</h3>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">Baseline Risk: <b>{sim_result['baseline_risk']:.1%}</b></p>
            <p style="font-size: 1.5rem; font-weight: 700; color: #10B981; margin: 0.5rem 0;">
                Simulated New Risk: {sim_result['simulated_risk']:.1%}
            </p>
            <p style="font-size: 1rem; color: #0284C7; font-weight: 600;">
                Risk Reduction: {sim_result['absolute_reduction']*100:.1f} percentage points ({sim_result['relative_reduction_pct']:.1f}% relative drop)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📈 Step-Wise Path to Low-Risk Tier (<30%)")
    traj = cf_simulator.compute_safe_target_trajectory(patient_df)
    st.dataframe(pd.DataFrame(traj), use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# TAB 4: POPULATION MODEL VALIDATION
# -------------------------------------------------------------
with tab4:
    st.subheader("📊 Rigorous Clinical Performance Benchmarks (Held-Out Test Cohort N=247)")
    st.caption("Stratified leak-free evaluation across KCCAM-DNN, FT-Transformer, TabPFN, and the Adaptive Ensemble:")

    if metrics_df is not None:
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, color="#D1FAE5").format("{:.4f}"),
            use_container_width=True
        )

    st.markdown("### System Architecture Blueprint")
    st.info("""
    **Pipeline Structure**:
    1. **Data Integration**: Ingests and harmonizes Local Clinical Records (Systolic/Diastolic BP, mmol/L units) with PIMA Indian Dataset.
    2. **Clinical Preprocessing**: Zero-to-NaN biological filter, iterative multivariate imputation, IQR clipping, and Kendall feature ranking.
    3. **Parallel Learning**:
       - **KCCAM-DNN**: Deep Neural Network guided by Kendall's correlation coefficient attention mechanism.
       - **FT-Transformer**: Numerical feature tokenizer with multi-head self-attention.
       - **TabPFN**: Tabular Prior-Data Fitted Bayesian Foundation Network.
    4. **Adaptive Ensembling**: Uncertainty-aware meta-learner combining probabilities dynamically.
    5. **XAI Layer**: SHAP waterfall explanations & LIME local rules.
    """)


# -------------------------------------------------------------
# TAB 5: EHR CLINICAL SUMMARY NOTE
# -------------------------------------------------------------
with tab5:
    st.subheader("📄 Electronic Health Record (EHR) Consultation Note")
    st.caption("Formatted summary note ready for export and copy-pasting into clinical records:")

    summary_text = f"""================================================================================
CLINICAL DECISION SUPPORT SYSTEM (CDSS) CONSULTATION REPORT
Date / Time: Current Session
Assisted by: Explainable AI Ensemble (KCCAM-DNN + FT-Transformer + TabPFN + SHAP)
================================================================================

PATIENT BIOMETRIC PROFILE:
  - Age                     : {int(patient_dict['Age'])} years
  - Pregnancies             : {int(patient_dict['Pregnancies'])}
  - Fasting Plasma Glucose  : {patient_dict['Glucose']:.1f} mg/dL
  - Blood Pressure (DBP/MAP): {patient_dict['BloodPressure']:.1f} / {patient_dict['MeanArterialPressure']:.1f} mmHg
  - Body Mass Index (BMI)   : {patient_dict['BMI']:.1f} kg/m²
  - Serum Insulin           : {patient_dict['Insulin']:.1f} μU/mL
  - Diabetes Pedigree Score : {patient_dict['DiabetesPedigreeFunction']:.2f}

DIAGNOSTIC TRIAGE ASSESSMENT:
  - Calibrated Risk Score   : {assessment['risk_score_pct']}%
  - Diagnostic Category     : {assessment['severity']}
  - Clinical Urgency        : {assessment['urgency']}

MODEL CONSENSUS BREAKDOWN:
  - KCCAM-DNN               : {p_kccam:.1%}
  - FT-Transformer          : {p_ft:.1%}
  - TabPFN                  : {p_tabpfn:.1%}
  - Adaptive Ensemble Final : {ensemble_prob:.1%}

PRIMARY BIOMARKER DRIVERS (SHAP EXPLAINABILITY):
"""
    for item in explanation["feature_impacts"][:4]:
        summary_text += f"  * {item['feature']:<25}: {item['shap_value']*100:+.1f}% impact ({item['impact']})\n"

    summary_text += f"""
CLINICAL ACTION PLAN & RECOMMENDATIONS:
  {assessment['primary_recommendation']}

Action Items:
"""
    for act in assessment["clinical_action_items"]:
        summary_text += f"  [ ] {act}\n"

    summary_text += "\n================================================================================"

    st.text_area("Clinical Report Text", summary_text, height=350)
    st.download_button(
        "Download Clinical EHR Summary (.txt)",
        summary_text,
        file_name=f"CDSS_Patient_Report_Risk_{assessment['risk_score_pct']}pct.txt",
        mime="text/plain",
        use_container_width=True
    )
