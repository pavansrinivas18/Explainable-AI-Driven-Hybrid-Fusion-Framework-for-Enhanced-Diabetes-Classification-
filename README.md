# Explainable AI (XAI) Clinical Decision Support System (CDSS) for Disease Diagnosis

[![Python 3.13+](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47-brightgreen.svg)](https://streamlit.io/)
[![License: Academic Major Project](https://img.shields.io/badge/License-Academic%20Project-purple.svg)]()

An Explainable Artificial Intelligence (XAI)-based Clinical Decision Support System designed for accurate, transparent, and clinician-verifiable disease diagnosis (Diabetes prediction). The system synergizes three diverse learning paradigms (**KCCAM-DNN**, **FT-Transformer**, and **TabPFN**) under an **Adaptive Ensemble Meta-Learner**, accompanied by dual-layer **SHAP & LIME** interpretability, counterfactual **What-If** clinical simulations, and an interactive clinician dashboard.

---

## Architecture Blueprint

```
PIMA Indian Dataset ──┐
                      ├─► Data Integration ─► Preprocessing ─┬─► KCCAM-DNN      ──┐
Local Clinical Cohort ┘   (Harmonize & MAP)   (Impute & Tau) ├─► FT-Transformer ──┼─► Adaptive Ensemble ─► SHAP & LIME ─► CDSS UI
                                                             └─► TabPFN         ──┘    (Meta-Learner)
```

### Core Architecture Components:

1. **Dual-Cohort Data Integration**:
   - **Local Dataset**: Incorporates Systolic & Diastolic BP, Glucose Consumption (harmonized from mmol/L to mg/dL), BMI, and Genetic scores.
   - **PIMA Indian Dataset**: Gold-standard international benchmark.
   - **Clinical Feature Engineering**: Mean Arterial Pressure (MAP), Glucose-to-Insulin ratio, and CardioMetabolic Index.

2. **Data Preprocessing & Biological Sanity Filter**:
   - Detects biological impossibilities (e.g., $0$ Glucose, $0$ BP, $0$ BMI) and converts them to true clinical missingness.
   - Leak-free iterative multivariate imputation and IQR clipping fit strictly on the training partition.
   - Monotonic association analysis using **Kendall's rank correlation coefficient ($\tau$)**.

3. **Parallel Model Learning**:
   - **KCCAM-DNN**: **K**endall's **C**orrelation **C**oefficient **A**ttention **M**echanism **D**eep **N**eural **N**etwork. Employs Kendall's $\tau$ vector as an inductive prior to dynamically gate feature attention before deep residual layers with LayerNorm and GELU.
     $$\mathbf{s} = W_{\text{attn}} \mathbf{x} + \mathbf{b}_{\text{attn}} + \lambda \cdot |\mathbf{\tau}|, \quad \mathbf{\alpha} = \text{Softmax}(\mathbf{s}), \quad \mathbf{z} = \mathbf{\alpha} \odot \mathbf{x}$$
   - **FT-Transformer**: **F**eature **T**okenizer **Transformer** for tabular data (Gorishniy et al., NeurIPS). Projects continuous vitals into high-dimensional embedding tokens, prepends a `[CLS]` token, and models all-to-all cross-feature clinical interactions via Multi-Head Self-Attention.
   - **TabPFN**: **Tab**ular **P**rior-Data **F**itted **N**etwork (Nature/ICLR). Pre-trained Bayesian transformer foundation model providing instant tabular posterior inference without iterative gradient descent.

4. **Adaptive Ensemble Learning**:
   - Fuses predictions from all three models using an uncertainty-aware dynamic meta-learner:
     $$P_{\text{Ensemble}}(y=1|\mathbf{x}) = \sum_{m=1}^3 \tilde{W}_m(\mathbf{x}) \cdot p_m(\mathbf{x})$$

5. **Explainable AI (SHAP & LIME)**:
   - Patient-level Waterfall plots and Force plots quantifying exact feature attributions.
   - LIME local surrogate rules indicating risk threshold boundaries.
   - Counterfactual "What-If" sandbox simulating risk reduction trajectories under lifestyle/biomarker improvements.

---

## Experimental Performance Benchmarks

Evaluated on the held-out test cohort ($N = 247$ patients) with stratified leak-free validation:

| Diagnostic Model | Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | ROC-AUC | PR-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KCCAM-DNN** | 83.00% | 83.59% | 83.59% | 82.35% | 83.59% | 0.9102 | 0.9189 | 0.1328 |
| **FT-Transformer** | 80.97% | 78.32% | **87.50%** | 73.95% | 82.66% | 0.8849 | 0.8778 | 0.1659 |
| **TabPFN** | 83.00% | **85.83%** | 80.47% | **85.71%** | 83.06% | 0.9217 | 0.9360 | **0.1143** |
| **Adaptive Ensemble** | **83.40%** | 84.25% | 83.59% | 83.19% | **0.8392** | **0.9272** | **0.9400** | **0.1143** |

> **Clinical Significance**: The Adaptive Ensemble achieves **0.9272 ROC-AUC** and **0.9400 PR-AUC** with an exceptional **0.1143 Brier score**, guaranteeing that risk scores reflect true clinical probabilities.

---

## Project Directory Structure

```
IV-I Major Project/Project/
├── data/
│   ├── local_diabetes_data.csv        # Local clinical cohort (Systolic/Diastolic BP, Glucose, etc.)
│   ├── pima_diabetes_data.csv         # PIMA benchmark dataset
│   ├── integrated_diabetes_data.csv   # Unified harmonized cohort (1,233 patients)
│   └── processed/                     # Leak-free scaled training and test partitions
├── src/
│   ├── __init__.py
│   ├── config.py                      # Central hyperparameters, seeds, and clinical reference ranges
│   ├── data_integration.py           # Unit harmonization (mmol/L -> mg/dL) & biomarker engineering
│   ├── preprocessor.py                # Zero-imputation, IQR clipping, scaling, Kendall feature ranking
│   ├── models/
│   │   ├── __init__.py
│   │   ├── kccam_dnn.py               # Kendall Correlation Attention Mechanism DNN (PyTorch)
│   │   ├── ft_transformer.py          # Feature Tokenizer Transformer for Tabular Data (PyTorch)
│   │   ├── tabpfn_model.py            # TabPFN Bayesian Foundation Model Wrapper
│   │   └── adaptive_ensemble.py       # Uncertainty-aware meta-learning ensemble
│   ├── train_and_evaluate.py          # Training, 10-fold CV, calibration, and benchmark evaluation
│   ├── xai_engine.py                  # SHAP & LIME patient-level and global explanation engines
│   ├── counterfactual.py              # What-If lifestyle & biomarker clinical simulator
│   └── cdss_recommender.py            # ADA-aligned clinical triage and recommendation rules
├── models/                            # Serialized model weights (.joblib, preprocessors)
├── reports/                           # Performance evaluation CSVs and detailed JSON metrics
├── notebooks/
│   └── clinical_cdss_research.ipynb   # Interactive exploratory research and validation notebook
├── app/
│   └── app.py                         # Clinician-Centric Interactive Streamlit Dashboard
├── tests/
│   └── test_pipeline.py               # Automated unit and integration test suite
├── requirements.txt                   # Project dependencies
└── README.md                          # Comprehensive documentation
```

---

## Quickstart & Execution

### 1. Installation & Environment
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run Data Integration & Model Training
To regenerate the unified cohort, train all three parallel models, and compute evaluation metrics:
```bash
python -m src.data_integration
python -m src.preprocessor
python -u -m src.train_and_evaluate
```

### 3. Run Automated Tests
Verify pipeline integrity and model inference:
```bash
python -m unittest tests/test_pipeline.py
```

### 4. Launch the Interactive Clinician Dashboard
Start the medical web interface:
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## Clinical Decision Support Workflow

1. **Patient Intake**: Enter vitals or select preset clinical cases (Diabetic Crisis, Healthy Control, Borderline Pre-Diabetic).
2. **Consensus Diagnosis**: Observe concurrent predictions from KCCAM-DNN, FT-Transformer, TabPFN, and the Adaptive Ensemble.
3. **Inspect SHAP Waterfall**: Review exactly how each vital shifted risk from baseline.
4. **Counterfactual Simulation**: Slide glucose and BMI targets to see required lifestyle reductions.
5. **Export Consultation Note**: One-click printable summary ready for EHR/EMR documentation.
