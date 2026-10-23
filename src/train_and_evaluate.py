"""
Training & Evaluation Pipeline.
Executes parallel model learning (KCCAM-DNN, FT-Transformer, TabPFN),
fits the Adaptive Ensemble, and evaluates performance on held-out test data
using clinical metrics: Accuracy, Precision, Recall/Sensitivity, Specificity,
F1-Score, ROC-AUC, PR-AUC, and Brier Score.
"""

import json
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.model_selection import StratifiedKFold

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    ALL_FEATURES
)
from src.preprocessor import ClinicalPreprocessor, run_preprocessing_pipeline
from src.models.kccam_dnn import KCCAM_DNN_Classifier
from src.models.ft_transformer import FT_Transformer_Classifier
from src.models.tabpfn_model import TabPFNModelWrapper
from src.models.adaptive_ensemble import AdaptiveEnsemble


def compute_clinical_metrics(y_true, y_pred, y_prob):
    """
    Computes rigorous clinical diagnostic evaluation metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall (Sensitivity)": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "Brier Score": float(brier_score_loss(y_true, y_prob)),
        "True Positives": int(tp),
        "False Negatives (Missed Cases)": int(fn),
        "True Negatives": int(tn),
        "False Positives": int(fp)
    }


def train_and_evaluate_all():
    print("=" * 70, flush=True)
    print("EXPLAINABLE AI CLINICAL DECISION SUPPORT SYSTEM: TRAINING PIPELINE", flush=True)
    print("=" * 70, flush=True)

    # 1. Load or run preprocessing pipeline
    try:
        preprocessor = ClinicalPreprocessor.load(MODELS_DIR / "clinical_preprocessor.joblib")
        X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_processed.csv")
        y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train_processed.csv").values.ravel()
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_processed.csv")
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test_processed.csv").values.ravel()
    except Exception:
        print("[TRAINING] Running fresh leak-free preprocessing...", flush=True)
        X_train, X_test, y_train, y_test, preprocessor = run_preprocessing_pipeline()

    n_features = X_train.shape[1]
    kendall_tau = preprocessor.get_kendall_vector()

    print(f"\n[TRAINING] Feature Count: {n_features}", flush=True)
    print(f"[TRAINING] Training Cohort Size: {len(X_train)} | Test Cohort Size: {len(X_test)}", flush=True)

    # Split a portion of train for meta-learner calibration
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    train_idx, val_idx = next(skf.split(X_train, y_train))
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]

    # -------------------------------------------------------------
    # Model 1: KCCAM-DNN
    # -------------------------------------------------------------
    print("\n" + "-" * 50, flush=True)
    print("1. Training KCCAM-DNN (Kendall Attention Deep Neural Network)...", flush=True)
    kccam_clf = KCCAM_DNN_Classifier(
        in_features=n_features,
        kendall_tau=kendall_tau
    )
    kccam_clf.fit(X_tr, y_tr)
    print("   [OK] KCCAM-DNN training complete.", flush=True)

    # -------------------------------------------------------------
    # Model 2: FT-Transformer
    # -------------------------------------------------------------
    print("\n" + "-" * 50, flush=True)
    print("2. Training FT-Transformer (Feature Tokenizer Transformer)...", flush=True)
    ft_clf = FT_Transformer_Classifier(
        n_features=n_features
    )
    ft_clf.fit(X_tr, y_tr)
    print("   [OK] FT-Transformer training complete.", flush=True)

    # -------------------------------------------------------------
    # Model 3: TabPFN
    # -------------------------------------------------------------
    print("\n" + "-" * 50, flush=True)
    print("3. Initializing TabPFN (Tabular Prior-Data Fitted Network)...", flush=True)
    tabpfn_clf = TabPFNModelWrapper(device="cpu", N_ensemble_configurations=2)
    tabpfn_clf.fit(X_tr, y_tr)
    print("   [OK] TabPFN initialization complete.", flush=True)

    # -------------------------------------------------------------
    # Model 4: Adaptive Ensemble Learning
    # -------------------------------------------------------------
    print("\n" + "-" * 50, flush=True)
    print("4. Fitting Adaptive Ensemble Meta-Learner...", flush=True)
    ensemble = AdaptiveEnsemble(
        kccam_model=kccam_clf,
        ft_model=ft_clf,
        tabpfn_model=tabpfn_clf,
        random_state=RANDOM_SEED
    )
    ensemble.fit_meta_learner(X_val, y_val)
    print("   [OK] Adaptive Ensemble meta-learner calibrated.", flush=True)

    # -------------------------------------------------------------
    # 5. Comprehensive Clinical Performance Evaluation
    # -------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("PERFORMANCE EVALUATION ON HELD-OUT TEST COHORT (N = %d)" % len(X_test), flush=True)
    print("=" * 70, flush=True)

    models_dict = {
        "KCCAM-DNN": kccam_clf,
        "FT-Transformer": ft_clf,
        "TabPFN": tabpfn_clf,
        "Adaptive Ensemble": ensemble
    }

    results = {}
    for name, model in models_dict.items():
        print(f"   Evaluating {name}...", flush=True)
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)
        metrics = compute_clinical_metrics(y_test, preds, probs)
        results[name] = metrics

    # Format into comparative DataFrame
    metrics_table = pd.DataFrame(results).T[
        ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC", "PR-AUC", "Brier Score"]
    ]

    print("\n" + metrics_table.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)

    # Save reports and artifacts
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(REPORTS_DIR / "model_comparison_metrics.csv")
    with open(REPORTS_DIR / "clinical_metrics_detailed.json", "w") as f:
        json.dump(results, f, indent=4)

    # Serialize champion ensemble and individual models
    joblib.dump(ensemble, MODELS_DIR / "adaptive_ensemble.joblib")
    joblib.dump(tabpfn_clf, MODELS_DIR / "tabpfn_model.joblib")
    joblib.dump(kccam_clf, MODELS_DIR / "kccam_model.joblib")
    joblib.dump(ft_clf, MODELS_DIR / "ft_transformer_model.joblib")

    print("\n[EXPORT] All models and reports saved successfully to 'models/' and 'reports/'.", flush=True)
    return results, ensemble, preprocessor


if __name__ == "__main__":
    train_and_evaluate_all()
