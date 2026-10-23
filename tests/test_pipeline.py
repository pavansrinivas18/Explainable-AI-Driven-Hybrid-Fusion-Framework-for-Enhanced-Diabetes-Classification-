"""
Automated Pipeline Tests.
Validates data integration, preprocessing sanity, model inference,
ensemble calibration, XAI explanation structure, and counterfactual simulation.
"""

import unittest
import numpy as np
import pandas as pd
import torch

from src.config import ALL_FEATURES
from src.data_integration import integrate_datasets
from src.preprocessor import ClinicalPreprocessor
from src.models.kccam_dnn import KCCAM_DNN_Classifier
from src.models.ft_transformer import FT_Transformer_Classifier
from src.models.tabpfn_model import TabPFNModelWrapper
from src.models.adaptive_ensemble import AdaptiveEnsemble
from src.counterfactual import CounterfactualSimulator


class TestCDSSPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Generate small synthetic test cohort conforming to schema
        cls.df = integrate_datasets(save_output=False)
        cls.X = cls.df[ALL_FEATURES].head(30)
        cls.y = cls.df["Outcome"].head(30)

        cls.preprocessor = ClinicalPreprocessor(random_state=42)
        cls.X_scaled = cls.preprocessor.fit_transform_train(cls.X, cls.y)
        cls.kendall_tau = cls.preprocessor.get_kendall_vector()

    def test_data_integration(self):
        self.assertGreater(len(self.df), 1000)
        self.assertIn("Outcome", self.df.columns)
        self.assertIn("MeanArterialPressure", self.df.columns)
        self.assertIn("Glucose_Insulin_Ratio", self.df.columns)
        self.assertIn("CardioMetabolic_Index", self.df.columns)

    def test_preprocessor_no_nans(self):
        self.assertFalse(self.X_scaled.isna().any().any())
        self.assertEqual(len(self.kendall_tau), len(ALL_FEATURES))
        self.assertTrue((self.kendall_tau >= 0).all())

    def test_kccam_dnn_inference(self):
        kccam = KCCAM_DNN_Classifier(
            in_features=len(ALL_FEATURES),
            kendall_tau=self.kendall_tau,
            epochs=5,
            batch_size=16
        )
        kccam.fit(self.X_scaled, self.y)
        probs = kccam.predict_proba(self.X_scaled)
        self.assertEqual(probs.shape, (len(self.X_scaled), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        # Test attention extraction
        weights = kccam.get_attention_weights(self.X_scaled)
        self.assertEqual(weights.shape, (len(self.X_scaled), len(ALL_FEATURES)))

    def test_ft_transformer_inference(self):
        ft = FT_Transformer_Classifier(
            n_features=len(ALL_FEATURES),
            d_token=32,
            n_blocks=1,
            epochs=5,
            batch_size=16
        )
        ft.fit(self.X_scaled, self.y)
        probs = ft.predict_proba(self.X_scaled)
        self.assertEqual(probs.shape, (len(self.X_scaled), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_tabpfn_inference(self):
        tabpfn = TabPFNModelWrapper(N_ensemble_configurations=1)
        tabpfn.fit(self.X_scaled, self.y)
        probs = tabpfn.predict_proba(self.X_scaled)
        self.assertEqual(probs.shape, (len(self.X_scaled), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        unc = tabpfn.get_uncertainty(self.X_scaled)
        self.assertEqual(len(unc), len(self.X_scaled))

    def test_adaptive_ensemble(self):
        # Create minimal dummy models
        kccam = KCCAM_DNN_Classifier(in_features=len(ALL_FEATURES), kendall_tau=self.kendall_tau, epochs=2)
        kccam.fit(self.X_scaled, self.y)
        ft = FT_Transformer_Classifier(n_features=len(ALL_FEATURES), epochs=2)
        ft.fit(self.X_scaled, self.y)
        tabpfn = TabPFNModelWrapper(N_ensemble_configurations=1)
        tabpfn.fit(self.X_scaled, self.y)

        ensemble = AdaptiveEnsemble(kccam, ft, tabpfn)
        ensemble.fit_meta_learner(self.X_scaled, self.y)
        probs = ensemble.predict_proba(self.X_scaled)
        self.assertEqual(probs.shape, (len(self.X_scaled), 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_counterfactual_simulation(self):
        kccam = KCCAM_DNN_Classifier(in_features=len(ALL_FEATURES), kendall_tau=self.kendall_tau, epochs=2)
        kccam.fit(self.X_scaled, self.y)
        ft = FT_Transformer_Classifier(n_features=len(ALL_FEATURES), epochs=2)
        ft.fit(self.X_scaled, self.y)
        tabpfn = TabPFNModelWrapper(N_ensemble_configurations=1)
        tabpfn.fit(self.X_scaled, self.y)

        ensemble = AdaptiveEnsemble(kccam, ft, tabpfn)
        cf = CounterfactualSimulator(ensemble, self.preprocessor)

        sample_patient = self.X.head(1)
        result = cf.simulate_what_if(sample_patient, {"Glucose": 90.0, "BMI": 22.0})
        self.assertIn("baseline_risk", result)
        self.assertIn("simulated_risk", result)
        self.assertIn("absolute_reduction", result)


if __name__ == "__main__":
    unittest.main()
