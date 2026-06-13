#!/usr/bin/env python3
"""
Unit tests for checking backend services and pipelines in the Streamlit App.
Verifies repo analysis, risk predictions, confidence gates, and forecasts.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import BASE_DIR

from app.services.repository_service import RepositoryService
from app.services.prediction_service import PredictionService
from app.services.explainability_service import ExplainabilityService
from app.services.forecasting_service import ForecastingService

class TestStreamlitAppBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_service = RepositoryService()
        self.pred_service = PredictionService()
        self.exp_service = ExplainabilityService()
        self.fore_service = ForecastingService()

    def test_repository_service(self) -> None:
        """Test listing, loading summary, and parsing quality metrics."""
        repos = self.repo_service.list_repositories()
        self.assertGreater(len(repos), 0, "No repositories listed by service.")
        
        # Test loading metrics for first repo
        first_repo = repos[0]
        df_metrics = self.repo_service.load_metrics(first_repo)
        self.assertFalse(df_metrics.empty, f"Quality metrics empty for {first_repo}")
        self.assertIn("file_path", df_metrics.columns)
        
        # Test summary
        summary = self.repo_service.load_summary(first_repo)
        self.assertEqual(summary["repository_name"], first_repo)
        self.assertGreater(summary["loc"], 0)

    def test_prediction_service(self) -> None:
        """Test loading Random Forest classifier and evaluating Trust Gating."""
        self.assertTrue(self.pred_service.is_ready(), "Model or Preprocessor not ready.")
        
        # Mock metrics
        df_mock = pd.DataFrame({
            "file_path": ["src/main.py", "tests/test_main.py"],
            "language": ["python", "python"],
            "loc": [100, 20],
            "complexity": [8, 2],
            "maintainability_index": [65.4, 95.0],
            "commit_count": [12, 2],
            "modification_count": [20, 3],
            "contributor_count": [3, 1],
            "commit_frequency": [0.1, 0.02],
            "repository_age_days": [120, 120]
        })
        
        df_preds = self.pred_service.predict(df_mock)
        self.assertIn("predicted_risk", df_preds.columns)
        self.assertIn("confidence", df_preds.columns)
        self.assertIn("trust_rating", df_preds.columns)
        
        # Verify Trust Gate values
        self.assertEqual(len(df_preds), 2)
        
        # Test specific Trust Gates
        self.assertEqual(self.pred_service.evaluate_trust_gate(95.0)[0], "High Confidence")
        self.assertEqual(self.pred_service.evaluate_trust_gate(75.0)[0], "Moderate Confidence")
        self.assertEqual(self.pred_service.evaluate_trust_gate(55.0)[0], "Manual Review Recommended")

    def test_explainability_service(self) -> None:
        """Test loading feature rankings and markdown reports."""
        df_imp = self.exp_service.get_feature_importances()
        self.assertFalse(df_imp.empty, "Feature importances DataFrame is empty.")
        self.assertIn("feature_name", df_imp.columns)
        
        # Test markdown load
        report = self.exp_service.get_explainability_report("error_analysis")
        self.assertFalse("not found" in report.lower(), "Error analysis markdown report missing.")

    def test_forecasting_service(self) -> None:
        """Test loading datasets and running multi-horizon risk predictions."""
        df_all = self.fore_service.get_forecast_dataset()
        self.assertFalse(df_all.empty, "Forecasting dataset is empty.")
        
        # Test forecasting trajectory
        df_forecast = self.fore_service.get_forecasts("axios", target="future_risk", horizon=30)
        self.assertFalse(df_forecast.empty, "Forecast trajectory was empty for axios.")
        self.assertIn("actual", df_forecast.columns)
        self.assertIn("xgboost", df_forecast.columns)
        self.assertIn("random_forest", df_forecast.columns)
        self.assertIn("persistence", df_forecast.columns)

if __name__ == "__main__":
    unittest.main()
