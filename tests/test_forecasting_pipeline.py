#!/usr/bin/env python3
"""
Test suite for Phase 9 Forecasting Pipeline.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add src to search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import BASE_DIR, PROCESSED_DIR, FINAL_DIR

from forecasting.temporal_dataset_builder import build_daily_logs
from forecasting.feature_windowing import build_forecasting_dataset
from forecasting.baseline_forecaster import PersistenceForecaster
from forecasting.random_forest_forecaster import RandomForestForecaster
from forecasting.xgboost_forecaster import XGBoostForecaster
from forecasting.evaluator import evaluate_predictions, calculate_mape

class TestForecastingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.daily_file = os.path.join(PROCESSED_DIR, "..", "intermediate", "daily_repo_logs.csv")
        self.dataset_file = os.path.join(FINAL_DIR, "forecasting_dataset.csv")
        self.models_dir = os.path.join(BASE_DIR, "models")
        self.reports_dir = os.path.join(BASE_DIR, "reports", "forecasting")

    def test_a_daily_log_builder(self) -> None:
        """Verify daily log builder runs and outputs valid log CSV."""
        df = build_daily_logs()
        self.assertFalse(df.empty, "Daily logs DataFrame should not be empty.")
        self.assertTrue(os.path.exists(self.daily_file), "daily_repo_logs.csv was not created.")
        
        # Check columns
        expected_cols = [
            "repository_name", "date", "commits_count", "bug_fixes_count", 
            "contributor_emails", "modifications_count", "complexity_sum", 
            "complexity_count", "maintainability_sum", "maintainability_count"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Daily logs missing column {col}")

    def test_b_feature_windowing(self) -> None:
        """Verify feature windowing generates correct rolling windows and targets."""
        df = build_forecasting_dataset()
        self.assertFalse(df.empty, "Forecasting dataset should not be empty.")
        self.assertTrue(os.path.exists(self.dataset_file), "forecasting_dataset.csv was not created.")
        
        # Check features
        expected_feat = [
            "commit_frequency_30d", "modification_count_60d", "avg_complexity_90d",
            "active_contributors_30d", "avg_maintainability_60d", "risk_score_90d"
        ]
        for feat in expected_feat:
            self.assertIn(feat, df.columns, f"Dataset missing rolling feature {feat}")
            
        # Check targets
        expected_targets = [
            "future_risk_30d", "future_risk_60d", "future_risk_90d",
            "future_defect_count_30d", "future_modification_intensity_90d"
        ]
        for target in expected_targets:
            self.assertIn(target, df.columns, f"Dataset missing target {target}")

    def test_c_persistence_forecaster(self) -> None:
        """Verify persistence forecaster returns correct counterpart values."""
        df = pd.DataFrame({
            "risk_score_30d": [1.2, 3.4, 5.6],
            "defect_count_60d": [0, 2, 5]
        })
        forecaster = PersistenceForecaster()
        
        preds_risk = forecaster.predict(df, "future_risk_30d", 30)
        np.testing.assert_array_equal(preds_risk, [1.2, 3.4, 5.6])
        
        preds_defects = forecaster.predict(df, "future_defect_count_60d", 60)
        np.testing.assert_array_equal(preds_defects, [0, 2, 5])

    def test_d_random_forest_forecaster(self) -> None:
        """Test training and prediction of Random Forest forecaster."""
        X_train = pd.DataFrame(np.random.rand(10, 5), columns=[f"feat_{i}" for i in range(5)])
        y_train = pd.Series(np.random.rand(10))
        X_test = pd.DataFrame(np.random.rand(5, 5), columns=[f"feat_{i}" for i in range(5)])
        
        rf = RandomForestForecaster(n_estimators=10, max_depth=3, random_state=42)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        
        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all(preds >= 0))

    def test_e_xgboost_forecaster(self) -> None:
        """Test training, prediction, and JSON saving/loading of XGBoost forecaster."""
        X_train = pd.DataFrame(np.random.rand(10, 5), columns=[f"feat_{i}" for i in range(5)])
        y_train = pd.Series(np.random.rand(10))
        X_test = pd.DataFrame(np.random.rand(5, 5), columns=[f"feat_{i}" for i in range(5)])
        
        xgb = XGBoostForecaster(n_estimators=10, max_depth=3, random_state=42)
        xgb.fit(X_train, y_train)
        preds_orig = xgb.predict(X_test)
        
        # Test JSON save/load
        temp_path = os.path.join(self.models_dir, "temp_test_xgb.json")
        xgb.save_model(temp_path)
        self.assertTrue(os.path.exists(temp_path))
        
        xgb_new = XGBoostForecaster()
        xgb_new.load_model(temp_path)
        preds_new = xgb_new.predict(X_test)
        
        np.testing.assert_array_almost_equal(preds_orig, preds_new)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_f_evaluator(self) -> None:
        """Test metric calculation in evaluator."""
        y_true = np.array([1.0, 2.0, 3.0, 0.0])
        y_pred = np.array([1.1, 1.9, 3.2, 0.5])
        
        mape = calculate_mape(y_true, y_pred)
        self.assertGreater(mape, 0.0)
        
        metrics = evaluate_predictions(y_true, y_pred)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("mape", metrics)
        self.assertGreater(metrics["rmse"], metrics["mae"])

if __name__ == "__main__":
    unittest.main()
