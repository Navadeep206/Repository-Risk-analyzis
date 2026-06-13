import os
import unittest
import pickle
import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import BASE_DIR
from ml.preprocessing import CodeRiskPreprocessor
from ml.data_loader import load_split_data

class TestMLPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.models_dir = os.path.join(BASE_DIR, "models")
        self.reports_dir = os.path.join(BASE_DIR, "reports")
        
        self.preproc_file = os.path.join(self.models_dir, "preprocessor.pkl")
        self.lr_file = os.path.join(self.models_dir, "logistic_regression.pkl")
        self.dt_file = os.path.join(self.models_dir, "decision_tree.pkl")
        self.rf_file = os.path.join(self.models_dir, "random_forest.pkl")
        self.xgb_file = os.path.join(self.models_dir, "xgboost.pkl")
        
        self.eval_report = os.path.join(self.reports_dir, "evaluation_report.md")
        self.feat_imp = os.path.join(self.reports_dir, "feature_importance.csv")
        self.model_comp = os.path.join(self.reports_dir, "model_comparison.csv")
        self.cv_results = os.path.join(self.reports_dir, "cross_validation_results.csv")

    def test_models_exist(self) -> None:
        """Verify all baseline models and the preprocessor exist on disk."""
        self.assertTrue(os.path.exists(self.preproc_file), "preprocessor.pkl does not exist.")
        self.assertTrue(os.path.exists(self.lr_file), "logistic_regression.pkl does not exist.")
        self.assertTrue(os.path.exists(self.dt_file), "decision_tree.pkl does not exist.")
        self.assertTrue(os.path.exists(self.rf_file), "random_forest.pkl does not exist.")
        self.assertTrue(os.path.exists(self.xgb_file), "xgboost.pkl does not exist.")

    def test_reports_exist(self) -> None:
        """Verify all reporting files are generated."""
        self.assertTrue(os.path.exists(self.eval_report), "evaluation_report.md does not exist.")
        self.assertTrue(os.path.exists(self.feat_imp), "feature_importance.csv does not exist.")
        self.assertTrue(os.path.exists(self.model_comp), "model_comparison.csv does not exist.")
        self.assertTrue(os.path.exists(self.cv_results), "cross_validation_results.csv does not exist.")

    def test_models_loadable(self) -> None:
        """Verify all models can be deserialized and are of correct type."""
        # Preprocessor
        preprocessor = CodeRiskPreprocessor.load(self.preproc_file)
        self.assertIsInstance(preprocessor, CodeRiskPreprocessor)
        self.assertGreater(len(preprocessor.feature_names), 0)
        
        # Logistic Regression
        with open(self.lr_file, "rb") as f:
            lr_model = pickle.load(f)
        self.assertIsInstance(lr_model, LogisticRegression)
        
        # Decision Tree
        with open(self.dt_file, "rb") as f:
            dt_model = pickle.load(f)
        self.assertIsInstance(dt_model, DecisionTreeClassifier)
        
        # Random Forest
        with open(self.rf_file, "rb") as f:
            rf_model = pickle.load(f)
        self.assertIsInstance(rf_model, RandomForestClassifier)
        
        # XGBoost
        with open(self.xgb_file, "rb") as f:
            xgb_model = pickle.load(f)
        self.assertIsInstance(xgb_model, XGBClassifier)

    def test_end_to_end_prediction(self) -> None:
        """Verify that preprocessor and all models run prediction on test data."""
        # Load sample data
        X_test, y_test = load_split_data("test_v2")
        
        # Transform using preprocessor
        preprocessor = CodeRiskPreprocessor.load(self.preproc_file)
        X_test_proc = preprocessor.transform(X_test)
        
        # Test predictions on each model
        model_paths = [self.lr_file, self.dt_file, self.rf_file, self.xgb_file]
        for path in model_paths:
            with open(path, "rb") as f:
                model = pickle.load(f)
            
            preds = model.predict(X_test_proc)
            self.assertEqual(len(preds), len(X_test), f"Model {path} prediction length mismatch.")
            # Labels should be 0, 1, or 2
            self.assertTrue(np.all((preds == 0) | (preds == 1) | (preds == 2)), f"Model {path} output contains invalid label values.")

if __name__ == "__main__":
    unittest.main()
