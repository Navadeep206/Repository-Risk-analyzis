import os
import unittest
import pandas as pd
import sys

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import BASE_DIR

class TestExplainabilityPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.exp_dir = os.path.join(BASE_DIR, "reports", "explainability")
        self.plots_dir = os.path.join(self.exp_dir, "plots")
        
        # Reports
        self.feat_imp = os.path.join(self.exp_dir, "feature_importance.csv")
        self.global_ranking = os.path.join(self.exp_dir, "global_feature_ranking.csv")
        self.perm_imp = os.path.join(self.exp_dir, "permutation_importance.csv")
        self.error_analysis = os.path.join(self.exp_dir, "error_analysis.md")
        self.domain_shift = os.path.join(self.exp_dir, "domain_shift_analysis.md")
        self.hybrid_failure = os.path.join(self.exp_dir, "hybrid_failure_analysis.md")
        self.trustworthiness = os.path.join(self.exp_dir, "model_trustworthiness.md")
        
        # Plots
        self.plot_feat = os.path.join(self.plots_dir, "feature_importance.png")
        self.plot_perm = os.path.join(self.plots_dir, "permutation_importance.png")
        self.plot_shift = os.path.join(self.plots_dir, "domain_shift.png")
        self.plot_cm = os.path.join(self.plots_dir, "confusion_matrix.png")
        self.plot_dist = os.path.join(self.plots_dir, "class_distribution.png")

    def test_explainability_reports_exist(self) -> None:
        """Verify that all markdown and CSV explainability reports exist and are non-empty."""
        self.assertTrue(os.path.exists(self.feat_imp), "feature_importance.csv is missing.")
        self.assertTrue(os.path.exists(self.global_ranking), "global_feature_ranking.csv is missing.")
        self.assertTrue(os.path.exists(self.perm_imp), "permutation_importance.csv is missing.")
        self.assertTrue(os.path.exists(self.error_analysis), "error_analysis.md is missing.")
        self.assertTrue(os.path.exists(self.domain_shift), "domain_shift_analysis.md is missing.")
        self.assertTrue(os.path.exists(self.hybrid_failure), "hybrid_failure_analysis.md is missing.")
        self.assertTrue(os.path.exists(self.trustworthiness), "model_trustworthiness.md is missing.")
        
        # Verify non-empty files
        self.assertGreater(os.path.getsize(self.feat_imp), 0, "feature_importance.csv is empty.")
        self.assertGreater(os.path.getsize(self.global_ranking), 0, "global_feature_ranking.csv is empty.")
        self.assertGreater(os.path.getsize(self.perm_imp), 0, "permutation_importance.csv is empty.")
        self.assertGreater(os.path.getsize(self.error_analysis), 0, "error_analysis.md is empty.")
        self.assertGreater(os.path.getsize(self.domain_shift), 0, "domain_shift_analysis.md is empty.")
        self.assertGreater(os.path.getsize(self.hybrid_failure), 0, "hybrid_failure_analysis.md is empty.")
        self.assertGreater(os.path.getsize(self.trustworthiness), 0, "model_trustworthiness.md is empty.")

    def test_explainability_plots_exist(self) -> None:
        """Verify that all explainability visual plots are correctly generated."""
        self.assertTrue(os.path.exists(self.plot_feat), "feature_importance.png is missing.")
        self.assertTrue(os.path.exists(self.plot_perm), "permutation_importance.png is missing.")
        self.assertTrue(os.path.exists(self.plot_shift), "domain_shift.png is missing.")
        self.assertTrue(os.path.exists(self.plot_cm), "confusion_matrix.png is missing.")
        self.assertTrue(os.path.exists(self.plot_dist), "class_distribution.png is missing.")

    def test_global_rankings_format(self) -> None:
        """Verify that global rankings CSV loads correctly and has expected columns."""
        df = pd.read_csv(self.global_ranking)
        expected_cols = [
            "feature_name", "rf_intrinsic_importance", 
            "rf_permutation_mean", "rf_permutation_std",
            "rank_intrinsic", "rank_permutation", "average_rank"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Column {col} is missing from global rankings.")
            
        # Verify it has exactly 11 features
        self.assertEqual(len(df), 11, "Global rankings should have exactly 11 features.")
        
        # Verify it is sorted by average_rank in ascending order
        self.assertTrue(df["average_rank"].is_monotonic_increasing, "Global rankings are not sorted by average rank.")

if __name__ == "__main__":
    unittest.main()
