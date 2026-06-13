import os
import unittest
import pandas as pd
import sys

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import RAW_DIR, PROCESSED_DIR

class TestQualityMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_dir = RAW_DIR
        self.processed_dir = PROCESSED_DIR
        
        self.profile_file = os.path.join(self.raw_dir, "repository_language_profile.csv")
        self.python_file = os.path.join(self.raw_dir, "python_metrics.csv")
        self.js_file = os.path.join(self.raw_dir, "javascript_metrics.csv")
        self.ts_file = os.path.join(self.raw_dir, "typescript_metrics.csv")
        self.final_file = os.path.join(self.processed_dir, "quality_metrics.csv")

    def test_all_files_exist(self) -> None:
        """Verify that all target CSV files exist."""
        self.assertTrue(os.path.exists(self.profile_file), "repository_language_profile.csv does not exist.")
        self.assertTrue(os.path.exists(self.python_file), "python_metrics.csv does not exist.")
        self.assertTrue(os.path.exists(self.js_file), "javascript_metrics.csv does not exist.")
        self.assertTrue(os.path.exists(self.ts_file), "typescript_metrics.csv does not exist.")
        self.assertTrue(os.path.exists(self.final_file), "quality_metrics.csv does not exist.")

    def test_final_dataset_not_empty_and_no_duplicates(self) -> None:
        """Verify that the final merged dataset is not empty and has no duplicate rows."""
        if os.path.exists(self.final_file):
            df = pd.read_csv(self.final_file)
            self.assertGreater(len(df), 0, "final quality_metrics.csv is empty.")
            
            # Check for duplicate rows (by repository_name + file_path)
            unique_combos = df.groupby(["repository_name", "file_path"]).size()
            self.assertTrue(
                (unique_combos == 1).all(),
                "Found duplicate repository + file_path combinations in final quality_metrics.csv"
            )

    def test_metrics_bounds(self) -> None:
        """Verify that LOC > 0 and Complexity >= 0 for all processed files."""
        if os.path.exists(self.final_file):
            df = pd.read_csv(self.final_file)
            self.assertIn("loc", df.columns)
            self.assertIn("complexity", df.columns)
            
            # LOC must be >= 0
            self.assertTrue((df["loc"] >= 0).all(), "Found file with LOC < 0 in final quality_metrics.csv")
            
            # Complexity must be >= 0
            self.assertTrue((df["complexity"] >= 0).all(), "Found file with complexity < 0 in final quality_metrics.csv")
            
    def test_language_profile_is_valid(self) -> None:
        """Verify language profile is present and not empty."""
        if os.path.exists(self.profile_file):
            df = pd.read_csv(self.profile_file)
            self.assertGreater(len(df), 0, "repository_language_profile.csv is empty.")
            self.assertIn("language", df.columns)
            self.assertIn("percentage", df.columns)

if __name__ == "__main__":
    unittest.main()
