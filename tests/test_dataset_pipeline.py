import os
import unittest
import pandas as pd
import sys

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import PROCESSED_DIR, FINAL_DIR

class TestDatasetPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.processed_dir = PROCESSED_DIR
        self.final_dir = FINAL_DIR
        
        self.merged_file = os.path.join(self.processed_dir, "merged_dataset.csv")
        self.engineered_file = os.path.join(self.processed_dir, "engineered_dataset.csv")
        self.labeled_file = os.path.join(self.processed_dir, "labeled_dataset.csv")
        self.clean_file = os.path.join(self.processed_dir, "clean_dataset.csv")
        self.ml_file = os.path.join(self.final_dir, "ml_dataset_v2.csv")
        self.train_file = os.path.join(self.final_dir, "train_v2.csv")
        self.val_file = os.path.join(self.final_dir, "validation_v2.csv")
        self.test_file = os.path.join(self.final_dir, "test_v2.csv")
        self.repos_metadata_file = os.path.join(self.final_dir, "..", "repositories_metadata.csv")

    def test_datasets_exist(self) -> None:
        """Verify that all hardened pipeline outputs exist."""
        self.assertTrue(os.path.exists(self.merged_file), "merged_dataset.csv does not exist.")
        self.assertTrue(os.path.exists(self.engineered_file), "engineered_dataset.csv does not exist.")
        self.assertTrue(os.path.exists(self.labeled_file), "labeled_dataset.csv does not exist.")
        self.assertTrue(os.path.exists(self.clean_file), "clean_dataset.csv does not exist.")
        self.assertTrue(os.path.exists(self.ml_file), "ml_dataset_v2.csv does not exist.")
        self.assertTrue(os.path.exists(self.train_file), "train_v2.csv does not exist.")
        self.assertTrue(os.path.exists(self.val_file), "validation_v2.csv does not exist.")
        self.assertTrue(os.path.exists(self.test_file), "test_v2.csv does not exist.")
        self.assertTrue(os.path.exists(self.repos_metadata_file), "repositories_metadata.csv does not exist.")

    def test_datasets_not_empty(self) -> None:
        """Verify that datasets are not empty."""
        for file_path in [self.merged_file, self.engineered_file, self.labeled_file, self.clean_file, self.ml_file, self.train_file, self.val_file, self.test_file]:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.assertGreater(len(df), 0, f"Dataset {file_path} is empty.")

    def test_dataset_size_requirement(self) -> None:
        """Verify that the final master dataset has at least 500 samples (files)."""
        if os.path.exists(self.ml_file):
            df = pd.read_csv(self.ml_file)
            self.assertGreaterEqual(len(df), 500, f"Dataset size is {len(df)}, which is less than the required 500 samples.")

    def test_multi_repository_presence(self) -> None:
        """Verify that files from multiple repositories are analyzed."""
        if os.path.exists(self.ml_file):
            df = pd.read_csv(self.ml_file)
            unique_repos = df["repository_name"].nunique()
            self.assertGreaterEqual(unique_repos, 3, f"Only found {unique_repos} repositories. Multi-repo mining failed.")

    def test_target_leakage_removed(self) -> None:
        """Verify that direct formula-based target leakage columns are removed."""
        if os.path.exists(self.ml_file):
            df = pd.read_csv(self.ml_file)
            # Old columns should not be in the master dataset
            self.assertNotIn("risk_score", df.columns, "risk_score column causes target leakage and should not be present.")
            self.assertNotIn("risk_label", df.columns, "risk_label column causes target leakage and should not be present.")
            
            # New columns must be present
            self.assertIn("bug_fix_commit_count", df.columns, "bug_fix_commit_count should be present as the bug-proneness proxy.")
            self.assertIn("historical_risk_label", df.columns, "historical_risk_label should be present.")

    def test_label_presence_across_splits(self) -> None:
        """Verify that LOW, MEDIUM, and HIGH labels are present in all splits."""
        for split_file in [self.train_file, self.val_file, self.test_file]:
            if os.path.exists(split_file):
                df = pd.read_csv(split_file)
                unique_labels = set(df["historical_risk_label"].unique())
                self.assertIn("LOW", unique_labels, f"LOW label missing from split {split_file}")
                self.assertIn("MEDIUM", unique_labels, f"MEDIUM label missing from split {split_file}")
                self.assertIn("HIGH", unique_labels, f"HIGH label missing from split {split_file}")

    def test_repository_split_safety(self) -> None:
        """Verify that repositories do not leak across splits (completely disjoint)."""
        if os.path.exists(self.train_file) and os.path.exists(self.val_file) and os.path.exists(self.test_file):
            df_train = pd.read_csv(self.train_file)
            df_val = pd.read_csv(self.val_file)
            df_test = pd.read_csv(self.test_file)
            
            repos_train = set(df_train["repository_name"].unique())
            repos_val = set(df_val["repository_name"].unique())
            repos_test = set(df_test["repository_name"].unique())
            
            # Intersection between any two splits should be empty
            self.assertEqual(len(repos_train.intersection(repos_val)), 0, "Repository leakage detected between Train and Validation splits.")
            self.assertEqual(len(repos_train.intersection(repos_test)), 0, "Repository leakage detected between Train and Test splits.")
            self.assertEqual(len(repos_val.intersection(repos_test)), 0, "Repository leakage detected between Validation and Test splits.")

if __name__ == "__main__":
    unittest.main()
