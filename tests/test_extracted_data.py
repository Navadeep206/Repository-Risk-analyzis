import os
import unittest
import pandas as pd
import sys

# Add src to search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import RAW_DIR

class TestExtractedData(unittest.TestCase):
    def setUp(self) -> None:
        self.raw_dir = RAW_DIR
        self.repo_name = "JOBPORTAL"
        self.stats_file = os.path.join(self.raw_dir, f"{self.repo_name}_repository_stats.csv")
        self.commits_file = os.path.join(self.raw_dir, f"{self.repo_name}_commits.csv")
        self.contributors_file = os.path.join(self.raw_dir, f"{self.repo_name}_contributors.csv")
        self.modifications_file = os.path.join(self.raw_dir, f"{self.repo_name}_modifications.csv")

    def test_csv_files_exist(self) -> None:
        """Verify that all target CSV files exist in data/raw/"""
        self.assertTrue(os.path.exists(self.stats_file), f"Stats file {self.stats_file} does not exist.")
        self.assertTrue(os.path.exists(self.commits_file), f"Commits file {self.commits_file} does not exist.")
        self.assertTrue(os.path.exists(self.contributors_file), f"Contributors file {self.contributors_file} does not exist.")
        self.assertTrue(os.path.exists(self.modifications_file), f"Modifications file {self.modifications_file} does not exist.")

    def test_csv_files_not_empty(self) -> None:
        """Verify that all target CSV files are not empty (have rows besides headers)"""
        for file_path in [self.stats_file, self.commits_file, self.contributors_file, self.modifications_file]:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                self.assertGreater(len(df), 0, f"CSV file {file_path} is empty.")

    def test_commit_hashes_unique(self) -> None:
        """Verify that commit hashes in JOBPORTAL_commits.csv are unique"""
        if os.path.exists(self.commits_file):
            df = pd.read_csv(self.commits_file)
            self.assertIn("commit_hash", df.columns)
            hashes = df["commit_hash"]
            self.assertEqual(len(hashes), len(hashes.unique()), "Commit hashes are not unique.")

    def test_modification_file_paths_not_null(self) -> None:
        """Verify that modification file paths in JOBPORTAL_modifications.csv are not null"""
        if os.path.exists(self.modifications_file):
            df = pd.read_csv(self.modifications_file)
            # Ensure new_path and old_path are not both null
            both_null = df["new_path"].isna() & df["old_path"].isna()
            self.assertFalse(both_null.any(), "Found modifications where both old_path and new_path are null.")

if __name__ == "__main__":
    unittest.main()
