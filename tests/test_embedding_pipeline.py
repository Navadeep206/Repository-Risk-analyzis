import os
import unittest
import numpy as np
import pandas as pd
import sys

# Ensure src is in search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from config import BASE_DIR, FINAL_DIR

class TestEmbeddingPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.intermediate_dir = os.path.join(BASE_DIR, "data", "intermediate")
        self.embeddings_dir = os.path.join(BASE_DIR, "data", "embeddings")
        self.final_dir = FINAL_DIR
        
        self.source_code_csv = os.path.join(self.intermediate_dir, "source_code_dataset.csv")
        self.clean_code_csv = os.path.join(self.intermediate_dir, "clean_source_dataset.csv")
        
        self.embeddings_npy = os.path.join(self.embeddings_dir, "embeddings.npy")
        self.metadata_csv = os.path.join(self.embeddings_dir, "embedding_metadata.csv")
        self.embeddings_parquet = os.path.join(self.embeddings_dir, "embeddings.parquet")
        
        self.final_dataset_parquet = os.path.join(self.final_dir, "embedding_dataset.parquet")

    def test_pipeline_outputs_exist(self) -> None:
        """Verify that all Phase 5 pipeline datasets exist on disk."""
        self.assertTrue(os.path.exists(self.source_code_csv), "source_code_dataset.csv does not exist.")
        self.assertTrue(os.path.exists(self.clean_code_csv), "clean_source_dataset.csv does not exist.")
        self.assertTrue(os.path.exists(self.embeddings_npy), "embeddings.npy does not exist.")
        self.assertTrue(os.path.exists(self.metadata_csv), "embedding_metadata.csv does not exist.")
        self.assertTrue(os.path.exists(self.embeddings_parquet), "embeddings.parquet does not exist.")
        self.assertTrue(os.path.exists(self.final_dataset_parquet), "embedding_dataset.parquet does not exist.")

    def test_numpy_embeddings(self) -> None:
        """Verify embeddings.npy shapes, dimensions, and finite values."""
        if os.path.exists(self.embeddings_npy):
            embeddings = np.load(self.embeddings_npy, mmap_mode="r")
            self.assertEqual(len(embeddings.shape), 2, "Embeddings array must be 2-dimensional.")
            self.assertEqual(embeddings.shape[1], 768, "Embedding dimensions must be exactly 768.")
            self.assertGreater(embeddings.shape[0], 0, "Embeddings array cannot be empty.")
            
            # Check for NaN or Inf values
            self.assertFalse(np.isnan(embeddings).any(), "Embeddings array contains NaN values.")
            self.assertFalse(np.isinf(embeddings).any(), "Embeddings array contains infinite values.")

    def test_metadata_alignment(self) -> None:
        """Verify that metadata maps correctly and matches clean file count."""
        if os.path.exists(self.metadata_csv) and os.path.exists(self.clean_code_csv) and os.path.exists(self.embeddings_npy):
            df_meta = pd.read_csv(self.metadata_csv)
            df_clean = pd.read_csv(self.clean_code_csv)
            embeddings = np.load(self.embeddings_npy, mmap_mode="r")
            
            self.assertEqual(len(df_meta), len(df_clean), "Metadata row count must match clean files count.")
            self.assertEqual(len(df_meta), len(embeddings), "Metadata count must match numpy array length.")
            
            # Verify unique sequential ids
            ids = df_meta["embedding_id"].tolist()
            self.assertEqual(ids, list(range(len(df_meta))), "Embedding IDs must be unique and sequential starting from 0.")

    def test_final_embedding_dataset(self) -> None:
        """Verify final training dataset format and required columns."""
        if os.path.exists(self.final_dataset_parquet):
            df_final = pd.read_parquet(self.final_dataset_parquet)
            self.assertGreater(len(df_final), 0, "Final embedding dataset cannot be empty.")
            
            # Required columns checklist
            required_cols = ["repository_name", "file_path", "language", "historical_risk_label", "embedding_id", "embedding"]
            for col in required_cols:
                self.assertIn(col, df_final.columns, f"Required column '{col}' is missing from the final dataset.")
                
            # Verify non-null labels and values
            self.assertEqual(df_final["historical_risk_label"].isna().sum(), 0, "Final dataset contains null risk labels.")
            self.assertEqual(df_final["embedding_id"].isna().sum(), 0, "Final dataset contains null embedding IDs.")
            
            # Label check
            unique_labels = set(df_final["historical_risk_label"].unique())
            self.assertTrue(unique_labels.issubset({"LOW", "MEDIUM", "HIGH"}), "Final dataset contains invalid labels.")

if __name__ == "__main__":
    unittest.main()
