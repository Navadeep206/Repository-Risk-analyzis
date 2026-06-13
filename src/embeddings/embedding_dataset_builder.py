#!/usr/bin/env python3
"""
Embedding dataset builder module for Phase 5.
Merges generated code embeddings with historical defect risk labels.
Saves data/final/embedding_dataset.parquet.
"""

import os
import sys
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_DIR, FINAL_DIR, ensure_dirs_exist

EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

def build_embedding_dataset() -> None:
    """
    Loads embeddings.parquet and ml_dataset_v2.csv,
    performs an inner join on repository_name and file_path,
    and writes the final merged dataset to data/final/embedding_dataset.parquet.
    """
    ensure_dirs_exist()
    
    parquet_path = os.path.join(EMBEDDINGS_DIR, "embeddings.parquet")
    labels_csv_path = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Embeddings parquet file not found: {parquet_path}. Run storage compiler first.")
    if not os.path.exists(labels_csv_path):
        raise FileNotFoundError(f"Master label dataset not found: {labels_csv_path}. Run baseline splits first.")
        
    print("[*] Loading generated embeddings and labels...")
    df_embeddings = pd.read_parquet(parquet_path)
    df_labels = pd.read_csv(labels_csv_path)[["repository_name", "file_path", "historical_risk_label"]]
    
    print(f"[*] Total embeddings loaded: {len(df_embeddings)}")
    print(f"[*] Total labeled files loaded: {len(df_labels)}")
    
    # Merge on keys
    df_merged = pd.merge(
        df_embeddings,
        df_labels,
        on=["repository_name", "file_path"],
        how="inner"
    )
    
    # Reorder columns to place key columns first
    required_cols = [
        "repository_name",
        "file_path",
        "language",
        "historical_risk_label",
        "embedding_id",
        "embedding"
    ]
    
    # Verify cols
    for col in required_cols:
        if col not in df_merged.columns:
            raise ValueError(f"Missing expected column after merge: {col}")
            
    df_final = df_merged[required_cols].copy()
    
    output_path = os.path.join(FINAL_DIR, "embedding_dataset.parquet")
    df_final.to_parquet(output_path, index=False)
    
    print(f"[+] Successfully compiled final embedding dataset with {len(df_final)} rows.")
    print(f"[+] Saved to {output_path}")

if __name__ == "__main__":
    build_embedding_dataset()
