#!/usr/bin/env python3
"""
Embedding storage and formatting compiler for Phase 5.
Generates data/embeddings/embedding_metadata.csv and data/embeddings/embeddings.parquet
from embeddings.npy and clean_source_dataset.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_DIR

INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

def compile_storage() -> None:
    """
    Constructs metadata CSV and combines metadata with numpy embeddings
    to export to a single Parquet file.
    """
    clean_csv_path = os.path.join(INTERMEDIATE_DIR, "clean_source_dataset.csv")
    npy_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Clean source dataset not found: {clean_csv_path}")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Embeddings array not found: {npy_path}")
        
    df_clean = pd.read_csv(clean_csv_path)
    
    # Load embeddings array using memmap mode to save memory
    embeddings = np.load(npy_path, mmap_mode="r")
    
    if len(df_clean) != len(embeddings):
        raise ValueError(
            f"Dimension mismatch: clean source dataset has {len(df_clean)} rows, "
            f"but embeddings array has {len(embeddings)} rows."
        )
        
    print("[*] Generating embedding storage formats...")
    
    # 1. Create embedding_metadata.csv
    metadata_df = pd.DataFrame({
        "embedding_id": range(len(df_clean)),
        "repository_name": df_clean["repository_name"],
        "file_path": df_clean["file_path"],
        "language": df_clean["language"]
    })
    
    metadata_csv_path = os.path.join(EMBEDDINGS_DIR, "embedding_metadata.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    print(f"[+] Saved metadata to {metadata_csv_path}")
    
    # 2. Create embeddings.parquet
    # We copy the metadata and append the embedding column as a list of floats
    parquet_df = metadata_df.copy()
    
    # Converting numpy arrays to a list of lists/arrays for pandas compatibility
    parquet_df["embedding"] = list(embeddings)
    
    parquet_path = os.path.join(EMBEDDINGS_DIR, "embeddings.parquet")
    # Using default engine (pyarrow is installed)
    parquet_df.to_parquet(parquet_path, index=False)
    print(f"[+] Saved combined embeddings to {parquet_path}")

if __name__ == "__main__":
    compile_storage()
