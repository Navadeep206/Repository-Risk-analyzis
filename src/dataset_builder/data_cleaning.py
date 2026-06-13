#!/usr/bin/env python3
"""
Data cleaning stage for Phase 3.5.
Handles missing values, removes duplicates, validates numeric ranges, and normalizes column headers.
"""

import os
import pandas as pd
import sys
from typing import Optional

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROCESSED_DIR, ensure_dirs_exist

def clean_data(output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Cleans the labeled dataset: handles missing values, removes duplicates,
    validates numeric boundaries, and normalizes column names.
    
    Args:
        output_file: Optional target file path for CSV.
        
    Returns:
        A pandas DataFrame of cleaned metrics.
    """
    labeled_file = os.path.join(PROCESSED_DIR, "labeled_dataset.csv")
    if not os.path.exists(labeled_file):
        raise FileNotFoundError(f"Labeled dataset file does not exist: {labeled_file}")
        
    df = pd.read_csv(labeled_file)
    
    # 1. Normalize column names (strip whitespace and lowercase)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # 2. Remove duplicate rows (based on combination of repository_name and file_path)
    initial_len = len(df)
    df = df.drop_duplicates(subset=["repository_name", "file_path"]).reset_index(drop=True)
    duplicate_count = initial_len - len(df)
    if duplicate_count > 0:
        print(f"[*] Removed {duplicate_count} duplicate rows.")
        
    # 3. Handle missing values
    # Fill missing values in maintainability_index with -1.0 (placeholder for JS/TS)
    if "maintainability_index" in df.columns:
        df["maintainability_index"] = df["maintainability_index"].fillna(-1.0)
        
    # Fill other numeric columns with 0
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if col != "maintainability_index":
            df[col] = df[col].fillna(0)
            
    # Fill categorical missing values
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("unknown")
        
    # 4. Validate numeric columns (ensure loc >= 0, complexity >= 0, etc.)
    clip_cols = [
        "loc", "complexity", "commit_count", "modification_count",
        "contributor_count", "bug_fix_commit_count", "repository_age_days"
    ]
    for col in clip_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            
    # Clip average_complexity to be non-negative
    if "average_complexity" in df.columns:
        df["average_complexity"] = df["average_complexity"].clip(lower=0.0)
        
    if not output_file:
        output_file = os.path.join(PROCESSED_DIR, "clean_dataset.csv")
        
    ensure_dirs_exist()
    df.to_csv(output_file, index=False)
    print(f"[+] Cleaned dataset saved to {output_file}. Total rows: {len(df)}")
    return df

if __name__ == "__main__":
    clean_data()
