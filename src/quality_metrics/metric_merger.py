#!/usr/bin/env python3
"""
Metric merger tool. Combines Python, JavaScript, and TypeScript metrics into processed/quality_metrics.csv.
"""

import os
import pandas as pd
import sys
from typing import Optional

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs_exist

def merge_metrics(output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Merges python, javascript, and typescript metrics into a single processed quality_metrics.csv dataset.
    
    Args:
        output_file: Optional target file path for CSV.
        
    Returns:
        A pandas DataFrame of merged quality metrics.
    """
    python_file = os.path.join(RAW_DIR, "python_metrics.csv")
    js_file = os.path.join(RAW_DIR, "javascript_metrics.csv")
    ts_file = os.path.join(RAW_DIR, "typescript_metrics.csv")
    
    dataframes = []
    
    # Load each file if it exists and has content
    for file_path in [python_file, js_file, ts_file]:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if not df.empty:
                    dataframes.append(df)
            except Exception as e:
                print(f"[-] Warning: Failed to load {file_path}: {e}")
                
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
    else:
        # Return empty df with target schema
        merged_df = pd.DataFrame(columns=[
            "repository_name", "file_path", "language", "loc", "complexity",
            "warnings", "errors", "maintainability_index"
        ])
        
    if not output_file:
        output_file = os.path.join(PROCESSED_DIR, "quality_metrics.csv")
        
    ensure_dirs_exist()
    merged_df.to_csv(output_file, index=False)
    print(f"[+] Merged quality metrics saved to {output_file}. Total rows: {len(merged_df)}.")
    return merged_df

if __name__ == "__main__":
    merge_metrics()
