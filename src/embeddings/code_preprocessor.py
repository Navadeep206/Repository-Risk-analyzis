#!/usr/bin/env python3
"""
Source code preprocessing module for Phase 5.
Cleans code files by removing empty, binary, minified, or excessively large files.
"""

import os
import sys
import pandas as pd
from typing import Tuple

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_DIR

INTERMEDIATE_DIR = os.path.join(DATA_DIR, "intermediate")

def clean_source_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the source code dataset by applying quality filters.
    
    Filters:
        - Removes empty or whitespace-only files.
        - Removes minified files (ends with .min.js or has lines > 1000 characters).
        - Skips excessively large files (> 250KB or > 5000 lines) to prevent memory bloating.
        - Tracks character_count and line_count.
    """
    print("[*] Preprocessing and cleaning source files...")
    cleaned_records = []
    
    # Thresholds
    MAX_FILE_SIZE_KB = 250
    MAX_LINE_COUNT = 5000
    MAX_LINE_LEN = 1000  # Indicator of minification
    
    for idx, row in df.iterrows():
        code = str(row["source_code"]).strip()
        
        # 1. Skip empty files
        if not code:
            continue
            
        # 2. Check line counts and length
        lines = code.splitlines()
        line_count = len(lines)
        char_count = len(code)
        
        # Skip if line count exceeds threshold
        if line_count > MAX_LINE_COUNT:
            # print(f"[-] Skipping {row['repository_name']}/{row['file_path']} (too many lines: {line_count})")
            continue
            
        # Skip if total size in KB exceeds threshold
        file_size_kb = char_count / 1024
        if file_size_kb > MAX_FILE_SIZE_KB:
            # print(f"[-] Skipping {row['repository_name']}/{row['file_path']} (too large: {file_size_kb:.1f} KB)")
            continue
            
        # 3. Detect minified files
        file_path = row["file_path"].lower()
        if file_path.endswith(".min.js") or file_path.endswith(".min.ts"):
            continue
            
        # Check for very long single lines (sign of minification/bundling)
        has_too_long_line = any(len(line) > MAX_LINE_LEN for line in lines)
        if has_too_long_line:
            # print(f"[-] Skipping {row['repository_name']}/{row['file_path']} (contains excessively long lines)")
            continue
            
        # If it passes all checks, compile the clean record
        cleaned_records.append({
            "repository_name": row["repository_name"],
            "file_path": row["file_path"],
            "language": row["language"],
            "source_code": code,
            "character_count": char_count,
            "line_count": line_count
        })
        
    df_clean = pd.DataFrame(cleaned_records)
    output_path = os.path.join(INTERMEDIATE_DIR, "clean_source_dataset.csv")
    df_clean.to_csv(output_path, index=False)
    print(f"[+] Cleaning complete. Kept {len(df_clean)} / {len(df)} files. Saved to {output_path}")
    return df_clean

def run_preprocessing() -> None:
    input_path = os.path.join(INTERMEDIATE_DIR, "source_code_dataset.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source code dataset not found: {input_path}. Run extractor first.")
        
    df = pd.read_csv(input_path)
    clean_source_code(df)

if __name__ == "__main__":
    run_preprocessing()
