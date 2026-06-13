#!/usr/bin/env python3
"""
Label generation stage for Phase 3.5.
Calculates historical bug-fixing labels based on commit history keywords.
"""

import os
import pandas as pd
import sys
from typing import Optional

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROCESSED_DIR, ensure_dirs_exist

def calculate_historical_label(bug_fix_count: int) -> str:
    """
    Categorizes files into LOW, MEDIUM, or HIGH risk based on historical bug-fix commits.
    
    Logic:
        - LOW: 0 bug-fixing commits
        - MEDIUM: 1 to 2 bug-fixing commits
        - HIGH: 3 or more bug-fixing commits
        
    Args:
        bug_fix_count: Number of bug-fixing commits affecting this file.
        
    Returns:
        Categorical risk string ("LOW", "MEDIUM", or "HIGH").
    """
    if bug_fix_count == 0:
        return "LOW"
    elif bug_fix_count <= 2:
        return "MEDIUM"
    else:
        return "HIGH"

def generate_labels(output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Generates historical risk labels for all files in the engineered dataset.
    
    Args:
        output_file: Optional target file path for CSV.
        
    Returns:
        A pandas DataFrame containing calculated historical risk labels.
    """
    engineered_file = os.path.join(PROCESSED_DIR, "engineered_dataset.csv")
    if not os.path.exists(engineered_file):
        raise FileNotFoundError(f"Engineered dataset file does not exist: {engineered_file}")
        
    df = pd.read_csv(engineered_file)
    
    # Ensure bug_fix_commit_count exists and fill NA
    if "bug_fix_commit_count" not in df.columns:
        df["bug_fix_commit_count"] = 0
    else:
        df["bug_fix_commit_count"] = df["bug_fix_commit_count"].fillna(0).astype(int)
        
    # Generate historical_risk_label
    df["historical_risk_label"] = df["bug_fix_commit_count"].apply(calculate_historical_label)
    
    if not output_file:
        output_file = os.path.join(PROCESSED_DIR, "labeled_dataset.csv")
        
    ensure_dirs_exist()
    df.to_csv(output_file, index=False)
    print(f"[+] Labeled dataset saved to {output_file}. Total rows: {len(df)}")
    print(f"[+] Label distribution:\n{df['historical_risk_label'].value_counts()}")
    return df

if __name__ == "__main__":
    generate_labels()
