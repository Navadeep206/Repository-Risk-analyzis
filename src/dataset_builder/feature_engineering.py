#!/usr/bin/env python3
"""
Feature engineering stage for Phase 3.5.
Generates code change frequencies, repository ages, and average complexities for multiple repositories.
"""

import os
import pandas as pd
import sys
from typing import Optional, Dict

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs_exist

def calculate_repo_ages() -> Dict[str, int]:
    """
    Calculates repository ages in days based on commit dates in raw files.
    
    Returns:
        A dictionary mapping repository names to age in days.
    """
    repo_ages = {}
    for file in os.listdir(RAW_DIR):
        if file.endswith("_commits.csv"):
            repo_name = file[:-12]  # Strip "_commits.csv"
            try:
                df_commits = pd.read_csv(os.path.join(RAW_DIR, file))
                if not df_commits.empty and "committer_date" in df_commits.columns:
                    dates = pd.to_datetime(df_commits["committer_date"], errors="coerce", utc=True)
                    latest = dates.max()
                    oldest = dates.min()
                    if pd.notna(latest) and pd.notna(oldest):
                        age_days = (latest - oldest).days
                    else:
                        age_days = 1
                else:
                    age_days = 1
                if age_days <= 0:
                    age_days = 1
                repo_ages[repo_name] = age_days
            except Exception as e:
                print(f"[-] Warning: Failed to compute age for {repo_name}: {e}")
                repo_ages[repo_name] = 1
    return repo_ages

def engineer_features(output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Performs feature engineering on the merged dataset.
    
    Args:
        output_file: Optional target file path for CSV.
        
    Returns:
        A pandas DataFrame containing engineered features.
    """
    merged_file = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
    if not os.path.exists(merged_file):
        raise FileNotFoundError(f"Merged dataset file does not exist: {merged_file}")
        
    df = pd.read_csv(merged_file)
    
    # 1. Compute repository age in days for each repository
    repo_ages = calculate_repo_ages()
    
    # Map repository age to each row
    df["repository_age_days"] = df["repository_name"].map(repo_ages).fillna(1).astype(int)
    
    # 2. Calculate commit_frequency and file_change_frequency
    df["commit_frequency"] = df["commit_count"] / df["repository_age_days"]
    df["file_change_frequency"] = df["modification_count"] / df["repository_age_days"]
    
    # 3. Add modifications_per_file (alias to modification_count)
    df["modifications_per_file"] = df["modification_count"]
    
    # 4. Add average_maintainability (alias to maintainability_index)
    df["average_maintainability"] = df["maintainability_index"]
    
    # 5. Calculate average_complexity
    df["average_complexity"] = df["complexity"]  # Default fallback
    
    # Try reading function metrics to compute mean function complexity per file
    func_file = os.path.join(RAW_DIR, "function_metrics.csv")
    if os.path.exists(func_file):
        try:
            df_func = pd.read_csv(func_file)
            if not df_func.empty and "file_path" in df_func.columns and "complexity" in df_func.columns:
                func_agg = df_func.groupby("file_path")["complexity"].mean().reset_index().rename(
                    columns={"complexity": "mean_func_complexity"}
                )
                df = pd.merge(df, func_agg, on="file_path", how="left")
                # Fallback to file complexity if no functions are present
                df["average_complexity"] = df["mean_func_complexity"].fillna(df["complexity"])
                df = df.drop(columns=["mean_func_complexity"])
        except Exception as e:
            print(f"[-] Warning: Failed to parse function metrics for average complexity: {e}")
            
    if not output_file:
        output_file = os.path.join(PROCESSED_DIR, "engineered_dataset.csv")
        
    ensure_dirs_exist()
    df.to_csv(output_file, index=False)
    print(f"[+] Engineered dataset saved to {output_file}. Total rows: {len(df)}")
    return df

if __name__ == "__main__":
    engineer_features()
