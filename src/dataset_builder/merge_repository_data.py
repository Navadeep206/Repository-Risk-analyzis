#!/usr/bin/env python3
"""
Source dataset merger for Phase 3.5.
Combines static quality metrics with repository modifications, commit aggregates,
and calculates bug_fix_commit_count per file to resolve target leakage.
"""

import os
import pandas as pd
import sys
from typing import Optional, List

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs_exist

def merge_repository_data(output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Merges repository metadata, static quality metrics, and file modifications.
    Detects bug-fix commits using message keywords and computes bug_fix_commit_count.
    Writes the merged results to data/processed/merged_dataset.csv.
    
    Args:
        output_file: Optional target file path for CSV.
        
    Returns:
        A pandas DataFrame of merged metrics.
    """
    # 1. Load consolidated quality metrics (file path, loc, cyclomatic_complexity, maintainability_index)
    quality_file = os.path.join(PROCESSED_DIR, "quality_metrics.csv")
    if not os.path.exists(quality_file):
        raise FileNotFoundError(f"Processed quality metrics file does not exist: {quality_file}")
        
    df_quality = pd.read_csv(quality_file)
    
    # 2. Find all repositories that have been mined (matching *_modifications.csv)
    repo_names = []
    for file in os.listdir(RAW_DIR):
        if file.endswith("_modifications.csv"):
            repo_name = file[:-18]  # Strip "_modifications.csv"
            repo_names.append(repo_name)
            
    print(f"[*] Found mined repositories for merge: {repo_names}")
    
    merged_dfs: List[pd.DataFrame] = []
    
    for repo_name in repo_names:
        # Filter quality metrics for this repository
        df_quality_repo = df_quality[df_quality["repository_name"] == repo_name].copy()
        
        # Load modifications CSV for this repository
        mod_file = os.path.join(RAW_DIR, f"{repo_name}_modifications.csv")
        if not os.path.exists(mod_file):
            print(f"[-] Warning: Modifications file for {repo_name} does not exist at {mod_file}. Skipping.")
            continue
            
        df_mod = pd.read_csv(mod_file)
        
        # Load commits to find bug-fixing commits
        commits_file = os.path.join(RAW_DIR, f"{repo_name}_commits.csv")
        bug_fix_hashes = set()
        if os.path.exists(commits_file):
            try:
                df_commits = pd.read_csv(commits_file)
                df_commits["message"] = df_commits["message"].fillna("").astype(str)
                
                # Keywords to identify bug-fixing commits
                keywords = ["fix", "bug", "hotfix", "regression", "patch", "issue"]
                pattern = "|".join(keywords)
                
                df_commits["is_bug_fix"] = df_commits["message"].str.contains(pattern, case=False, na=False)
                bug_fix_hashes = set(df_commits[df_commits["is_bug_fix"]]["commit_hash"])
                print(f"[*] Identified {len(bug_fix_hashes)} bug-fixing commits out of {len(df_commits)} total for {repo_name}")
            except Exception as e:
                print(f"[-] Failed to scan commits for {repo_name}: {e}")
        else:
            print(f"[-] Warning: Commits file for {repo_name} does not exist. No bug-fix history resolved.")
            
        if df_mod.empty:
            df_quality_repo["modification_count"] = 0
            df_quality_repo["commit_count"] = 0
            df_quality_repo["contributor_count"] = 0
            df_quality_repo["bug_fix_commit_count"] = 0
            merged_dfs.append(df_quality_repo)
            continue
            
        # Resolve file path: new_path if not null, else old_path
        df_mod["file_path"] = df_mod["new_path"].fillna(df_mod["old_path"])
        df_mod["is_bug_fix"] = df_mod["commit_hash"].isin(bug_fix_hashes)
        
        # Group by file_path to aggregate metrics
        mod_agg = df_mod.groupby("file_path").agg(
            modification_count=("commit_hash", "count"),
            commit_count=("commit_hash", "nunique"),
            contributor_count=("author_email", "nunique"),
            bug_fix_commit_count=("commit_hash", lambda x: int(x[df_mod.loc[x.index, "is_bug_fix"]].nunique()))
        ).reset_index()
        
        # Merge quality metrics with modification aggregates
        repo_merged = pd.merge(df_quality_repo, mod_agg, on="file_path", how="left")
        
        # Fill missing values with 0
        repo_merged["modification_count"] = repo_merged["modification_count"].fillna(0).astype(int)
        repo_merged["commit_count"] = repo_merged["commit_count"].fillna(0).astype(int)
        repo_merged["contributor_count"] = repo_merged["contributor_count"].fillna(0).astype(int)
        repo_merged["bug_fix_commit_count"] = repo_merged["bug_fix_commit_count"].fillna(0).astype(int)
        
        merged_dfs.append(repo_merged)
        
    if merged_dfs:
        final_merged = pd.concat(merged_dfs, ignore_index=True)
    else:
        # Default empty with structure
        final_merged = pd.DataFrame(columns=[
            "repository_name", "file_path", "language", "loc", "complexity",
            "maintainability_index", "commit_count", "modification_count", "contributor_count", "bug_fix_commit_count"
        ])
        
    # Rename cyclomatic_complexity to complexity if present
    if "cyclomatic_complexity" in final_merged.columns:
        final_merged = final_merged.rename(columns={"cyclomatic_complexity": "complexity"})
        
    # Ensure correct columns order and structure
    required_cols = [
        "repository_name", "file_path", "language", "loc", "complexity",
        "maintainability_index", "commit_count", "modification_count", "contributor_count", "bug_fix_commit_count"
    ]
    
    # Reindex columns to handle case differences or missing cols
    final_merged = final_merged.reindex(columns=required_cols)
    
    if not output_file:
        output_file = os.path.join(PROCESSED_DIR, "merged_dataset.csv")
        
    ensure_dirs_exist()
    final_merged.to_csv(output_file, index=False)
    print(f"[+] Merged dataset saved to {output_file}. Total rows: {len(final_merged)}")
    return final_merged

if __name__ == "__main__":
    merge_repository_data()
