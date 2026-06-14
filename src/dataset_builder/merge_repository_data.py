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
            df_quality_repo["ownership_concentration"] = 1.0
            df_quality_repo["contributor_entropy"] = 0.0
            df_quality_repo["bus_factor"] = 1
            df_quality_repo["recent_churn"] = 0.0
            df_quality_repo["time_decayed_churn"] = 0.0
            df_quality_repo["time_since_last_bug_fix"] = 1000.0
            df_quality_repo["historical_bug_density"] = 0.0
            merged_dfs.append(df_quality_repo)
            continue
            
        # Resolve file path: new_path if not null, else old_path
        df_mod["file_path"] = df_mod["new_path"].fillna(df_mod["old_path"])
        df_mod["is_bug_fix"] = df_mod["commit_hash"].isin(bug_fix_hashes)
        
        # Prepare columns for calculation
        df_mod["added_lines"] = pd.to_numeric(df_mod["added_lines"], errors="coerce").fillna(0.0)
        df_mod["deleted_lines"] = pd.to_numeric(df_mod["deleted_lines"], errors="coerce").fillna(0.0)
        df_mod["author_email"] = df_mod["author_email"].fillna("unknown").astype(str)
        
        # Calculate commit_date timestamps and days ago relative to the latest commit in the repository
        df_mod["commit_date_dt"] = pd.to_datetime(df_mod["commit_date"], errors="coerce", utc=True)
        latest_repo_date = df_mod["commit_date_dt"].max()
        if pd.isna(latest_repo_date):
            latest_repo_date = pd.Timestamp.now(tz="UTC")
            
        df_mod["days_ago"] = (latest_repo_date - df_mod["commit_date_dt"]).dt.total_seconds() / (24 * 3600.0)
        df_mod["days_ago"] = df_mod["days_ago"].fillna(1000.0).clip(lower=0.0)
        
        # Group by file_path to aggregate baseline metrics
        mod_agg = df_mod.groupby("file_path").agg(
            modification_count=("commit_hash", "count"),
            commit_count=("commit_hash", "nunique"),
            contributor_count=("author_email", "nunique"),
            bug_fix_commit_count=("commit_hash", lambda x: int(x[df_mod.loc[x.index, "is_bug_fix"]].nunique()))
        ).reset_index()
        
        # 1. HHI, Contributor Entropy, Bus Factor
        import numpy as np
        author_counts = df_mod.groupby(["file_path", "author_email"]).size().reset_index(name="count")
        file_totals = df_mod.groupby("file_path").size().reset_index(name="total")
        
        author_props = pd.merge(author_counts, file_totals, on="file_path")
        author_props["p_a"] = author_props["count"] / author_props["total"]
        author_props["p_a_squared"] = author_props["p_a"] ** 2
        author_props["entropy_term"] = -author_props["p_a"] * np.log2(author_props["p_a"] + 1e-9)
        
        hhi = author_props.groupby("file_path")["p_a_squared"].sum().reset_index(name="ownership_concentration")
        entropy = author_props.groupby("file_path")["entropy_term"].sum().reset_index(name="contributor_entropy")
        
        def calc_bus_factor(props):
            sorted_props = sorted(props, reverse=True)
            cum_sum = 0.0
            count = 0
            for p in sorted_props:
                cum_sum += p
                count += 1
                if cum_sum >= 0.5:
                    break
            return count
            
        bus_factor = author_props.groupby("file_path")["p_a"].apply(calc_bus_factor).reset_index(name="bus_factor")
        
        # 2. Recent Churn (added + deleted lines in last 30 days)
        recent_mask = df_mod["days_ago"] <= 30.0
        # If there are no modifications in the last 30 days, we handle the empty group case
        recent_mods = df_mod[recent_mask]
        if not recent_mods.empty:
            recent_churn = recent_mods.groupby("file_path").apply(
                lambda x: float((x["added_lines"] + x["deleted_lines"]).sum())
            ).reset_index(name="recent_churn")
        else:
            recent_churn = pd.DataFrame(columns=["file_path", "recent_churn"])
            
        # 3. Time-decayed Churn: sum of (added_lines + deleted_lines) * exp(-0.01 * days_ago)
        df_mod["weighted_churn"] = (df_mod["added_lines"] + df_mod["deleted_lines"]) * np.exp(-0.01 * df_mod["days_ago"])
        decayed_churn = df_mod.groupby("file_path")["weighted_churn"].sum().reset_index(name="time_decayed_churn")
        
        # 4. Time since last bug fix (days)
        bug_fix_mods = df_mod[df_mod["is_bug_fix"]]
        if not bug_fix_mods.empty:
            time_since_bug_fix = bug_fix_mods.groupby("file_path")["days_ago"].min().reset_index(name="time_since_last_bug_fix")
        else:
            time_since_bug_fix = pd.DataFrame(columns=["file_path", "time_since_last_bug_fix"])
            
        # Merge all modification features
        mod_features = mod_agg
        mod_features = pd.merge(mod_features, hhi, on="file_path", how="left")
        mod_features = pd.merge(mod_features, entropy, on="file_path", how="left")
        mod_features = pd.merge(mod_features, bus_factor, on="file_path", how="left")
        mod_features = pd.merge(mod_features, recent_churn, on="file_path", how="left")
        mod_features = pd.merge(mod_features, decayed_churn, on="file_path", how="left")
        mod_features = pd.merge(mod_features, time_since_bug_fix, on="file_path", how="left")
        
        # Merge quality metrics with modification aggregates
        repo_merged = pd.merge(df_quality_repo, mod_features, on="file_path", how="left")
        
        # Fill missing values
        repo_merged["modification_count"] = repo_merged["modification_count"].fillna(0).astype(int)
        repo_merged["commit_count"] = repo_merged["commit_count"].fillna(0).astype(int)
        repo_merged["contributor_count"] = repo_merged["contributor_count"].fillna(0).astype(int)
        repo_merged["bug_fix_commit_count"] = repo_merged["bug_fix_commit_count"].fillna(0).astype(int)
        
        repo_merged["ownership_concentration"] = repo_merged["ownership_concentration"].fillna(1.0)
        repo_merged["contributor_entropy"] = repo_merged["contributor_entropy"].fillna(0.0)
        repo_merged["bus_factor"] = repo_merged["bus_factor"].fillna(1).astype(int)
        repo_merged["recent_churn"] = repo_merged["recent_churn"].fillna(0.0)
        repo_merged["time_decayed_churn"] = repo_merged["time_decayed_churn"].fillna(0.0)
        repo_merged["time_since_last_bug_fix"] = repo_merged["time_since_last_bug_fix"].fillna(1000.0)
        
        # 5. Historical Bug Density (bug_fix_commit_count / loc)
        repo_merged["historical_bug_density"] = (repo_merged["bug_fix_commit_count"] / repo_merged["loc"].replace(0, 1.0)).fillna(0.0)
        
        merged_dfs.append(repo_merged)
        
    if merged_dfs:
        final_merged = pd.concat(merged_dfs, ignore_index=True)
    else:
        # Default empty with structure
        final_merged = pd.DataFrame(columns=[
            "repository_name", "file_path", "language", "loc", "complexity",
            "maintainability_index", "commit_count", "modification_count", "contributor_count", "bug_fix_commit_count",
            "ownership_concentration", "contributor_entropy", "bus_factor", "recent_churn", "time_decayed_churn",
            "time_since_last_bug_fix", "historical_bug_density"
        ])
        
    # Rename cyclomatic_complexity to complexity if present
    if "cyclomatic_complexity" in final_merged.columns:
        final_merged = final_merged.rename(columns={"cyclomatic_complexity": "complexity"})
        
    # Ensure correct columns order and structure
    required_cols = [
        "repository_name", "file_path", "language", "loc", "complexity",
        "maintainability_index", "commit_count", "modification_count", "contributor_count", "bug_fix_commit_count",
        "ownership_concentration", "contributor_entropy", "bus_factor", "recent_churn", "time_decayed_churn",
        "time_since_last_bug_fix", "historical_bug_density"
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
