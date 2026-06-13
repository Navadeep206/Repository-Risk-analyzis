#!/usr/bin/env python3
"""
Temporal Dataset Builder for Phase 9.
Generates daily repository logs of activity, contributors, and quality metrics
from raw commits and modifications files.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Set, Tuple

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import RAW_DIR, PROCESSED_DIR, ensure_dirs_exist

def normalize_path(path: str) -> str:
    """Normalizes file paths to avoid mismatch."""
    if pd.isna(path) or not isinstance(path, str):
        return ""
    return path.replace("\\", "/").strip("/")

def build_daily_logs() -> pd.DataFrame:
    """
    Scans raw directory, identifies mined repositories, and generates daily logs of:
    - Commits and bug-fixes
    - Modifications
    - Unique contributor emails
    - Aggregated complexity and maintainability index of modified files
    
    Saves results to data/intermediate/daily_repo_logs.csv.
    """
    ensure_dirs_exist()
    
    # 1. Load quality metrics for static complexity/maintainability lookup
    quality_file = os.path.join(PROCESSED_DIR, "quality_metrics.csv")
    if not os.path.exists(quality_file):
        raise FileNotFoundError(f"Quality metrics file not found: {quality_file}")
    df_quality = pd.read_csv(quality_file)
    
    # Normalize paths in quality metrics
    df_quality["norm_path"] = df_quality["file_path"].apply(normalize_path)
    
    # Map (repo_name, norm_path) -> (complexity, maintainability)
    metrics_lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
    repo_globals: Dict[str, Tuple[float, float]] = {} # Repo-wide average complexity and maintainability
    
    for _, row in df_quality.iterrows():
        repo = row["repository_name"]
        path = row["norm_path"]
        comp = float(row.get("complexity", 0.0))
        maint = float(row.get("maintainability_index", 100.0))
        metrics_lookup[(repo, path)] = (comp, maint)
        
    # Calculate repository-wide averages for fallbacks
    for repo, group in df_quality.groupby("repository_name"):
        comp_avg = group["complexity"].mean() if "complexity" in group.columns else 0.0
        maint_avg = group["maintainability_index"].mean() if "maintainability_index" in group.columns else 100.0
        repo_globals[repo] = (comp_avg if pd.notna(comp_avg) else 0.0, maint_avg if pd.notna(maint_avg) else 100.0)

    # 2. Find mined repositories
    repo_names = []
    for file in os.listdir(RAW_DIR):
        if file.endswith("_modifications.csv"):
            repo_name = file[:-18]
            repo_names.append(repo_name)
            
    print(f"[*] Found repositories for temporal building: {repo_names}")
    
    all_daily_logs = []
    
    for repo_name in repo_names:
        commits_file = os.path.join(RAW_DIR, f"{repo_name}_commits.csv")
        mod_file = os.path.join(RAW_DIR, f"{repo_name}_modifications.csv")
        
        if not os.path.exists(commits_file) or not os.path.exists(mod_file):
            print(f"[-] Missing commits or modifications for {repo_name}. Skipping.")
            continue
            
        # Load commits
        df_commits = pd.read_csv(commits_file)
        if df_commits.empty:
            print(f"[-] Empty commits for {repo_name}. Skipping.")
            continue
            
        # Parse commit dates (timezone naive UTC)
        df_commits["committer_date"] = pd.to_datetime(df_commits["committer_date"], errors="coerce", utc=True).dt.tz_localize(None)
        df_commits = df_commits.dropna(subset=["committer_date"])
        
        # Identify bug-fix commits
        keywords = ["fix", "bug", "hotfix", "regression", "patch", "issue"]
        pattern = "|".join(keywords)
        df_commits["message"] = df_commits["message"].fillna("").astype(str)
        df_commits["is_bug_fix"] = df_commits["message"].str.contains(pattern, case=False, na=False)
        bug_fix_hashes = set(df_commits[df_commits["is_bug_fix"]]["commit_hash"])
        
        print(f"[*] Repository {repo_name}: {len(df_commits)} commits, {len(bug_fix_hashes)} bug-fixes")
        
        # Load modifications
        df_mod = pd.read_csv(mod_file)
        if df_mod.empty:
            print(f"[-] Empty modifications for {repo_name}. Skipping.")
            continue
            
        # Parse modification dates
        df_mod["commit_date"] = pd.to_datetime(df_mod["commit_date"], errors="coerce", utc=True).dt.tz_localize(None)
        df_mod = df_mod.dropna(subset=["commit_date"])
        df_mod["file_path"] = df_mod["new_path"].fillna(df_mod["old_path"]).apply(normalize_path)
        df_mod["is_bug_fix"] = df_mod["commit_hash"].isin(bug_fix_hashes)
        
        # Resolve static metrics for each modification
        complexities = []
        maintainabilities = []
        fallback_comp, fallback_maint = repo_globals.get(repo_name, (0.0, 100.0))
        
        for _, row in df_mod.iterrows():
            fpath = row["file_path"]
            comp, maint = metrics_lookup.get((repo_name, fpath), (fallback_comp, fallback_maint))
            complexities.append(comp)
            maintainabilities.append(maint)
            
        df_mod["static_complexity"] = complexities
        df_mod["static_maintainability"] = maintainabilities
        
        # Determine range of dates
        min_date = min(df_commits["committer_date"].min(), df_mod["commit_date"].min()).normalize()
        max_date = max(df_commits["committer_date"].max(), df_mod["commit_date"].max()).normalize()
        date_range = pd.date_range(start=min_date, end=max_date, freq="D")
        
        # Aggregate commits by day
        df_commits["date_only"] = df_commits["committer_date"].dt.normalize()
        commit_agg = df_commits.groupby("date_only").agg(
            commits_count=("commit_hash", "count"),
            bug_fixes_count=("is_bug_fix", "sum"),
            contributors=("author_email", lambda x: ";".join(x.dropna().unique()))
        ).reindex(date_range)
        
        # Aggregate modifications by day
        df_mod["date_only"] = df_mod["commit_date"].dt.normalize()
        mod_agg = df_mod.groupby("date_only").agg(
            modifications_count=("commit_hash", "count"),
            complexity_sum=("static_complexity", "sum"),
            complexity_count=("static_complexity", "count"),
            maintainability_sum=("static_maintainability", "sum"),
            maintainability_count=("static_maintainability", "count")
        ).reindex(date_range)
        
        # Combine aggregations
        repo_daily = pd.DataFrame(index=date_range)
        repo_daily["repository_name"] = repo_name
        repo_daily["date"] = repo_daily.index.strftime("%Y-%m-%d")
        
        repo_daily["commits_count"] = commit_agg["commits_count"].fillna(0).astype(int)
        repo_daily["bug_fixes_count"] = commit_agg["bug_fixes_count"].fillna(0).astype(int)
        repo_daily["contributor_emails"] = commit_agg["contributors"].fillna("")
        
        repo_daily["modifications_count"] = mod_agg["modifications_count"].fillna(0).astype(int)
        repo_daily["complexity_sum"] = mod_agg["complexity_sum"].fillna(0.0)
        repo_daily["complexity_count"] = mod_agg["complexity_count"].fillna(0).astype(int)
        repo_daily["maintainability_sum"] = mod_agg["maintainability_sum"].fillna(0.0)
        repo_daily["maintainability_count"] = mod_agg["maintainability_count"].fillna(0).astype(int)
        
        all_daily_logs.append(repo_daily)
        print(f"[+] Prepared daily log for {repo_name} with {len(repo_daily)} days.")
        
    if all_daily_logs:
        df_final = pd.concat(all_daily_logs, ignore_index=True)
    else:
        df_final = pd.DataFrame()
        
    # Ensure intermediate dir exists
    intermediate_dir = os.path.join(PROCESSED_DIR, "..", "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    out_file = os.path.join(intermediate_dir, "daily_repo_logs.csv")
    df_final.to_csv(out_file, index=False)
    print(f"[+] Saved daily logs for all repos to {out_file}. Total rows: {len(df_final)}")
    return df_final

if __name__ == "__main__":
    build_daily_logs()
