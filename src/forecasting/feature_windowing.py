#!/usr/bin/env python3
"""
Feature Windowing for Phase 9.
Generates past rolling window features and future forecast targets
for each weekly snapshot.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROCESSED_DIR, FINAL_DIR, ensure_dirs_exist

def build_forecasting_dataset(input_file: str = None, output_file: str = None) -> pd.DataFrame:
    """
    Loads daily logs, performs rolling window aggregation (past 30d, 60d, 90d),
    creates future targets (30d, 60d, 90d), and saves the final forecasting dataset.
    """
    if not input_file:
        input_file = os.path.join(PROCESSED_DIR, "..", "intermediate", "daily_repo_logs.csv")
    if not output_file:
        output_file = os.path.join(FINAL_DIR, "forecasting_dataset.csv")
        
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Daily repo logs file not found: {input_file}")
        
    df_daily = pd.read_csv(input_file)
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    
    # Calculate repository-wide globals from daily data for fallbacks
    repo_globals = {}
    for repo, group in df_daily.groupby("repository_name"):
        comp_count = group["complexity_count"].sum()
        maint_count = group["maintainability_count"].sum()
        
        comp_avg = group["complexity_sum"].sum() / comp_count if comp_count > 0 else 0.0
        maint_avg = group["maintainability_sum"].sum() / maint_count if maint_count > 0 else 100.0
        repo_globals[repo] = (comp_avg, maint_avg)
        
    snapshots: List[dict] = []
    
    # Process each repository separately
    for repo_name, repo_df in df_daily.groupby("repository_name"):
        repo_df = repo_df.sort_values("date").set_index("date")
        
        min_date = repo_df.index.min()
        max_date = repo_df.index.max()
        
        # Snapshot frequency: weekly (7 days). Start 90 days in, end 90 days before the end.
        start_date = min_date + pd.Timedelta(days=90)
        end_date = max_date - pd.Timedelta(days=90)
        
        if start_date > end_date:
            print(f"[-] Repository {repo_name} has too short a timeline ({ (max_date - min_date).days } days). Skipping.")
            continue
            
        snapshot_dates = pd.date_range(start=start_date, end=end_date, freq="7D")
        print(f"[*] Repository {repo_name}: generating {len(snapshot_dates)} snapshots from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        fallback_comp, fallback_maint = repo_globals.get(repo_name, (0.0, 100.0))
        
        for T in snapshot_dates:
            row = {
                "repository_name": repo_name,
                "snapshot_date": T.strftime("%Y-%m-%d"),
            }
            
            # --- PAST FEATURES ---
            for W in [30, 60, 90]:
                past_df = repo_df.loc[T - pd.Timedelta(days=W) : T]
                
                # Commit counts (frequency)
                row[f"commit_frequency_{W}d"] = int(past_df["commits_count"].sum())
                row[f"defect_count_{W}d"] = int(past_df["bug_fixes_count"].sum())
                
                # Contributor sets
                emails = set()
                for em_str in past_df["contributor_emails"].dropna():
                    if em_str:
                        emails.update([e.strip() for e in em_str.split(";") if e.strip()])
                row[f"active_contributors_{W}d"] = len(emails)
                
                # Modifications
                row[f"modification_count_{W}d"] = int(past_df["modifications_count"].sum())
                
                # Average metrics
                comp_c = past_df["complexity_count"].sum()
                maint_c = past_df["maintainability_count"].sum()
                
                row[f"avg_complexity_{W}d"] = float(past_df["complexity_sum"].sum() / comp_c) if comp_c > 0 else fallback_comp
                row[f"avg_maintainability_{W}d"] = float(past_df["maintainability_sum"].sum() / maint_c) if maint_c > 0 else fallback_maint
                
                # Risk score
                row[f"risk_score_{W}d"] = row[f"defect_count_{W}d"] * 5.0 + row[f"modification_count_{W}d"] * 0.1
                
            # --- FUTURE TARGETS ---
            for H in [30, 60, 90]:
                future_df = repo_df.loc[T + pd.Timedelta(days=1) : T + pd.Timedelta(days=H)]
                
                future_defects = int(future_df["bug_fixes_count"].sum())
                future_mods = int(future_df["modifications_count"].sum())
                
                row[f"future_defect_count_{H}d"] = future_defects
                row[f"future_modification_intensity_{H}d"] = future_mods
                row[f"future_risk_{H}d"] = future_defects * 5.0 + future_mods * 0.1
                
            snapshots.append(row)
            
    df_snapshots = pd.DataFrame(snapshots)
    ensure_dirs_exist()
    
    # Create final directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_snapshots.to_csv(output_file, index=False)
    print(f"[+] Saved final forecasting dataset to {output_file}. Total rows: {len(df_snapshots)}")
    return df_snapshots

if __name__ == "__main__":
    build_forecasting_dataset()
