#!/usr/bin/env python3
"""
Dataset split stage for Phase 3.5.
Implements repository-aware group stratified splitting and file-level stratified fallback splitting.
Saves ml_dataset_v2.csv and corresponding splits in data/final/.
"""

import os
import pandas as pd
import sys
import itertools
from typing import Tuple, List, Dict, Set

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import PROCESSED_DIR, FINAL_DIR, ensure_dirs_exist

def find_best_group_split(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Searches all possible partition assignments of repositories to Train/Val/Test
    to find the layout closest to 70/15/15 that maximizes label presence coverage.
    """
    repos = df["repository_name"].unique()
    n_repos = len(repos)
    
    # Pre-calculate sizes and unique labels per repository
    repo_data = {}
    for repo in repos:
        repo_df = df[df["repository_name"] == repo]
        repo_data[repo] = {
            "size": len(repo_df),
            "labels": set(repo_df["historical_risk_label"].unique())
        }
        
    total_len = len(df)
    best_assignment = None
    best_score = float("inf")
    best_coverage = 0
    
    # If N is small, do brute force. Otherwise, do randomized search to avoid exponential explosion.
    if n_repos <= 9:
        iterator = itertools.product([0, 1, 2], repeat=n_repos)
    else:
        import random
        random.seed(42)
        sampled = set()
        # Generate up to 20,000 unique random assignments
        while len(sampled) < 20000:
            assignment = tuple(random.choice([0, 1, 2]) for _ in range(n_repos))
            sampled.add(assignment)
        iterator = list(sampled)
        
    for assignment in iterator:
        train_len = 0
        val_len = 0
        test_len = 0
        
        train_labels: Set[str] = set()
        val_labels: Set[str] = set()
        test_labels: Set[str] = set()
        
        for idx, split_idx in enumerate(assignment):
            repo = repos[idx]
            size = repo_data[repo]["size"]
            labels = repo_data[repo]["labels"]
            
            if split_idx == 0:
                train_len += size
                train_labels.update(labels)
            elif split_idx == 1:
                val_len += size
                val_labels.update(labels)
            else:
                test_len += size
                test_labels.update(labels)
                
        # Calculate distinct label coverage across splits (maximum is 9: 3 splits * 3 labels)
        coverage = len(train_labels) + len(val_labels) + len(test_labels)
        
        # Calculate split ratios
        train_frac = train_len / total_len if total_len > 0 else 0
        val_frac = val_len / total_len if total_len > 0 else 0
        test_frac = test_len / total_len if total_len > 0 else 0
        
        # Score deviation from 70/15/15 using Mean Squared Error
        score = (train_frac - 0.70)**2 + (val_frac - 0.15)**2 + (test_frac - 0.15)**2
        
        # Prioritize higher label coverage first, then minimize deviation
        if coverage > best_coverage:
            best_coverage = coverage
            best_score = score
            best_assignment = assignment
        elif coverage == best_coverage:
            if score < best_score:
                best_score = score
                best_assignment = assignment
                
    # Map the best assignment to repo lists
    train_repos = []
    val_repos = []
    test_repos = []
    
    if best_assignment is not None:
        for idx, split_idx in enumerate(best_assignment):
            repo = repos[idx]
            if split_idx == 0:
                train_repos.append(repo)
            elif split_idx == 1:
                val_repos.append(repo)
            else:
                test_repos.append(repo)
                
    return train_repos, val_repos, test_repos

def split_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Saves the master ml_dataset_v2.csv and splits it into train/validation/test sets (70/15/15).
    
    Returns:
        A tuple of (master_df, train_df, val_df, test_df).
    """
    clean_file = os.path.join(PROCESSED_DIR, "clean_dataset.csv")
    if not os.path.exists(clean_file):
        raise FileNotFoundError(f"Clean dataset file does not exist: {clean_file}")
        
    df = pd.read_csv(clean_file)
    
    # 1. Select master columns for v2
    required_cols = [
        "repository_name", "file_path", "language", "loc", "complexity",
        "maintainability_index", "commit_count", "modification_count",
        "contributor_count", "commit_frequency", "repository_age_days",
        "bug_fix_commit_count", "ownership_concentration", "contributor_entropy",
        "bus_factor", "recent_churn", "time_decayed_churn", "time_since_last_bug_fix",
        "historical_bug_density", "historical_risk_label"
    ]
    
    # Ensure all required columns are present (safe fallback)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
            
    df_master = df[required_cols].copy()
    
    # Save the master dataset v2
    ml_file = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    ensure_dirs_exist()
    df_master.to_csv(ml_file, index=False)
    print(f"[+] Master ML dataset saved to {ml_file}. Total rows: {len(df_master)}")
    
    # 2. Perform splitting
    repos = df_master["repository_name"].unique()
    
    if len(repos) >= 3:
        print(f"[*] Found {len(repos)} repositories. Running repository-safe group stratified split...")
        train_repos, val_repos, test_repos = find_best_group_split(df_master)
        
        df_train = df_master[df_master["repository_name"].isin(train_repos)].copy()
        df_val = df_master[df_master["repository_name"].isin(val_repos)].copy()
        df_test = df_master[df_master["repository_name"].isin(test_repos)].copy()
        
        print(f"[+] Repository assignment: Train={train_repos}, Val={val_repos}, Test={test_repos}")
    else:
        print(f"[!] Warning: Only {len(repos)} repository found/mined. Group splitting is impossible.")
        print("[*] Falling back to file-level stratified split to ensure label distribution...")
        
        # Stratified split at the file level grouping by historical_risk_label
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for label, group_df in df_master.groupby("historical_risk_label"):
            if len(group_df) >= 3:
                g_train = group_df.sample(frac=0.7, random_state=42)
                g_temp = group_df.drop(g_train.index)
                g_val = g_temp.sample(frac=0.5, random_state=42)
                g_test = g_temp.drop(g_val.index)
                train_dfs.append(g_train)
                val_dfs.append(g_val)
                test_dfs.append(g_test)
            else:
                # Handle edge cases for very small splits (e.g. only 1 or 2 samples of a label)
                # Assign them all to train to avoid empty sets
                train_dfs.append(group_df)
                
        df_train = pd.concat(train_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
        df_val = pd.concat(val_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True) if val_dfs else pd.DataFrame(columns=required_cols)
        df_test = pd.concat(test_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True) if test_dfs else pd.DataFrame(columns=required_cols)
        
    train_file = os.path.join(FINAL_DIR, "train_v2.csv")
    val_file = os.path.join(FINAL_DIR, "validation_v2.csv")
    test_file = os.path.join(FINAL_DIR, "test_v2.csv")
    
    df_train.to_csv(train_file, index=False)
    df_val.to_csv(val_file, index=False)
    df_test.to_csv(test_file, index=False)
    
    # Also save as train.csv, validation.csv, test.csv for Phase D compatibility
    df_train.to_csv(os.path.join(FINAL_DIR, "train.csv"), index=False)
    df_val.to_csv(os.path.join(FINAL_DIR, "validation.csv"), index=False)
    df_test.to_csv(os.path.join(FINAL_DIR, "test.csv"), index=False)
    
    print(f"[+] Split sizes -> Train: {len(df_train)} ({len(df_train)/len(df_master)*100:.2f}%), "
          f"Val: {len(df_val)} ({len(df_val)/len(df_master)*100:.2f}%), "
          f"Test: {len(df_test)} ({len(df_test)/len(df_master)*100:.2f}%)")
          
    return df_master, df_train, df_val, df_test

if __name__ == "__main__":
    split_dataset()
