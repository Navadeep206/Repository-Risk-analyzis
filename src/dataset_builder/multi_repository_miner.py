#!/usr/bin/env python3
"""
Multi-repository miner for Phase 3.5.
Clones target GitHub repositories (Python, JavaScript, TypeScript),
mines their histories, runs code quality audits, and creates repository metadata.
Optimized to skip extraction if outputs already exist.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any
from git import Repo

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_DIR, RAW_DIR, PROCESSED_DIR, REPOS_DIR, ensure_dirs_exist
from commit_extractor import extract_commits
from contributor_extractor import analyze_contributors
from modification_extractor import extract_modifications
from quality_metrics.quality_pipeline import run_quality_pipeline
from quality_metrics.language_detector import detect_languages

# Configured target repositories to mine (small, fast, diverse)
TARGET_REPOSITORIES = [
    {"name": "click", "url": "https://github.com/pallets/click", "lang": "python"},
    {"name": "jinja", "url": "https://github.com/pallets/jinja", "lang": "python"},
    {"name": "express", "url": "https://github.com/expressjs/express", "lang": "javascript"},
    {"name": "redux", "url": "https://github.com/reduxjs/redux", "lang": "typescript"},
    {"name": "axios", "url": "https://github.com/axios/axios", "lang": "typescript"},
    {"name": "lodash", "url": "https://github.com/lodash/lodash", "lang": "javascript"},
    {"name": "databases", "url": "https://github.com/encode/databases", "lang": "python"}
]

def mine_all_repositories() -> None:
    """
    Clones and mines all configured repositories, producing metadata and individual extraction files.
    """
    print("[*] Starting Multi-Repository Mining...")
    ensure_dirs_exist()
    
    metadata_records = []
    quality_dfs = []
    
    for repo_info in TARGET_REPOSITORIES:
        name = repo_info["name"]
        url = repo_info["url"]
        local_path = os.path.join(REPOS_DIR, name)
        
        print(f"\n==========================================")
        print(f"Mining Repository: {name} ({url})")
        print(f"==========================================")
        
        # Check if all outputs already exist to optimize execution
        commits_file = os.path.join(RAW_DIR, f"{name}_commits.csv")
        contribs_file = os.path.join(RAW_DIR, f"{name}_contributors.csv")
        mods_file = os.path.join(RAW_DIR, f"{name}_modifications.csv")
        quality_file = os.path.join(PROCESSED_DIR, f"{name}_quality_metrics.csv")
        lang_profile_file = os.path.join(RAW_DIR, f"{name}_language_profile.csv")
        
        if (os.path.exists(commits_file) and os.path.exists(contribs_file) and 
            os.path.exists(mods_file) and os.path.exists(quality_file) and 
            os.path.exists(lang_profile_file)):
            print(f"[*] All extracted outputs for {name} already exist. Skipping mining.")
            try:
                df_commits = pd.read_csv(commits_file)
                df_contribs = pd.read_csv(contribs_file)
                df_quality = pd.read_csv(quality_file)
                df_lang = pd.read_csv(lang_profile_file)
                dominant_lang = df_lang.iloc[0]["language"] if not df_lang.empty else "unknown"
                
                # Repository age
                age_days = 1
                if not df_commits.empty and "committer_date" in df_commits.columns:
                    dates = pd.to_datetime(df_commits["committer_date"], errors="coerce", utc=True)
                    latest = dates.max()
                    oldest = dates.min()
                    if pd.notna(latest) and pd.notna(oldest):
                        age_days = max((latest - oldest).days, 1)
                
                # Repository size (total LOC of analyzed files)
                total_loc = 0
                if not df_quality.empty and "loc" in df_quality.columns:
                    total_loc = int(df_quality["loc"].sum())
                    
                metadata_records.append({
                    "repository_name": name,
                    "repository_language": dominant_lang,
                    "repository_size": total_loc,
                    "repository_age": age_days,
                    "total_commits": len(df_commits),
                    "total_contributors": len(df_contribs)
                })
                quality_dfs.append(df_quality)
                print(f"[+] Loaded existing stats for {name}: Lang={dominant_lang}, Size={total_loc} LOC, Age={age_days} days, Commits={len(df_commits)}")
                continue
            except Exception as e:
                print(f"[-] Failed to load existing data for {name}: {e}. Re-mining.")
        
        # 1. Clone repo if it doesn't exist
        if not os.path.exists(local_path):
            print(f"[*] Cloning {name} to {local_path}...")
            try:
                Repo.clone_from(url, local_path)
                print(f"[+] Cloned successfully.")
            except Exception as e:
                print(f"[-] Failed to clone {name}: {e}. Skipping repository.")
                continue
        else:
            print(f"[*] Repository {name} already cloned at {local_path}.")
            
        # 2. Extract Commits
        try:
            df_commits = extract_commits(local_path, commits_file)
        except Exception as e:
            print(f"[-] Commit extraction failed for {name}: {e}")
            df_commits = pd.DataFrame()
            
        # 3. Extract Contributors
        try:
            df_contribs = analyze_contributors(local_path, contribs_file)
        except Exception as e:
            print(f"[-] Contributor extraction failed for {name}: {e}")
            df_contribs = pd.DataFrame()
            
        # 4. Extract Modifications
        try:
            extract_modifications(local_path, mods_file)
        except Exception as e:
            print(f"[-] Modifications extraction failed for {name}: {e}")
            
        # 5. Run universal quality metrics pipeline
        try:
            df_quality = run_quality_pipeline(local_path, quality_file)
            if not df_quality.empty:
                quality_dfs.append(df_quality)
        except Exception as e:
            print(f"[-] Quality metrics pipeline failed for {name}: {e}")
            
        # 6. Gather repository statistics
        try:
            df_lang = detect_languages(local_path, lang_profile_file)
            dominant_lang = df_lang.iloc[0]["language"] if not df_lang.empty else "unknown"
            
            age_days = 1
            if not df_commits.empty and "committer_date" in df_commits.columns:
                dates = pd.to_datetime(df_commits["committer_date"], errors="coerce", utc=True)
                latest = dates.max()
                oldest = dates.min()
                if pd.notna(latest) and pd.notna(oldest):
                    age_days = max((latest - oldest).days, 1)
            
            total_loc = 0
            if not df_quality.empty and "loc" in df_quality.columns:
                total_loc = int(df_quality["loc"].sum())
                
            total_commits = len(df_commits)
            total_contributors = len(df_contribs) if not df_contribs.empty else 0
            
            metadata_records.append({
                "repository_name": name,
                "repository_language": dominant_lang,
                "repository_size": total_loc,
                "repository_age": age_days,
                "total_commits": total_commits,
                "total_contributors": total_contributors
            })
            print(f"[+] Gathered stats for {name}: Lang={dominant_lang}, Size={total_loc} LOC, Age={age_days} days, Commits={total_commits}")
        except Exception as e:
            print(f"[-] Failed to compile metadata stats for {name}: {e}")
            
    # Save combined repository metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_file = os.path.join(DATA_DIR, "repositories_metadata.csv")
    metadata_df.to_csv(metadata_file, index=False)
    print(f"\n[+] Saved repositories metadata to {metadata_file}.")
    
    # Save consolidated processed/quality_metrics.csv
    if quality_dfs:
        merged_quality = pd.concat(quality_dfs, ignore_index=True)
        master_quality_file = os.path.join(PROCESSED_DIR, "quality_metrics.csv")
        merged_quality.to_csv(master_quality_file, index=False)
        print(f"[+] Consolidated master quality metrics written to {master_quality_file}. Total rows: {len(merged_quality)}")
    else:
        print("[-] Warning: No quality metrics extracted across repositories.")

if __name__ == "__main__":
    mine_all_repositories()
