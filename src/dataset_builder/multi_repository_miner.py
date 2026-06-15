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

# =============================================================================
# PHASE 1 — Starter repositories (real-data verified, fast to mine)
# Expand to PHASE_2_REPOSITORIES after Phase 1 validation succeeds.
# =============================================================================
TARGET_REPOSITORIES = [
    {"name": "requests",  "url": "https://github.com/psf/requests",        "lang": "python"},
    {"name": "pytest",    "url": "https://github.com/pytest-dev/pytest",    "lang": "python"},
    {"name": "click",     "url": "https://github.com/pallets/click",        "lang": "python"},
    {"name": "databases", "url": "https://github.com/encode/databases",     "lang": "python"},
]

# =============================================================================
# PHASE 2 — Batch 1: Small/medium repos (uncomment after Phase 1 verified)
# =============================================================================
PHASE_2_BATCH_1 = [
    {"name": "jinja",     "url": "https://github.com/pallets/jinja",       "lang": "python"},
    {"name": "fastapi",   "url": "https://github.com/tiangolo/fastapi",    "lang": "python"},
    {"name": "express",   "url": "https://github.com/expressjs/express",   "lang": "javascript"},
    {"name": "redux",     "url": "https://github.com/reduxjs/redux",       "lang": "typescript"},
]

# =============================================================================
# PHASE 2 — Batch 2: Medium repos
# =============================================================================
PHASE_2_BATCH_2 = [
    {"name": "axios",       "url": "https://github.com/axios/axios",           "lang": "typescript"},
    {"name": "lodash",      "url": "https://github.com/lodash/lodash",         "lang": "javascript"},
    {"name": "svelte",      "url": "https://github.com/sveltejs/svelte",       "lang": "typescript"},
    {"name": "prisma",      "url": "https://github.com/prisma/prisma",         "lang": "typescript"},
]

# =============================================================================
# PHASE 2 — Batch 3: Larger repos (depth-limit recommended)
# =============================================================================
PHASE_2_BATCH_3 = [
    {"name": "localstack",        "url": "https://github.com/localstack/localstack",           "lang": "python"},
    {"name": "scikit-learn",      "url": "https://github.com/scikit-learn/scikit-learn",       "lang": "python"},
    {"name": "prefect",           "url": "https://github.com/PrefectHQ/prefect",               "lang": "python"},
    {"name": "great_expectations","url": "https://github.com/great-expectations/great_expectations","lang": "python"},
]

# =============================================================================
# PHASE 2 — Batch 4: Large repos (clone with --depth=200)
# =============================================================================
PHASE_2_BATCH_4 = [
    {"name": "airflow",       "url": "https://github.com/apache/airflow",           "lang": "python"},
    {"name": "django",        "url": "https://github.com/django/django",            "lang": "python"},
    {"name": "pandas",        "url": "https://github.com/pandas-dev/pandas",        "lang": "python"},
    {"name": "ray",           "url": "https://github.com/ray-project/ray",          "lang": "python"},
    # {"name": "elasticsearch", "url": "https://github.com/elastic/elasticsearch",    "lang": "java"},
]


# =============================================================================
# PHASE 2 — Batch 5: Very large (clone with --depth=100, file limits apply)
# =============================================================================
PHASE_2_BATCH_5 = [
    {"name": "pytorch", "url": "https://github.com/pytorch/pytorch",   "lang": "python"},
    {"name": "ansible", "url": "https://github.com/ansible/ansible",   "lang": "python"},
]

# Combine all phases and batches into the final target list of 22 repositories (excluding elasticsearch)
TARGET_REPOSITORIES = (
    TARGET_REPOSITORIES +
    PHASE_2_BATCH_1 +
    PHASE_2_BATCH_2 +
    PHASE_2_BATCH_3 +
    PHASE_2_BATCH_4 +
    PHASE_2_BATCH_5
)

# Apply configuration details for heavy repositories to prevent memory exhaustion
for repo in TARGET_REPOSITORIES:
    if repo["name"] == "pytorch":
        repo["depth"] = 200
    elif repo["name"] == "ray":
        repo["depth"] = 500

# Reorder TARGET_REPOSITORIES so that heavy repositories (pytorch, ray) run last
heavy_repos = ["ray", "pytorch"]
TARGET_REPOSITORIES = [r for r in TARGET_REPOSITORIES if r["name"] not in heavy_repos] + \
                      [r for r in TARGET_REPOSITORIES if r["name"] in heavy_repos]




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
                clone_kwargs = {}
                if "depth" in repo_info:
                    clone_kwargs["depth"] = repo_info["depth"]
                    print(f"[*] Using shallow clone with depth={repo_info['depth']} for {name}")
                Repo.clone_from(url, local_path, **clone_kwargs)
                print(f"[+] Cloned successfully.")
            except Exception as e:
                print(f"[-] Failed to clone {name}: {e}. Skipping repository.")
                continue
        else:
            print(f"[*] Repository {name} already cloned at {local_path}.")

        # 1b. Ensure full history is available (convert shallow clone if needed)
        try:
            repo_obj = Repo(local_path)
            if repo_obj.git.rev_parse("--is-shallow-repository").strip() == "true":
                if "depth" in repo_info:
                    print(f"[*] Keeping {name} as a shallow repository as configured (depth={repo_info['depth']}).")
                else:
                    print(f"[*] {name} is a shallow clone. Fetching full history (unshallow)...")
                    repo_obj.git.fetch("--unshallow", "--quiet")
                    print(f"[+] Unshallow complete for {name}.")
        except Exception as e:
            print(f"[!] Unshallow check skipped for {name}: {e}")
            
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
            # Write empty CSV with correct headers so merge step can still include this repo
            pd.DataFrame(columns=[
                "commit_hash", "author_email", "commit_date",
                "old_path", "new_path", "change_type",
                "added_lines", "deleted_lines", "net_lines",
                "complexity", "nloc"
            ]).to_csv(mods_file, index=False)
            print(f"[!] Empty modifications file written for {name} to allow downstream merge.")
            
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
