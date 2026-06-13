#!/usr/bin/env python3
"""
Utility script to extract high-level statistics from a git repository.
"""

import os
import argparse
import pandas as pd
from typing import Dict, Any, Optional
from pydriller import Repository
from config import RAW_DIR, ensure_dirs_exist

def analyze_repository(repo_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes high-level statistics of a git repository and writes them to a CSV file.
    
    Args:
        repo_path: Absolute path to the git repository.
        output_file: Optional path for the CSV output.
        
    Returns:
        A dictionary containing the calculated statistics.
    """
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
        
    repo_name = os.path.basename(repo_path.rstrip("/"))
    if not output_file:
        output_file = os.path.join(RAW_DIR, f"{repo_name}_repository_stats.csv")

    ensure_dirs_exist()
    print(f"[*] Analyzing repository at: {repo_path}")
    
    total_commits = 0
    contributors = set()
    modified_files = set()
    
    # Traverse commits using PyDriller to calculate stats
    for commit in Repository(repo_path).traverse_commits():
        total_commits += 1
        if commit.author.email:
            contributors.add(commit.author.email)
        for m_file in commit.modified_files:
            active_path = m_file.new_path or m_file.old_path
            if active_path:
                modified_files.add(active_path)
                
    total_contributors = len(contributors)
    total_modified_files = len(modified_files)
    
    # Print results
    print("-" * 40)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 40)
    print(f"{'Total Commits':<25} | {total_commits:<15}")
    print(f"{'Unique Contributors':<25} | {total_contributors:<15}")
    print(f"{'Total Modified Files':<25} | {total_modified_files:<15}")
    print("-" * 40)
    
    # Prepare CSV columns
    stats_data = {
        "total_commits": [total_commits],
        "total_contributors": [total_contributors],
        "total_modified_files": [total_modified_files],
        "repository_name": [repo_name]
    }
    
    df = pd.DataFrame(stats_data)
    df.to_csv(output_file, index=False)
    print(f"[+] Output saved to: {output_file}")
    
    return {
        "total_commits": total_commits,
        "total_contributors": total_contributors,
        "total_modified_files": total_modified_files,
        "repository_name": repo_name
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze high-level statistics of a git repository.")
    parser.add_argument("repo_path", help="Path to the local repository directory")
    parser.add_argument("--output", "-o", help="Target CSV file for the output (optional)", default=None)
    args = parser.parse_args()
    
    try:
        analyze_repository(args.repo_path, args.output)
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main()