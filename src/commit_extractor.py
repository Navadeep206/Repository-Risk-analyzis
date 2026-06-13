#!/usr/bin/env python3
"""
Utility script to extract commit history from a git repository and save it to a CSV file under data/raw/.
"""

import os
import argparse
import pandas as pd
from typing import Optional
from pydriller import Repository
from config import RAW_DIR, ensure_dirs_exist

def extract_commits(repo_path: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Extracts commit history from a local git repository and exports to a CSV.
    
    Args:
        repo_path: Path to the local git repository.
        output_file: Optional path for the CSV output.
        
    Returns:
        A pandas DataFrame containing the extracted commit data.
    """
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    repo_name = os.path.basename(repo_path.rstrip("/"))
    if not output_file:
        output_file = os.path.join(RAW_DIR, f"{repo_name}_commits.csv")
    
    print(f"[*] Extracting commits from: {repo_path}")
    print("[*] Processing history (this may take a few moments for large repositories)...")
    
    commit_data = []
    
    # Traverse commits using PyDriller
    for commit in Repository(repo_path).traverse_commits():
        commit_data.append({
            "commit_hash": commit.hash,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "author_date": commit.author_date,
            "committer_name": commit.committer.name,
            "committer_email": commit.committer.email,
            "committer_date": commit.committer_date,
            "message": commit.msg.strip(),
            "branches": ", ".join(commit.branches),
            "is_merge": commit.merge,
            "files_changed": commit.files,
            "insertions": commit.insertions,
            "deletions": commit.deletions,
            "lines_changed": commit.lines
        })
        
    df = pd.DataFrame(commit_data)
    
    # Ensure raw directory exists
    ensure_dirs_exist()
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"[+] Successfully extracted {len(df)} commits.")
    print(f"[+] Output saved to: {output_file}")
    
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract commit history from a git repository.")
    parser.add_argument("repo_path", help="Path to the local repository directory")
    parser.add_argument("--output", "-o", help="Target CSV file for the output (optional)", default=None)
    
    args = parser.parse_args()
    try:
        extract_commits(args.repo_path, args.output)
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main()