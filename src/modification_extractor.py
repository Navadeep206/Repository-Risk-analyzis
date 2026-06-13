#!/usr/bin/env python3
"""
Utility script to extract file-level modifications (churn, path changes, and code complexity) from a repository.
"""

import os
import argparse
import pandas as pd
from typing import Optional
from pydriller import Repository
from config import RAW_DIR, ensure_dirs_exist

def extract_modifications(repo_path: str, output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Traverses commits, logging each modified file, lines added/deleted, type of change, and code complexity.
    
    Args:
        repo_path: Path to the local git repository.
        output_file: Optional path for the CSV output.
        
    Returns:
        A pandas DataFrame containing the modification records, or None if empty.
    """
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
        
    repo_name = os.path.basename(repo_path.rstrip("/"))
    if not output_file:
        output_file = os.path.join(RAW_DIR, f"{repo_name}_modifications.csv")
        
    print(f"[*] Extracting file modifications from: {repo_path}")
    print("[*] Parsing files (this may take a few moments for large histories)...")
    
    modifications = []
    
    # Traverse commits and inspect modified files
    for commit in Repository(repo_path).traverse_commits():
        for file in commit.modified_files:
            modifications.append({
                "commit_hash": commit.hash,
                "author_email": commit.author.email,
                "commit_date": commit.author_date,
                "old_path": file.old_path,
                "new_path": file.new_path,
                "change_type": file.change_type.name if file.change_type else "UNKNOWN",
                "added_lines": file.added_lines,
                "deleted_lines": file.deleted_lines,
                "net_lines": file.added_lines - file.deleted_lines,
                "complexity": file.complexity if file.complexity is not None else -1,
                "nloc": file.nloc if file.nloc is not None else -1
            })
            
    df = pd.DataFrame(modifications)
    
    if len(df) == 0:
        print("[-] No file modifications found.")
        return None
        
    # Ensure raw directory exists
    ensure_dirs_exist()
    
    df.to_csv(output_file, index=False)
    print(f"[+] Successfully extracted {len(df)} file modifications.")
    
    # Summarize top 5 files by modification frequency (churn hot spots)
    # Using new_path or old_path if new_path is None
    df["active_path"] = df["new_path"].fillna(df["old_path"])
    hot_spots = df.groupby("active_path").size().reset_index(name="modification_count")
    hot_spots = hot_spots.sort_values(by="modification_count", ascending=False).reset_index(drop=True)
    
    print("-" * 60)
    print(f"{'Hotspot File Path':<45} | {'Modifications':<12}")
    print("-" * 60)
    for idx, row in hot_spots.head(5).iterrows():
        print(f"{row['active_path'][:44]:<45} | {row['modification_count']:<12}")
    print("-" * 60)
    
    print(f"[+] Output saved to: {output_file}")
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract detailed file-level modifications and complexity.")
    parser.add_argument("repo_path", help="Path to the local repository directory")
    parser.add_argument("--output", "-o", help="Target CSV file for the output (optional)", default=None)
    
    args = parser.parse_args()
    try:
        extract_modifications(args.repo_path, args.output)
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main()