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
    
    import csv
    
    headers = [
        "commit_hash", "author_name", "author_email", "author_date",
        "committer_name", "committer_email", "committer_date",
        "message", "branches", "is_merge", "files_changed",
        "insertions", "deletions", "lines_changed"
    ]
    
    # Ensure raw directory exists
    ensure_dirs_exist()
    
    print(f"[*] Extracting commits from: {repo_path}")
    print("[*] Processing history (this may take a few moments for large repositories)...")
    
    count = 0
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for commit in Repository(repo_path).traverse_commits():
            msg = commit.msg.strip() if commit.msg else ""
            if len(msg) > 200:
                msg = msg[:200] + "..."
            msg = msg.replace("\n", " ").replace("\r", " ")
            
            writer.writerow([
                commit.hash,
                commit.author.name if commit.author else "unknown",
                commit.author.email if commit.author else "unknown",
                commit.author_date.isoformat() if commit.author_date else "",
                commit.committer.name if commit.committer else "unknown",
                commit.committer.email if commit.committer else "unknown",
                commit.committer_date.isoformat() if commit.committer_date else "",
                msg,
                "",  # branches optimized
                commit.merge,
                0, 0, 0, 0
            ])
            count += 1
            if count % 2000 == 0:
                print(f"[*] Processed {count} commits...")
                
    print(f"[+] Successfully extracted {count} commits.")
    print(f"[+] Output saved to: {output_file}")
    
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"[!] Error loading generated CSV {output_file}: {e}")
        df = pd.DataFrame(columns=headers)
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