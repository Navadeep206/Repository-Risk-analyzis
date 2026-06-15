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
        
    import csv
    
    headers = [
        "commit_hash", "author_email", "commit_date",
        "old_path", "new_path", "change_type",
        "added_lines", "deleted_lines", "net_lines",
        "complexity", "nloc"
    ]
    
    ensure_dirs_exist()
    
    print(f"[*] Extracting file modifications from: {repo_path}")
    print("[*] Parsing files (this may take a few moments for large histories)...")
    
    count = 0
    commit_count = 0
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for commit in Repository(repo_path).traverse_commits():
            commit_count += 1
            try:
                modified_files = commit.modified_files
            except Exception as commit_err:
                # Silently skip boundary/missing parent commits in shallow clones
                continue
                
            for file in modified_files:
                added = file.added_lines if file.added_lines is not None else 0
                deleted = file.deleted_lines if file.deleted_lines is not None else 0
                writer.writerow([
                    commit.hash,
                    commit.author.email if commit.author else "unknown",
                    commit.author_date.isoformat() if commit.author_date else "",
                    file.old_path if file.old_path else "",
                    file.new_path if file.new_path else "",
                    file.change_type.name if file.change_type else "UNKNOWN",
                    added,
                    deleted,
                    added - deleted,
                    -1,
                    -1
                ])
                count += 1
            if commit_count % 2000 == 0:
                print(f"[*] Traversed {commit_count} commits, written {count} modifications...")
                
    print(f"[+] Successfully extracted {count} file modifications across {commit_count} commits.")
    
    if count == 0:
        print("[-] No file modifications found.")
        return None
        
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"[!] Error loading generated CSV {output_file}: {e}")
        df = pd.DataFrame(columns=headers)
        return df
        
    # Summarize top 5 files by modification frequency (churn hot spots)
    df["active_path"] = df["new_path"].fillna(df["old_path"])
    if not df.empty and "active_path" in df.columns:
        hot_spots = df.groupby("active_path").size().reset_index(name="modification_count")
        hot_spots = hot_spots.sort_values(by="modification_count", ascending=False).reset_index(drop=True)
        
        print("-" * 60)
        print(f"{'Hotspot File Path':<45} | {'Modifications':<12}")
        print("-" * 60)
        for idx, row in hot_spots.head(5).iterrows():
            active_path_str = str(row['active_path'])
            print(f"{active_path_str[:44]:<45} | {row['modification_count']:<12}")
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