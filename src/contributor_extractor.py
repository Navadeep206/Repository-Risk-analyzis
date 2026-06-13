#!/usr/bin/env python3
"""
Utility script to extract contributor contribution details and calculate repository "ownership" and "bus factor" indicators.
"""

import os
import argparse
import pandas as pd
from typing import Optional
from pydriller import Repository
from config import RAW_DIR, ensure_dirs_exist

def analyze_contributors(repo_path: str, output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Analyzes contributor statistics, sorting them by commit volume and calculating contribution percentages.
    
    Args:
        repo_path: Path to the local git repository.
        output_file: Optional path for the CSV output.
        
    Returns:
        A pandas DataFrame containing the contributor data, or None if no commits found.
    """
    if not os.path.exists(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    repo_name = os.path.basename(repo_path.rstrip("/"))
    if not output_file:
        output_file = os.path.join(RAW_DIR, f"{repo_name}_contributors.csv")
        
    print(f"[*] Extracting contributor metrics from: {repo_path}")
    
    contributor_stats = {}
    
    # Traverse commits
    total_commits = 0
    for commit in Repository(repo_path).traverse_commits():
        author_email = commit.author.email
        author_name = commit.author.name
        
        # Use email as key to handle developers using different names
        if author_email not in contributor_stats:
            contributor_stats[author_email] = {
                "name": author_name,
                "email": author_email,
                "commits": 0,
                "insertions": 0,
                "deletions": 0,
                "first_commit_date": commit.author_date,
                "last_commit_date": commit.author_date
            }
            
        stats = contributor_stats[author_email]
        stats["commits"] += 1
        stats["insertions"] += commit.insertions
        stats["deletions"] += commit.deletions
        
        # Keep track of dates
        if commit.author_date < stats["first_commit_date"]:
            stats["first_commit_date"] = commit.author_date
        if commit.author_date > stats["last_commit_date"]:
            stats["last_commit_date"] = commit.author_date
            
        total_commits += 1
        
    if total_commits == 0:
        print("[-] No commits found to analyze.")
        return None
        
    # Convert to DataFrame
    df = pd.DataFrame(contributor_stats.values())
    
    # Calculate percentages
    df["commit_share_pct"] = (df["commits"] / total_commits) * 100
    df["total_churn"] = df["insertions"] + df["deletions"]
    
    # Sort by commit count descending
    df = df.sort_values(by="commits", ascending=False).reset_index(drop=True)
    
    # Calculate cumulative percentage to help determine Bus Factor (number of developers representing e.g. 50%+ of commits)
    df["cumulative_share_pct"] = df["commit_share_pct"].cumsum()
    
    # Ensure raw directory exists
    ensure_dirs_exist()
    
    df.to_csv(output_file, index=False)
    print(f"[+] Successfully analyzed {len(df)} unique contributors.")
    
    # Print high-level summary of top 5 contributors
    print("-" * 60)
    print(f"{'Contributor':<25} | {'Commits':<8} | {'Share %':<8} | {'Cumulative %':<12}")
    print("-" * 60)
    for idx, row in df.head(5).iterrows():
        print(f"{row['name'][:24]:<25} | {row['commits']:<8} | {row['commit_share_pct']:<8.2f} | {row['cumulative_share_pct']:<12.2f}")
    print("-" * 60)
    
    # Simple Bus Factor estimate (developers responsible for >50% of the commits)
    bus_factor_df = df[df["cumulative_share_pct"] <= 50]
    bus_factor = len(bus_factor_df) + 1  # Add 1 for the threshold crossover
    # Cap bus factor at total contributors
    bus_factor = min(bus_factor, len(df))
    
    print(f"[+] Estimated Bus Factor (developers representing >50% commits): {bus_factor}")
    print(f"[+] Output saved to: {output_file}")
    
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze contributor statistics and key risk factors.")
    parser.add_argument("repo_path", help="Path to the local repository directory")
    parser.add_argument("--output", "-o", help="Target CSV file for the output (optional)", default=None)
    
    args = parser.parse_args()
    try:
        analyze_contributors(args.repo_path, args.output)
    except Exception as e:
        print(f"[-] Error: {e}")

if __name__ == "__main__":
    main()