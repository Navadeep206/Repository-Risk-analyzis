#!/usr/bin/env python3
import os
import pandas as pd

# Define paths
base_dir = "/Users/navadeepguduru/Repository mining /repository-risk-intelligence"
raw_dir = os.path.join(base_dir, "data/raw")
processed_dir = os.path.join(base_dir, "data/processed")
final_dataset_path = os.path.join(base_dir, "data/final/ml_dataset_v2.csv")

repositories = [
    "requests", "pytest", "click", "databases", "jinja", "fastapi", "express", "redux",
    "axios", "lodash", "svelte", "prisma", "localstack", "scikit-learn", "prefect",
    "great_expectations", "airflow", "django", "pandas", "ansible", "ray", "pytorch"
]

# Load final dataset to compute dataset_rows
if os.path.exists(final_dataset_path):
    final_df = pd.read_csv(final_dataset_path)
    dataset_counts = final_df.groupby("repository_name").size().to_dict()
else:
    print(f"Warning: final dataset not found at {final_dataset_path}")
    final_df = pd.DataFrame()
    dataset_counts = {}

ingestion_records = []
validation_records = []

for repo in repositories:
    # Commit count
    commits_file = os.path.join(raw_dir, f"{repo}_commits.csv")
    commit_count = len(pd.read_csv(commits_file)) if os.path.exists(commits_file) else 0
    
    # Contributor count
    contribs_file = os.path.join(raw_dir, f"{repo}_contributors.csv")
    contributors = len(pd.read_csv(contribs_file)) if os.path.exists(contribs_file) else 0
    
    # Modifications count
    mods_file = os.path.join(raw_dir, f"{repo}_modifications.csv")
    modifications = len(pd.read_csv(mods_file)) if os.path.exists(mods_file) else 0
    
    # Quality metrics count
    quality_file = os.path.join(processed_dir, f"{repo}_quality_metrics.csv")
    quality_rows = len(pd.read_csv(quality_file)) if os.path.exists(quality_file) else 0
    
    # Dataset rows
    dataset_rows = dataset_counts.get(repo, 0)
    
    status = "SUCCESS" if (commit_count > 0 and contributors > 0 and modifications > 0 and quality_rows > 0 and dataset_rows > 0) else "FAILED"
    
    ingestion_records.append({
        "repository_name": repo,
        "commit_count": commit_count,
        "contributors": contributors,
        "modifications": modifications,
        "quality_metric_rows": quality_rows,
        "dataset_rows": dataset_rows,
        "status": status
    })
    
    is_valid = dataset_rows > 0
    validation_records.append({
        "repository_name": repo,
        "dataset_rows": dataset_rows,
        "is_valid": is_valid
    })

# Create DataFrames
df_ingestion = pd.DataFrame(ingestion_records)
df_validation = pd.DataFrame(validation_records)

# Save to CSV in project root and data directory
paths_to_save = [
    base_dir,
    os.path.join(base_dir, "data")
]

for p in paths_to_save:
    df_ingestion.to_csv(os.path.join(p, "repository_ingestion_report.csv"), index=False)
    df_validation.to_csv(os.path.join(p, "repository_validation_report.csv"), index=False)
    print(f"[+] Saved reports to {p}/")

# Print summary stats for markdown report
print("\n--- Summary Statistics ---")
print(f"Total Repositories: {len(repositories)}")
print(f"Total Dataset Rows: {df_ingestion['dataset_rows'].sum()}")
print(f"Total Commits: {df_ingestion['commit_count'].sum()}")
print(f"Total Contributors: {df_ingestion['contributors'].sum()}")
print(f"Total Modifications: {df_ingestion['modifications'].sum()}")
print(f"Total Quality Rows: {df_ingestion['quality_metric_rows'].sum()}")
print("\nIngestion Breakdown:")
print(df_ingestion.to_string(index=False))
