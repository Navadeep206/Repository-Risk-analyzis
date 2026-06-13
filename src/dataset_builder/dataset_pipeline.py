#!/usr/bin/env python3
"""
Dataset construction pipeline orchestrator for Phase 3.5.
Executes multi-repo mining, merging, feature engineering, label generation,
data cleaning, and repository-safe group stratified splitting in sequence.
"""

import os
import argparse
import sys

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import ensure_dirs_exist

from dataset_builder.merge_repository_data import merge_repository_data
from dataset_builder.feature_engineering import engineer_features
from dataset_builder.label_generation import generate_labels
from dataset_builder.data_cleaning import clean_data
from dataset_builder.split_dataset import split_dataset

def run_dataset_pipeline(skip_mining: bool = False) -> None:
    """
    Runs the dataset building pipeline end-to-end.
    """
    print("[*] Starting Dataset Construction Pipeline (Phase 3.5)...")
    ensure_dirs_exist()
    
    # 0. Mine Repositories (if not skipped)
    if not skip_mining:
        print("\n=== STEP 0: Mining Multiple Repositories ===")
        from dataset_builder.multi_repository_miner import mine_all_repositories
        mine_all_repositories()
    else:
        print("\n=== STEP 0: Skipping Repository Mining (Using existing datasets) ===")
        
    # 1. Merge
    print("\n=== STEP 1: Merging Data Sources ===")
    merge_repository_data()
    
    # 2. Feature Engineering
    print("\n=== STEP 2: Feature Engineering ===")
    engineer_features()
    
    # 3. Label Generation (Target Leakage Free)
    print("\n=== STEP 3: Label Generation ===")
    generate_labels()
    
    # 4. Data Cleaning
    print("\n=== STEP 4: Data Cleaning ===")
    clean_data()
    
    # 5 & 6. Group Stratified Splitting and Final Dataset Creation
    print("\n=== STEP 5 & 6: Splitting and Creating Master v2 Dataset ===")
    split_dataset()
    
    print("\n[+] Dataset Construction Pipeline v2 completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Phase 3.5 dataset construction pipeline.")
    parser.add_argument("--skip-mining", action="store_true", help="Skip mining repositories and use existing CSVs")
    args = parser.parse_args()
    
    run_dataset_pipeline(skip_mining=args.skip_mining)
