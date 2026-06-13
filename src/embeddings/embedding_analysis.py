#!/usr/bin/env python3
"""
Embedding analysis module for Phase 5.
Analyzes vector statistics (norms, dimensions, counts) and writes reports/embedding_statistics.md.
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

def analyze_embeddings() -> None:
    """
    Computes mathematical statistics on CodeBERT embeddings
    and generates reports/embedding_statistics.md.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    npy_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    metadata_csv_path = os.path.join(EMBEDDINGS_DIR, "embedding_metadata.csv")
    
    if not os.path.exists(npy_path) or not os.path.exists(metadata_csv_path):
        raise FileNotFoundError("Required embeddings array or metadata file is missing. Generate them first.")
        
    df_meta = pd.read_csv(metadata_csv_path)
    embeddings = np.load(npy_path, mmap_mode="r")
    
    # Calculate statistics
    num_embeddings = len(embeddings)
    dim_size = embeddings.shape[1] if num_embeddings > 0 else 0
    
    # Calculate norms
    norms = np.linalg.norm(embeddings, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    min_norm = np.min(norms)
    max_norm = np.max(norms)
    
    # Calculate size on disk
    file_size_mb = os.path.getsize(npy_path) / (1024 * 1024)
    
    # Count per repository & language
    repo_counts = df_meta["repository_name"].value_counts()
    lang_counts = df_meta["language"].value_counts()
    
    # Write report
    report_path = os.path.join(REPORTS_DIR, "embedding_statistics.md")
    
    with open(report_path, "w") as f:
        f.write("# Embedding Analysis Report - Phase 5 CodeBERT\n\n")
        f.write("This report provides a statistical breakdown and structural audit of the generated CodeBERT embeddings.\n\n")
        
        # 1. High-level info
        f.write("## 1. Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| **Total Files Embedded** | {num_embeddings} |\n")
        f.write(f"| **Embedding Dimension** | {dim_size} (Expected: 768) |\n")
        f.write(f"| **NumPy Array Disk Size** | {file_size_mb:.2f} MB |\n")
        f.write(f"| **Mean L2 Norm** | {mean_norm:.4f} |\n")
        f.write(f"| **L2 Norm Std Dev** | {std_norm:.4f} |\n")
        f.write(f"| **Min L2 Norm** | {min_norm:.4f} |\n")
        f.write(f"| **Max L2 Norm** | {max_norm:.4f} |\n\n")
        
        f.write("---\n\n")
        
        # 2. Counts per repository
        f.write("## 2. Repository Distribution\n\n")
        f.write("| Repository | Count | Percentage |\n")
        f.write("|------------|-------|------------|\n")
        for repo, count in repo_counts.items():
            pct = (count / num_embeddings) * 100
            f.write(f"| {repo} | {count} | {pct:.2f}% |\n")
            
        f.write("\n---\n\n")
        
        # 3. Counts per language
        f.write("## 3. Language Distribution\n\n")
        f.write("| Language | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        for lang, count in lang_counts.items():
            pct = (count / num_embeddings) * 100
            f.write(f"| {lang} | {count} | {pct:.2f}% |\n")
            
        f.write("\n---\n\n")
        
        # 4. Verification Check
        f.write("## 4. Quality Audit & Verification Check\n\n")
        is_nan = np.isnan(embeddings).any()
        is_inf = np.isinf(embeddings).any()
        
        f.write("- **Null Values Check**: " + ("❌ FAILED (NaN values detected)" if is_nan else "✅ PASSED (No NaN values)") + "\n")
        f.write("- **Infinite Values Check**: " + ("❌ FAILED (Infinite values detected)" if is_inf else "✅ PASSED (No infinite values)") + "\n")
        f.write("- **Dimensionality Audit**: " + (f"✅ PASSED (Confirmed {dim_size}-D)" if dim_size == 768 else f"❌ FAILED (Found {dim_size}-D, expected 768)") + "\n")
        
    print(f"[+] Embedding analysis complete. Saved report to {report_path}")

if __name__ == "__main__":
    analyze_embeddings()
