#!/usr/bin/env python3
"""
Step 9: Visualization compilation for Phase 12.
Renders and saves all required plots:
- repository_shift_heatmap.png
- embedding_drift_heatmap.png (regenerated/unified)
- coral_alignment.png
- loro_comparison.png
- adaptation_performance.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Prevent GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory and package directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import BASE_DIR
from evaluator import load_master_dataset, load_aligned_embeddings
from coral_alignment import coral_align

def generate_visualizations():
    print("[*] Generating Phase 12 plots...")
    reports_dir = os.path.join(BASE_DIR, "reports", "domain_adaptation")
    plots_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load raw data
    df = load_master_dataset()
    repos = df["repository_name"].dropna().unique().tolist()
    features = [
        "loc", "complexity", "maintainability_index", "commit_count",
        "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
    ]
    
    # 1. Repository Shift Heatmap (Tabular Features Cosine Similarity)
    centroids = {}
    for repo in repos:
        df_repo = df[df["repository_name"] == repo]
        scaled_features = StandardScaler().fit_transform(df_repo[features].fillna(0).values)
        centroids[repo] = np.mean(scaled_features, axis=0)
        
    shift_matrix = np.zeros((len(repos), len(repos)))
    for i, r_a in enumerate(repos):
        for j, r_b in enumerate(repos):
            c_a = centroids[r_a]
            c_b = centroids[r_b]
            norm_a = np.linalg.norm(c_a)
            norm_b = np.linalg.norm(c_b)
            similarity = np.dot(c_a, c_b) / (norm_a * norm_b + 1e-8)
            shift_matrix[i, j] = similarity
            
    df_shift = pd.DataFrame(shift_matrix, index=repos, columns=repos)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_shift, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Repository Tabular Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "repository_shift_heatmap.png"), dpi=150)
    plt.close()
    
    # 2. Embedding Drift Heatmap (Already handled in analysis but verify/regenerate here to ensure complete plots/)
    df_emb = load_aligned_embeddings()
    emb_centroids = {repo: np.mean(np.stack(df_emb[df_emb["repository_name"] == repo]["embedding"].values), axis=0) for repo in repos}
    emb_matrix = np.zeros((len(repos), len(repos)))
    for i, r_a in enumerate(repos):
        for j, r_b in enumerate(repos):
            c_a = emb_centroids[r_a]
            c_b = emb_centroids[r_b]
            norm_a = np.linalg.norm(c_a)
            norm_b = np.linalg.norm(c_b)
            emb_matrix[i, j] = np.dot(c_a, c_b) / (norm_a * norm_b + 1e-8)
            
    df_emb_sim = pd.DataFrame(emb_matrix, index=repos, columns=repos)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_emb_sim, annot=True, cmap="coolwarm", fmt=".3f", vmin=0.5, vmax=1.0)
    plt.title("Centroid Cosine Similarity Heatmap (CodeBERT Embeddings)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "embedding_drift_heatmap.png"), dpi=150)
    plt.close()
    
    # 3. CORAL Alignment PCA Plot (Demonstrating alignment before vs after for click -> express)
    source_repo = "click"
    target_repo = "express"
    
    X_s = df[df["repository_name"] == source_repo][features].fillna(0).values
    X_t = df[df["repository_name"] == target_repo][features].fillna(0).values
    
    # Apply CORAL
    X_s_coral, X_t_scaled = coral_align(X_s, X_t)
    X_s_scaled = StandardScaler().fit_transform(X_s)  # scaled but not aligned
    
    # PCA to 2D
    pca_before = PCA(n_components=2)
    X_s_2d_before = pca_before.fit_transform(X_s_scaled)
    X_t_2d_before = pca_before.transform(X_t_scaled)
    
    pca_after = PCA(n_components=2)
    X_s_2d_after = pca_after.fit_transform(X_s_coral)
    X_t_2d_after = pca_after.transform(X_t_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before CORAL
    ax1.scatter(X_s_2d_before[:, 0], X_s_2d_before[:, 1], color="red", alpha=0.5, label=f"Source ({source_repo})")
    ax1.scatter(X_t_2d_before[:, 0], X_t_2d_before[:, 1], color="blue", alpha=0.5, label=f"Target ({target_repo})")
    ax1.set_title("Before CORAL Covariance Alignment")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # After CORAL
    ax2.scatter(X_s_2d_after[:, 0], X_s_2d_after[:, 1], color="green", alpha=0.5, label=f"Source ({source_repo}) Aligned")
    ax2.scatter(X_t_2d_after[:, 0], X_t_2d_after[:, 1], color="blue", alpha=0.5, label=f"Target ({target_repo})")
    ax2.set_title("After CORAL Covariance Alignment")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    coral_plot_path = os.path.join(plots_dir, "coral_alignment.png")
    plt.savefig(coral_plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved CORAL alignment PCA plots to {coral_plot_path}")
    
    # 4. LORO Comparison Charts (From benchmark results)
    bench_path = os.path.join(reports_dir, "robustness_benchmark.csv")
    if os.path.exists(bench_path):
        df_bench = pd.read_csv(bench_path)
        
        plt.figure(figsize=(10, 6))
        # Plot Macro F1
        sns.barplot(data=df_bench, x="model_name", y="avg_loro_macro_f1", hue="model_name", palette="viridis", legend=False)
        plt.axhline(y=0.5277, color="red", linestyle="--", label="Baseline RF (0.5277)")
        plt.title("LORO Average Macro F1 Comparison (Phase 12 Domain Adaptation)")
        plt.ylabel("Average LORO Macro F1")
        plt.xlabel("Model Configuration")
        plt.ylim(0, 0.7)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loro_comparison.png"), dpi=150)
        plt.close()
        
        # 5. Adaptation Performance Charts (F1 and Accuracy side by side)
        plt.figure(figsize=(10, 6))
        df_melt = df_bench.melt(id_vars=["model_name"], value_vars=["avg_loro_accuracy", "avg_loro_macro_f1"],
                                var_name="metric", value_name="score")
        
        sns.barplot(data=df_melt, x="model_name", y="score", hue="metric", palette="Set2")
        plt.title("Model Accuracy vs. Macro F1 across Adaptation Configurations")
        plt.ylabel("Score")
        plt.xlabel("Model")
        plt.ylim(0, 0.8)
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "adaptation_performance.png"), dpi=150)
        plt.close()
        print("[+] Phase 12 plots compiled successfully.")
    else:
        print("[!] Robustness benchmark CSV not found. Skip charts.")

if __name__ == "__main__":
    generate_visualizations()
