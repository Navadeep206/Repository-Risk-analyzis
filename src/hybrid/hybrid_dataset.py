#!/usr/bin/env python3
"""
Hybrid dataset loader.
Merges tabular features and CodeBERT embeddings, scales tabular inputs,
and partitions them into repository-disjoint splits.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List
from sklearn.utils.class_weight import compute_class_weight

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR, FINAL_DIR
from ml.preprocessing import CodeRiskPreprocessor

LABEL_MAP: Dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}

INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

class HybridPyTorchDataset(Dataset):
    """
    Custom PyTorch Dataset that yields tabular features (11-D),
    embedding vectors (768-D), and target integer labels.
    """
    def __init__(self, X_tabular: np.ndarray, X_embedding: np.ndarray, y: np.ndarray) -> None:
        self.X_tabular = torch.tensor(X_tabular, dtype=torch.float32)
        self.X_embedding = torch.tensor(X_embedding, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_tabular[idx], self.X_embedding[idx], self.y[idx]

def get_hybrid_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads ml_dataset_v2.csv and embeddings.parquet, merges them, 
    applies preprocessing on tabular features, and maps labels.
    
    Returns:
        df_merged: Merged DataFrame containing metadata.
        X_tabular: Scaled and encoded tabular feature matrix.
        X_embedding: CodeBERT embedding matrix.
        y: Integer class labels.
    """
    csv_path = os.path.join(FINAL_DIR, "ml_dataset_v2.csv")
    parquet_path = os.path.join(BASE_DIR, "data", "embeddings", "embeddings.parquet")
    preproc_path = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ML dataset v2 not found: {csv_path}")
    if not os.path.exists(parquet_path):
        # Fallback to final parquet path if not in embeddings dir
        parquet_path = os.path.join(FINAL_DIR, "embedding_dataset.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError("Embedding parquet not found.")

    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor pkl not found: {preproc_path}")

    # Load datasets
    df_csv = pd.read_csv(csv_path)
    df_parquet = pd.read_parquet(parquet_path)

    # Clean parquet column names if needed
    if "embedding" not in df_parquet.columns:
        raise ValueError("Parquet file does not contain 'embedding' column.")

    # Merge on repository_name and file_path
    df_merged = pd.merge(df_csv, df_parquet, on=["repository_name", "file_path"], suffixes=("", "_parquet"))
    
    # In case language columns diverged, we keep standard columns
    if "language_parquet" in df_merged.columns:
        df_merged = df_merged.drop(columns=["language_parquet"])

    # Load and run preprocessor on tabular features
    preprocessor = CodeRiskPreprocessor.load(preproc_path)
    X_tabular = preprocessor.transform(df_merged)
    
    # Re-verify feature size (must be 11)
    if X_tabular.shape[1] != 11:
        raise ValueError(f"Tabular features dimension mismatch: expected 11, got {X_tabular.shape[1]}")

    # Stack embeddings
    X_embedding = np.stack(df_merged["embedding"].values).astype(np.float32)
    
    # Map target labels
    y = df_merged["historical_risk_label"].map(LABEL_MAP).values
    
    return df_merged, X_tabular, X_embedding, y

def get_hybrid_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Splits the merged hybrid dataset into train/validation/test sets
    according to disjoint repository mapping and returns PyTorch DataLoader instances.
    """
    df_merged, X_tabular, X_embedding, y = get_hybrid_data()
    
    # Define repository-disjoint splits
    train_repos = ["axios", "redux", "click"]
    val_repos = ["express"]
    test_repos = ["databases", "jinja"]
    
    # Train Split indices
    train_mask = df_merged["repository_name"].isin(train_repos).values
    val_mask = df_merged["repository_name"].isin(val_repos).values
    test_mask = df_merged["repository_name"].isin(test_repos).values

    print(f"[+] Splits count -> Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    # Separate matrices
    X_tab_train, X_emb_train, y_train = X_tabular[train_mask], X_embedding[train_mask], y[train_mask]
    X_tab_val, X_emb_val, y_val = X_tabular[val_mask], X_embedding[val_mask], y[val_mask]
    X_tab_test, X_emb_test, y_test = X_tabular[test_mask], X_embedding[test_mask], y[test_mask]

    # Calculate class weights on train split to prevent imbalance bias
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"[+] Computed Training Class Weights: {weights} for classes {classes}")

    # Datasets
    train_dataset = HybridPyTorchDataset(X_tab_train, X_emb_train, y_train)
    val_dataset = HybridPyTorchDataset(X_tab_val, X_emb_val, y_val)
    test_dataset = HybridPyTorchDataset(X_tab_test, X_emb_test, y_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights_tensor

if __name__ == "__main__":
    tr, va, te, w = get_hybrid_dataloaders()
    for xt, xe, yl in tr:
        print(f"Tabular shape: {xt.shape}, Embedding shape: {xe.shape}, Label shape: {yl.shape}")
        break
