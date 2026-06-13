#!/usr/bin/env python3
"""
Dataset loader for the deep learning pipeline.
Partitions embedding_dataset.parquet into disjoint repository-aware splits and creates PyTorch DataLoaders.
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
from config import FINAL_DIR

LABEL_MAP: Dict[str, int] = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2
}

INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

class EmbeddingPyTorchDataset(Dataset):
    """
    Custom PyTorch Dataset that yields 768-D embedding vectors and target integer labels.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

def get_dataloaders(batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Loads embedding_dataset.parquet, partitions into repository-disjoint splits,
    and returns DataLoader instances along with PyTorch class weights for loss balancing.
    """
    parquet_path = os.path.join(FINAL_DIR, "embedding_dataset.parquet")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Embedding dataset parquet file not found: {parquet_path}. Run Phase 5 first.")
        
    df = pd.read_parquet(parquet_path)
    
    # Map text labels to integers
    df["label"] = df["historical_risk_label"].map(LABEL_MAP)
    
    # Define repository-disjoint splits to match Phase 4 exactly
    train_repos = ["axios", "redux", "click"]
    val_repos = ["express"]
    test_repos = ["databases", "jinja"]
    
    df_train = df[df["repository_name"].isin(train_repos)].copy()
    df_val = df[df["repository_name"].isin(val_repos)].copy()
    df_test = df[df["repository_name"].isin(test_repos)].copy()
    
    print(f"[+] Loaded splits -> Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Extract feature matrices and labels
    X_train = np.stack(df_train["embedding"].values).astype(np.float32)
    y_train = df_train["label"].values
    
    X_val = np.stack(df_val["embedding"].values).astype(np.float32)
    y_val = df_val["label"].values
    
    X_test = np.stack(df_test["embedding"].values).astype(np.float32)
    y_test = df_test["label"].values
    
    # Calculate class weights from training split to balance CrossEntropyLoss
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"[+] Computed Training Class Weights: {weights} for classes {classes}")
    
    # Create PyTorch datasets
    train_dataset = EmbeddingPyTorchDataset(X_train, y_train)
    val_dataset = EmbeddingPyTorchDataset(X_val, y_val)
    test_dataset = EmbeddingPyTorchDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, class_weights_tensor

if __name__ == "__main__":
    tr_load, va_load, te_load, weights = get_dataloaders()
    print("[+] Dataset loader test complete.")
    for batch_x, batch_y in tr_load:
        print(f"Batch X shape: {batch_x.shape}, Batch y shape: {batch_y.shape}")
        break
