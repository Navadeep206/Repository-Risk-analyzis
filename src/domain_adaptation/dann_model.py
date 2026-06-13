#!/usr/bin/env python3
"""
Experiment 4: Domain-Adversarial Neural Network (DANN).
Uses a PyTorch architecture with a Gradient Reversal Layer (GRL) to predict
code risk labels while hiding the repository identity.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory and package directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import BASE_DIR
from ml.data_loader import LABEL_MAP
from evaluator import load_aligned_embeddings, get_loro_folds, compute_metrics

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Gradient Reversal Layer (GRL)
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANN(nn.Module):
    def __init__(self, input_dim: int = 776, latent_dim: int = 64, num_classes: int = 3, num_domains: int = 6):
        super(DANN, self).__init__()
        
        # 1. Feature Extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )
        
        # 2. Risk Label Predictor
        self.risk_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
        # 3. Domain Discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains)
        )
        
    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict risk label
        risk_output = self.risk_predictor(features)
        
        # Adversarial domain prediction
        reversed_features = GradReverse.apply(features, alpha)
        domain_output = self.domain_discriminator(reversed_features)
        
        return risk_output, domain_output

def train_dann_fold(df_train: pd.DataFrame, df_test: pd.DataFrame, features: list, repo_map: dict, epochs: int = 30, batch_size: int = 32) -> np.ndarray:
    """
    Trains a DANN model on a single LORO fold and returns test predictions.
    """
    # Align features (X has shape [N, 776])
    X_train_num = df_train[features].fillna(0).values
    X_train_emb = np.stack(df_train["embedding"].values)
    X_train = np.hstack([X_train_num, X_train_emb])
    
    X_test_num = df_test[features].fillna(0).values
    X_test_emb = np.stack(df_test["embedding"].values)
    X_test = np.hstack([X_test_num, X_test_emb])
    
    # Target scale features locally per domain for stability
    mean_tr = np.mean(X_train, axis=0)
    std_tr = np.std(X_train, axis=0) + 1e-8
    X_train = (X_train - mean_tr) / std_tr
    X_test = (X_test - mean_tr) / std_tr
    
    y_train = df_train["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
    d_train = df_train["repository_name"].map(repo_map).values
    d_test = df_test["repository_name"].map(repo_map).values
    
    # Load into PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    d_train_t = torch.LongTensor(d_train)
    
    X_test_t = torch.FloatTensor(X_test)
    d_test_t = torch.LongTensor(d_test)
    
    dataset = TensorDataset(X_train_t, y_train_t, d_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = DANN(input_dim=X_train.shape[1], latent_dim=64, num_classes=3, num_domains=len(repo_map))
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(epochs):
        # Progressively increase alpha trade-off parameter as in standard DANN paper
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        
        for batch_x, batch_y, batch_d in dataloader:
            optimizer.zero_grad()
            
            # Predict
            pred_risk, pred_domain = model(batch_x, alpha=alpha)
            
            # Loss computations
            loss_risk = criterion(pred_risk, batch_y)
            loss_domain = criterion(pred_domain, batch_d)
            
            # Combined Loss
            loss = loss_risk + 0.5 * loss_domain
            
            loss.backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(X_test_t)
        test_preds = torch.argmax(test_logits, dim=1).numpy()
        
    return test_preds

def run_dann_experiments():
    print("[*] Running Experiment 4: Domain-Adversarial Neural Network (DANN)...")
    reports_dir = os.path.join(BASE_DIR, "reports", "domain_adaptation")
    os.makedirs(reports_dir, exist_ok=True)
    
    df_merged = load_aligned_embeddings()
    repos = df_merged["repository_name"].dropna().unique().tolist()
    repo_map = {repo: i for i, repo in enumerate(repos)}
    
    features = [
        "loc", "complexity", "maintainability_index", "commit_count",
        "modification_count", "contributor_count", "commit_frequency", "repository_age_days"
    ]
    
    dann_f1s = []
    
    # Loop over folds
    for held_out, df_train, df_test in get_loro_folds(df_merged):
        print(f"[*] Training DANN on fold with held-out: {held_out}")
        y_test = df_test["historical_risk_label"].map(LABEL_MAP).fillna(0).astype(int).values
        
        preds = train_dann_fold(df_train, df_test, features, repo_map)
        metrics = compute_metrics(y_test, preds)
        dann_f1s.append(metrics["macro_f1"])
        
    avg_f1 = np.mean(dann_f1s)
    print(f"[+] DANN LORO Avg Macro F1: {avg_f1:.4f}")
    
    # Save results to CSV
    df_dann = pd.DataFrame([
        {"model": "DANN (Embeddings + Metrics)", "avg_loro_macro_f1": avg_f1}
    ])
    results_path = os.path.join(reports_dir, "dann_results.csv")
    df_dann.to_csv(results_path, index=False)
    print(f"[+] Saved DANN results to {results_path}")

if __name__ == "__main__":
    run_dann_experiments()
