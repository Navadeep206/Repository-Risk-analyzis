#!/usr/bin/env python3
"""
Model training module for Phase 7.
Handles neural network training, validation monitoring, early stopping, and metric logging.
Supports Tabular, Embedding, and Hybrid architectures.
"""

import os
import sys

# MUST be set before importing torch, sklearn or any OpenMP-linked library
# Prevents EXC_BAD_ACCESS (SIGSEGV) from duplicate libomp on macOS ARM64
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import copy
import torch
import torch.nn as nn
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import f1_score

# Ensure parent directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import BASE_DIR

def train_hybrid_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    class_weights: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.0005,
    patience: int = 15,
    device: torch.device = torch.device("cpu")
) -> List[Dict[str, Any]]:
    """
    Trains the given PyTorch module using AdamW, weighted CrossEntropyLoss, and early stopping.
    Handles the multi-input forward signature: model(X_tab, X_emb).
    """
    model = model.to(device)
    # Use AdamW optimizer with learning rate 0.0005 and small weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    history: List[Dict[str, Any]] = []
    
    print(f"[*] Training model on device: {device}")
    
    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_tab_batch, X_emb_batch, y_batch in train_loader:
            X_tab_batch = X_tab_batch.to(device)
            X_emb_batch = X_emb_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_tab_batch, X_emb_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * y_batch.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_tab_batch, X_emb_batch, y_batch in val_loader:
                X_tab_batch = X_tab_batch.to(device)
                X_emb_batch = X_emb_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_tab_batch, X_emb_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * y_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
                
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "validation_loss": val_loss,
            "validation_accuracy": val_acc,
            "validation_f1_macro": val_f1
        })
        
        # 3. Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch+1}. Restoring best checkpoint.")
                break
                
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return history
