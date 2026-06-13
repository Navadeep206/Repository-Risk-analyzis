#!/usr/bin/env python3
"""
Fusion and Ablation models for Phase 7.
Contains:
- TabularOnlyPredictor (Model A)
- EmbeddingOnlyPredictor (Model B, identical to Phase 6 MLP)
- HybridRiskPredictor (Model C, multi-branch fusion network)
"""

import torch
import torch.nn as nn
from hybrid.tabular_encoder import TabularEncoder
from hybrid.embedding_encoder import EmbeddingEncoder

class TabularOnlyPredictor(nn.Module):
    """
    Model A: Tabular Only Predictor.
    Maps preprocessed 11-D tabular features to risk labels.
    """
    def __init__(self, input_dim: int = 11, num_classes: int = 3) -> None:
        super(TabularOnlyPredictor, self).__init__()
        self.encoder = TabularEncoder(input_dim=input_dim, output_dim=128)
        
        self.fc = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(64, num_classes)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_tabular: torch.Tensor, x_embedding: torch.Tensor = None) -> torch.Tensor:
        # x_embedding is ignored
        x = self.encoder(x_tabular)
        
        x = self.fc(x)
        if x.size(0) > 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        logits = self.out(x)
        return logits

class EmbeddingOnlyPredictor(nn.Module):
    """
    Model B: Embeddings Only Predictor.
    Identical to the Phase 6 MLP model mapping 768-D code embeddings to risk labels.
    """
    def __init__(self, input_dim: int = 768, num_classes: int = 3) -> None:
        super(EmbeddingOnlyPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        
        self.out = nn.Linear(128, num_classes)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_tabular: torch.Tensor, x_embedding: torch.Tensor) -> torch.Tensor:
        # x_tabular is ignored
        x = x_embedding
        
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.relu3(x)
        
        # Output Logits
        logits = self.out(x)
        return logits

class HybridRiskPredictor(nn.Module):
    """
    Model C: Hybrid Fusion Predictor.
    Multi-branch architecture concatenating tabular (128-D) and embedding (256-D) representations.
    """
    def __init__(self, tabular_dim: int = 11, embedding_dim: int = 768, num_classes: int = 3) -> None:
        super(HybridRiskPredictor, self).__init__()
        self.tabular_encoder = TabularEncoder(input_dim=tabular_dim, output_dim=128)
        self.embedding_encoder = EmbeddingEncoder(input_dim=embedding_dim, output_dim=256)
        
        # Fusion classification branch
        self.fc1 = nn.Linear(128 + 256, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        
        self.out = nn.Linear(128, num_classes)
        
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_tabular: torch.Tensor, x_embedding: torch.Tensor) -> torch.Tensor:
        # Parallel encoders
        feat_tabular = self.tabular_encoder(x_tabular)
        feat_embedding = self.embedding_encoder(x_embedding)
        
        # Concatenate branches (128 + 256 = 384)
        x = torch.cat((feat_tabular, feat_embedding), dim=1)
        
        # Fusion Classifier
        x = self.fc1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.relu2(x)
        
        # Output Logits
        logits = self.out(x)
        return logits

if __name__ == "__main__":
    tab_in = torch.randn(5, 11)
    emb_in = torch.randn(5, 768)
    
    m_a = TabularOnlyPredictor()
    m_b = EmbeddingOnlyPredictor()
    m_c = HybridRiskPredictor()
    
    print("Model A out shape:", m_a(tab_in, emb_in).shape)
    print("Model B out shape:", m_b(tab_in, emb_in).shape)
    print("Model C out shape:", m_c(tab_in, emb_in).shape)
