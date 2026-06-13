#!/usr/bin/env python3
"""
Embedding encoder branch.
"""

import torch
import torch.nn as nn

class EmbeddingEncoder(nn.Module):
    """
    Encoder for CodeBERT embeddings (768-D -> 256-D).
    """
    def __init__(self, input_dim: int = 768, output_dim: int = 256) -> None:
        super(EmbeddingEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        self.fc2 = nn.Linear(512, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        return x
