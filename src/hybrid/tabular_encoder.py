#!/usr/bin/env python3
"""
Tabular feature encoder branch.
"""

import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
    """
    Encoder for preprocessed tabular features (11-D -> 128-D).
    """
    def __init__(self, input_dim: int = 11, output_dim: int = 128) -> None:
        super(TabularEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if x.size(0) > 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
