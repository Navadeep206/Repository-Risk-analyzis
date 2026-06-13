#!/usr/bin/env python3
"""
PyTorch Multilayer Perceptron model for Phase 6.
Predicts code repository risk classes (LOW, MEDIUM, HIGH) from 768-D code embeddings.
"""

import torch
import torch.nn as nn

class RepositoryRiskPredictor(nn.Module):
    """
    Multilayer Perceptron classifier utilizing batch normalization, dropout regularization,
    and Kaiming normal weight initialization.
    """
    def __init__(self, input_dim: int = 768, num_classes: int = 3) -> None:
        super(RepositoryRiskPredictor, self).__init__()
        
        # Dense Architecture
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
        
        # Initialize weights using Kaiming (He) normal initialization
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Applies Kaiming normal weight initialization for linear layers
        to prevent vanishing/exploding gradients with ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass. Output vectors represent raw class logits.
        """
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1:  # BatchNorm fails on batch size of 1
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

if __name__ == "__main__":
    # Quick sanity test
    model = RepositoryRiskPredictor()
    dummy_input = torch.randn(5, 768)
    logits = model(dummy_input)
    print(f"Logits shape: {logits.shape}")
    probs = torch.softmax(logits, dim=1)
    print(f"Probs shape: {probs.shape}")
    print(f"Sample probabilities: {probs[0]}")
