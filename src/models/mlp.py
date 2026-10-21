import torch
import torch.nn as nn

class TransitMLP(nn.Module):
    """
    Standard Feed-Forward Multilayer Perceptron for unimodal light-curve baseline.
    """
    def __init__(self, input_size=80):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.model(x)


class FusionMLP(nn.Module):
    """
    Non-linear Multilayer Perceptron for fused multimodal inputs (80 LC bins + 3 metadata features = 83).
    Includes batch normalization and dropout for robust regularized learning.
    """
    def __init__(self, input_dim=83):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.mlp(x)
