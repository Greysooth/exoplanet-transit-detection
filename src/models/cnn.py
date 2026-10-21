import torch
import torch.nn as nn

class TransitCNN(nn.Module):
    """
    Lightweight 1D Convolutional Neural Network for transit light-curve feature extraction.
    Features 3 convolutional blocks designed to match the physical width of transit dips.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Conv1d kernel 11 matches broad ingress/egress profiles
            nn.Conv1d(1, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 80 -> 40
            
            # Block 2: Conv1d kernel 5 matches transit bottom profile
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 40 -> 20
            
            # Block 3: Conv1d kernel 3 captures fine temporal variations
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Output temporal length T' = 20 for 80-bin input
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def extract_sequence_features(self, x):
        """Returns sequence feature map (B, 128, T') for hybrid/attention models."""
        return self.features(x)

    def extract_features(self, x):
        """Returns 128-dimensional spatial feature vector (B, 128)."""
        x = self.extract_sequence_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        return self.classifier(self.extract_features(x))
