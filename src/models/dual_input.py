import torch
import torch.nn as nn
from .cnn import TransitCNN
from .metadata_mlp import MetadataMLP

class DualInputModel(nn.Module):
    """
    Standard Late Fusion Architecture:
    Concatenates 128-dim CNN light-curve spatial features with 8-dim MetadataMLP features
    into a unified 136-dim representation before classification.
    """
    def __init__(self, cnn_ckpt_path=None):
        super().__init__()
        self.cnn = TransitCNN()
        if cnn_ckpt_path is not None:
            self.cnn.load_state_dict(torch.load(cnn_ckpt_path, weights_only=True))

        self.meta_mlp = MetadataMLP()
        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, meta):
        lc_feat = self.cnn.extract_features(x)
        meta_feat = self.meta_mlp(meta)
        fused = torch.cat([lc_feat, meta_feat], dim=1)
        return self.classifier(fused)
