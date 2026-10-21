import torch
import torch.nn as nn
from .cnn import TransitCNN
from .metadata_mlp import MetadataMLP

class TwoStageDualModel(nn.Module):
    """
    Two-Stage Multimodal Model:
    Allows isolated representation pre-training of the CNN spatial branch (Stage 1),
    followed by metadata projection alignment (Stage 2A) and end-to-end joint fine-tuning (Stage 2B).
    """
    def __init__(self, cnn_ckpt_path=None):
        super().__init__()
        self.cnn = TransitCNN()
        if cnn_ckpt_path:
            self.cnn.load_state_dict(torch.load(cnn_ckpt_path, weights_only=True))

        self.meta_mlp = MetadataMLP()
        
        # Primary Multimodal Fusion Head (128 CNN + 8 Meta = 136)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        # Dedicated auxiliary head for Stage 1 CNN pre-training
        self.stage1_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, meta=None, mode="joint"):
        """
        mode="stage1": Evaluates only the light curve via the isolated CNN branch.
        mode="joint" : Evaluates both inputs through the multi-modal fusion architecture.
        """
        lc_feat = self.cnn.extract_features(x)
        if mode == "stage1":
            return self.stage1_classifier(lc_feat)
        else:
            meta_feat = self.meta_mlp(meta)
            fused = torch.cat([lc_feat, meta_feat], dim=1)
            return self.classifier(fused)
