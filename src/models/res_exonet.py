import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import TransitCNN

class LearnableAttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_layer = nn.Linear(d_model, 1)

    def forward(self, seq):
        scores = self.attn_layer(seq)
        weights = F.softmax(scores, dim=1)
        pooled = torch.sum(seq * weights, dim=1)
        return pooled

class ResExoNet(nn.Module):
    """
    Three-Branch Residual Architecture:
    - Branch 1: CNN Spatial Features + Attention Pooling (128-dim)
    - Branch 2: Metadata Deep MLP Projection (32-dim)
    - Branch 3: Direct Raw Residual Skip Connection (80 LC + 3 Meta = 83-dim)
    - Total Concatenated Fusion: 128 + 32 + 83 = 243-dim input to deep classifier head.
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.cnn = TransitCNN()
        self.pool = LearnableAttentionPooling(d_model)
        
        self.stage1_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        fusion_in_dim = d_model + 32 + 83  # 243
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, meta, mode="joint"):
        seq_features = self.cnn.extract_sequence_features(x)
        seq_features = seq_features.permute(0, 2, 1)
        lc_spatial_emb = self.pool(seq_features)
        
        if mode == "stage1":
            return self.stage1_classifier(lc_spatial_emb)
            
        meta_context_emb = self.meta_mlp(meta)
        flat_lc = x.squeeze(1) if x.ndim == 3 else x
        raw_residual_skip = torch.cat([flat_lc, meta], dim=1)
        fused_tensor = torch.cat([lc_spatial_emb, meta_context_emb, raw_residual_skip], dim=1)
        return self.classifier(fused_tensor)
