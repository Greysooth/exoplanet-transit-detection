import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn import TransitCNN
from .metadata_mlp import MetadataMLP

class AttentionPooling(nn.Module):
    """
    Learnable Attention Pooling: dynamically weights each temporal phase slice,
    allowing the model to focus on the central transit dip while suppressing out-of-transit noise.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attn_layer = nn.Linear(d_model, 1)

    def forward(self, seq):
        # seq: (B, T', d_model)
        scores = self.attn_layer(seq)            # (B, T', 1)
        weights = F.softmax(scores, dim=1)       # (B, T', 1)
        pooled = torch.sum(seq * weights, dim=1) # (B, d_model)
        return pooled


class HybridCNN(nn.Module):
    """
    CNN-Transformer Hybrid (Light-curve only):
    Extracts spatial sequence features via CNN, applies self-attention with learnable
    positional embeddings, and pools using AttentionPooling.
    """
    def __init__(
        self,
        cnn_ckpt_path=None,
        max_len=20,
        d_model=128,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.cnn = TransitCNN()
        if cnn_ckpt_path:
            self.cnn.load_state_dict(torch.load(cnn_ckpt_path, weights_only=True))

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        seq = self.cnn.extract_sequence_features(x)  # (B, 128, T')
        seq = seq.permute(0, 2, 1)                   # (B, T', 128)
        seq = seq + self.pos_embedding[:, : seq.size(1), :]
        seq = self.transformer(seq)                  # (B, T', 128)
        pooled = self.pool(seq)                      # (B, 128)
        return self.classifier(pooled)


class HybridDualInput(nn.Module):
    """
    Flagship Multimodal Hybrid Architecture:
    Combines 1D-CNN spatial feature extraction, 1-layer Transformer self-attention,
    learnable attention-weighted pooling, and MetadataMLP fusion.
    """
    def __init__(
        self,
        cnn_ckpt_path=None,
        max_len=20,
        d_model=128,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.cnn = TransitCNN()
        if cnn_ckpt_path:
            self.cnn.load_state_dict(torch.load(cnn_ckpt_path, weights_only=True))

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)
        self.meta_mlp = MetadataMLP()

        self.classifier = nn.Sequential(
            nn.Linear(d_model + 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x, meta):
        seq = self.cnn.extract_sequence_features(x)  # (B, 128, T')
        seq = seq.permute(0, 2, 1)                   # (B, T', 128)
        seq = seq + self.pos_embedding[:, : seq.size(1), :]
        seq = self.transformer(seq)                  # (B, T', 128)
        lc_feat = self.pool(seq)                     # (B, 128)
        meta_feat = self.meta_mlp(meta)              # (B, 8)
        fused = torch.cat([lc_feat, meta_feat], dim=1)  # (B, 136)
        return self.classifier(fused)
