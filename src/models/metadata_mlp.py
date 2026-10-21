import torch
import torch.nn as nn

class MetadataMLP(nn.Module):
    """
    Sub-network projecting 3 astrophysical stellar parameters
    (T_eff, P_orb, R_star) into a latent embedding vector (8-dim).
    """
    def __init__(self, input_dim=3, hidden_dim=16, out_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, meta):
        return self.model(meta)
