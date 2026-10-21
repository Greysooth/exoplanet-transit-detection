import torch
import torch.nn as nn

class TransitRNN(nn.Module):
    """
    Recurrent baseline (2-layer GRU) with temporal average pooling.
    Evaluates purely sequential temporal modeling of photometric dips.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # x is (B, 1, T) or (B, T)
        if x.ndim == 2:
            x = x.unsqueeze(2)  # (B, T, 1)
        elif x.ndim == 3 and x.size(1) == 1:
            x = x.permute(0, 2, 1)  # (B, T, 1)
        out, _ = self.rnn(x)
        out = torch.mean(out, dim=1)  # Temporal mean pooling
        return self.classifier(out)
