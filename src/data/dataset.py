import torch
from torch.utils.data import Dataset
import numpy as np

class TransitDataset(Dataset):
    """
    Light-curve-only PyTorch Dataset.
    X: shape (N, 1, 80) tensor.
    y: shape (N,) integer class tensor.
    """
    def __init__(self, X, y):
        if isinstance(X, str):
            X = np.load(X)
        if isinstance(y, str):
            y = np.load(y)

        X_arr = np.array(X, dtype=np.float32)
        if X_arr.ndim == 2:
            X_arr = np.expand_dims(X_arr, axis=1)

        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MetadataDataset(Dataset):
    """
    Multimodal PyTorch Dataset: returns (light_curve, metadata, label).
    X: (N, 1, 80)
    meta: (N, 3)
    y: (N,)
    """
    def __init__(self, X, meta, y):
        if isinstance(X, str):
            X = np.load(X)
        if isinstance(meta, str):
            meta = np.load(meta)
        if isinstance(y, str):
            y = np.load(y)

        X_arr = np.array(X, dtype=np.float32)
        if X_arr.ndim == 2:
            X_arr = np.expand_dims(X_arr, axis=1)

        self.X = torch.tensor(X_arr, dtype=torch.float32)
        self.meta = torch.tensor(meta, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.meta[idx], self.y[idx]


class CombinedDataset(Dataset):
    """
    Flattened Multimodal Dataset: returns concatenated (80 LC + 3 Meta = 83 features).
    Ideal for dense MLP and baseline classifiers.
    """
    def __init__(self, X_combined, y):
        self.X = torch.tensor(X_combined, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
