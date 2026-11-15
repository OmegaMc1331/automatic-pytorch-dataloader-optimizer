from __future__ import annotations

import time

import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    def __init__(self, size: int = 64, feature_dim: int = 8, delay_s: float = 0.0) -> None:
        self.size = size
        self.feature_dim = feature_dim
        self.delay_s = delay_s

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.delay_s:
            time.sleep(self.delay_s)
        features = torch.full((self.feature_dim,), float(index), dtype=torch.float32)
        label = torch.tensor(index % 10, dtype=torch.long)
        return features, label


class RaisingDataset(Dataset):
    def __init__(self, size: int = 16) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        raise RuntimeError("synthetic dataset failure")
