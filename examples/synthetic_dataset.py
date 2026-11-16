from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataset(Dataset):
    def __init__(self, size: int = 512, feature_dim: int = 32) -> None:
        self.size = size
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        features = torch.randn(self.feature_dim, generator=generator)
        label = torch.tensor(index % 10, dtype=torch.long)
        return features, label


def create_dataset() -> SyntheticDataset:
    return SyntheticDataset()


def create_loader() -> DataLoader:
    return DataLoader(
        SyntheticDataset(size=128),
        batch_size=16,
        drop_last=True,
        num_workers=0,
    )


def count_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> int:
    return int(batch[0].shape[0])


def create_sample_counter() -> Callable[[Any], int]:
    return count_batch
