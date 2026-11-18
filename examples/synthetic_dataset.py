from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from apdo import LoaderConfig


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


def create_training_loader(config: LoaderConfig) -> DataLoader:
    return DataLoader(
        SyntheticDataset(size=128),
        batch_size=16,
        drop_last=True,
        **config.to_dataloader_kwargs(),
    )


def create_training_step_factory(device: str = "cpu") -> Callable[[], Callable[[Any], None]]:
    resolved_device = torch.device(device)
    loss_fn = nn.CrossEntropyLoss()

    def factory() -> Callable[[Any], None]:
        model = nn.Linear(32, 10).to(resolved_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def step(batch: tuple[torch.Tensor, torch.Tensor]) -> None:
            features, labels = batch
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()

        return step

    return factory


def count_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> int:
    return int(batch[0].shape[0])


def create_sample_counter() -> Callable[[Any], int]:
    return count_batch


def fingerprint_batch(batch: tuple[torch.Tensor, torch.Tensor]) -> Any:
    features, labels = batch
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()
        return (
            tuple(labels.reshape(-1).tolist()),
            tuple(features.reshape(features.shape[0], -1)[0].tolist()),
        )
    return (labels,)


def create_sample_fingerprint() -> Callable[[Any], Any]:
    return fingerprint_batch


training_step_factory = create_training_step_factory("cpu")
