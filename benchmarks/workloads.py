from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class InMemoryDataset(Dataset):
    def __init__(self, size: int = 4096, feature_dim: int = 64) -> None:
        generator = torch.Generator().manual_seed(2026)
        self.features = torch.randn(size, feature_dim, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)
        self.scale = torch.linspace(0.5, 1.5, feature_dim)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.clamp(self.features[index] * self.scale, -4.0, 4.0)
        return features, self.labels[index]


class DiskBackedDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        size: int,
        feature_dim: int,
        seed: int = 2027,
    ) -> None:
        self.path = path
        self.size = size
        self.feature_dim = feature_dim
        self.record_bytes = feature_dim * 4 + 8
        generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=generator).tolist()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index = self.order[index]
        with self.path.open("rb") as handle:
            handle.seek(record_index * self.record_bytes)
            record = handle.read(self.record_bytes)
        feature_bytes = self.feature_dim * 4
        features = torch.frombuffer(bytearray(record[:feature_bytes]), dtype=torch.float32)
        label = int.from_bytes(record[feature_bytes:], "little", signed=True)
        return features, torch.tensor(label, dtype=torch.long)


class CpuHeavyDataset(Dataset):
    def __init__(self, size: int = 1024, feature_dim: int = 128) -> None:
        generator = torch.Generator().manual_seed(2028)
        self.features = torch.randn(size, feature_dim, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)
        self.matrices = (
            torch.randn(4, feature_dim, feature_dim, generator=generator) / feature_dim**0.5
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        for matrix in self.matrices:
            features = torch.tanh(features @ matrix)
            features = features / (1.0 + features.abs())
        return features, self.labels[index]


def materialize_disk_data(
    path: Path,
    *,
    size: int = 2048,
    feature_dim: int = 128,
    seed: int = 2027,
) -> Path:
    if path.is_file() and path.stat().st_size == size * (feature_dim * 4 + 8):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(size, feature_dim, generator=generator)
    labels = torch.randint(0, 10, (size,), generator=generator)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        for index in range(size):
            handle.write(features[index].numpy().tobytes())
            handle.write(int(labels[index]).to_bytes(8, "little", signed=True))
    temporary.replace(path)
    return path


def build_workload(
    name: str,
    cache_directory: Path,
) -> tuple[Dataset, int, int]:
    if name == "in-memory":
        return InMemoryDataset(), 32, 50
    if name == "disk-backed":
        path = materialize_disk_data(cache_directory / "disk_backed_f32_d128.bin")
        return DiskBackedDataset(path, size=2048, feature_dim=128), 32, 40
    if name == "cpu-heavy":
        return CpuHeavyDataset(), 32, 20
    raise ValueError(f"unknown workload {name!r}; expected in-memory, disk-backed, or cpu-heavy")
