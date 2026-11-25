from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from apdo.training import TrainingStepFactory


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


class StartupBoundDataset(Dataset):
    def __init__(self, size: int = 4096, feature_dim: int = 8) -> None:
        generator = torch.Generator().manual_seed(2029)
        self.features = torch.randn(size, feature_dim, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


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
            handle.write(array("f", features[index].tolist()).tobytes())
            handle.write(int(labels[index]).to_bytes(8, "little", signed=True))
    temporary.replace(path)
    return path


class TrainingNetwork(nn.Module):
    def __init__(self, feature_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def build_training_step_factory(
    device: str | torch.device,
    *,
    feature_dim: int = 64,
    hidden_dim: int = 256,
) -> Callable[[], Callable[[Any], None]]:
    resolved_device = torch.device(device)
    loss_fn = nn.CrossEntropyLoss()

    def factory() -> Callable[[Any], None]:
        model = TrainingNetwork(feature_dim, hidden_dim).to(resolved_device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

        def step(batch: Any) -> None:
            features, labels = batch
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), labels)
            loss.backward()
            optimizer.step()

        return step

    return factory


def build_compute_bound_training_step_factory(
    device: str | torch.device,
    *,
    feature_dim: int = 128,
) -> Callable[[], Callable[[Any], None]]:
    resolved_device = torch.device(device)
    hidden_dim = 1024 if resolved_device.type == "cuda" else 512
    return build_training_step_factory(
        resolved_device,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
    )


class ComputeBoundDataset(Dataset):
    def __init__(self, size: int = 4096, feature_dim: int = 128) -> None:
        generator = torch.Generator().manual_seed(2030)
        self.features = torch.randn(size, feature_dim, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    name: str
    dataset: Dataset
    batch_size: int
    max_batches: int
    mode: str
    training_step_factory: TrainingStepFactory | None = None


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
    if name == "startup-bound":
        return StartupBoundDataset(), 32, 80
    if name == "compute-bound":
        return ComputeBoundDataset(), 64, 32
    if name == "training":
        return InMemoryDataset(), 32, 20
    raise ValueError(
        f"unknown workload {name!r}; expected in-memory, disk-backed, cpu-heavy, "
        "startup-bound, compute-bound, or training"
    )


def build_workload_spec(
    name: str,
    cache_directory: Path,
    *,
    device: str | torch.device = "cpu",
) -> WorkloadSpec:
    dataset, batch_size, max_batches = build_workload(name, cache_directory)
    if name == "compute-bound":
        return WorkloadSpec(
            name=name,
            dataset=dataset,
            batch_size=batch_size,
            max_batches=max_batches,
            mode="training",
            training_step_factory=build_compute_bound_training_step_factory(device),
        )
    if name == "training":
        return WorkloadSpec(
            name=name,
            dataset=dataset,
            batch_size=batch_size,
            max_batches=max_batches,
            mode="training",
            training_step_factory=build_training_step_factory(
                device,
                feature_dim=64,
                hidden_dim=256,
            ),
        )
    return WorkloadSpec(
        name=name,
        dataset=dataset,
        batch_size=batch_size,
        max_batches=max_batches,
        mode="loader",
    )


def create_cpu_heavy_dataset() -> Dataset:
    return CpuHeavyDataset()


def create_compute_bound_dataset() -> Dataset:
    return ComputeBoundDataset()


def create_compute_bound_training_step(
    device: str | torch.device = "cpu",
) -> TrainingStepFactory:
    return build_compute_bound_training_step_factory(device)
