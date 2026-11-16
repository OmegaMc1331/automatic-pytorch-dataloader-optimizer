from __future__ import annotations

from pathlib import Path

import torch
from benchmarks.workloads import (
    CpuHeavyDataset,
    DiskBackedDataset,
    InMemoryDataset,
    build_workload,
    materialize_disk_data,
)


def test_in_memory_workload_is_deterministic() -> None:
    dataset = InMemoryDataset(size=16, feature_dim=8)

    first = dataset[3]
    second = dataset[3]

    assert len(dataset) == 16
    assert first[0].shape == (8,)
    assert torch.equal(first[0], second[0])
    assert first[1].item() == second[1].item()


def test_disk_workload_uses_local_records(tmp_path: Path) -> None:
    path = materialize_disk_data(tmp_path / "data.bin", size=32, feature_dim=8)
    dataset = DiskBackedDataset(path, size=32, feature_dim=8)

    first = dataset[0]
    second = dataset[0]

    assert path.is_file()
    assert path.stat().st_size == 32 * (8 * 4 + 8)
    assert first[0].shape == (8,)
    assert first[0].dtype == torch.float32
    assert first[1].dtype == torch.long
    assert torch.equal(first[0], second[0])
    assert first[1].item() == second[1].item()


def test_cpu_heavy_workload_is_deterministic() -> None:
    dataset = CpuHeavyDataset(size=8, feature_dim=16)

    first = dataset[2]
    second = dataset[2]

    assert len(dataset) == 8
    assert first[0].shape == (16,)
    assert torch.equal(first[0], second[0])
    assert first[1].item() == second[1].item()


def test_build_workload_returns_cap_and_batch_size(tmp_path: Path) -> None:
    dataset, batch_size, max_batches = build_workload("disk-backed", tmp_path)

    assert isinstance(dataset, DiskBackedDataset)
    assert batch_size == 32
    assert max_batches == 40
    assert (tmp_path / "disk_backed_f32_d128.bin").is_file()
