from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

from apdo.benchmark import (
    benchmark_config,
    dataset_length,
    plan_batches,
)
from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError
from tests.datasets import RaisingDataset, SyntheticDataset


class EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> torch.Tensor:
        raise IndexError(index)


class StreamingDataset(IterableDataset):
    def __iter__(self):
        yield torch.zeros(4)


class InterruptingDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        raise KeyboardInterrupt


def test_plan_batches_full_and_capped() -> None:
    full = plan_batches(10, 4, None)
    capped = plan_batches(10, 4, 2)

    assert full.batches == 3
    assert full.steady_samples == 6
    assert capped.batches == 2
    assert capped.steady_samples == 4


def test_plan_batches_rejects_short_runs() -> None:
    with pytest.raises(ConfigurationError):
        plan_batches(10, 4, 1)
    with pytest.raises(DatasetError):
        plan_batches(4, 8, None)


def test_dataset_validation() -> None:
    assert dataset_length(SyntheticDataset(size=4)) == 4

    with pytest.raises(DatasetError):
        dataset_length(EmptyDataset())
    with pytest.raises(DatasetError):
        dataset_length(StreamingDataset())


def test_benchmark_config_collects_measurements() -> None:
    dataset = SyntheticDataset(size=16, feature_dim=4)
    plan = plan_batches(len(dataset), 4, 3)

    measurement = benchmark_config(
        dataset,
        LoaderConfig(),
        phase="test",
        batch_size=4,
        device=torch.device("cpu"),
        plan=plan,
        warmup_runs=1,
        repetitions=2,
        max_ram_bytes=None,
    )

    assert measurement.succeeded
    assert measurement.summary is not None
    assert measurement.summary.median_throughput > 0
    assert len(measurement.passes) == 2
    assert all(len(item.latencies_s) == 2 for item in measurement.passes)
    assert measurement.cold_start_s is not None
    assert measurement.memory is not None


def test_worker_failure_is_recorded() -> None:
    dataset = RaisingDataset(size=16)
    plan = plan_batches(len(dataset), 4, 2)

    measurement = benchmark_config(
        dataset,
        LoaderConfig(num_workers=2),
        phase="test",
        batch_size=4,
        device=torch.device("cpu"),
        plan=plan,
        warmup_runs=1,
        repetitions=2,
        max_ram_bytes=None,
    )

    assert not measurement.succeeded
    assert measurement.summary is None
    assert measurement.error_type is not None
    assert "synthetic dataset failure" in (measurement.error_message or "")


def test_keyboard_interrupt_propagates() -> None:
    dataset = InterruptingDataset()
    plan = plan_batches(len(dataset), 4, 2)

    with pytest.raises(KeyboardInterrupt):
        benchmark_config(
            dataset,
            LoaderConfig(),
            phase="test",
            batch_size=4,
            device=torch.device("cpu"),
            plan=plan,
            warmup_runs=1,
            repetitions=2,
            max_ram_bytes=None,
        )
