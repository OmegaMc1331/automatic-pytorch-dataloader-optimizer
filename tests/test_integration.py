from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apdo import optimize
from tests.datasets import SyntheticDataset


def test_optimize_runs_real_cpu_benchmark(tmp_path: Path) -> None:
    dataset = SyntheticDataset(size=64, feature_dim=4, delay_s=0.0002)
    messages: list[str] = []

    result = optimize(
        dataset,
        batch_size=8,
        device="cpu",
        max_ram_gb=2,
        warmup_runs=1,
        repetitions=2,
        max_batches=4,
        max_workers=1,
        progress=messages.append,
    )

    assert result.settings.dataset_size == 64
    assert len(result.measurements) >= 2
    assert result.best_summary.median_throughput > 0
    assert any(item.succeeded and item.config.num_workers > 0 for item in result.measurements)
    assert result.fastest_config in {item.config for item in result.measurements if item.succeeded}
    assert result.improvement_percent is not None
    assert result.duration_s > 0
    assert any("Stage 1/2" in message for message in messages)
    assert any("Stage 2/2" in message for message in messages)

    loader = DataLoader(
        dataset,
        batch_size=8,
        **result.best_config.to_dataloader_kwargs(),
    )
    features, _ = next(iter(loader))
    assert features.shape == (8, 4)

    result.to_json(tmp_path / "benchmark.json")
    result.to_html(tmp_path / "benchmark.html")
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "benchmark.html").is_file()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_transfer_measurement() -> None:
    dataset = TensorDataset(
        torch.randn(32, 8),
        torch.arange(32, dtype=torch.long),
    )

    result = optimize(
        dataset,
        batch_size=8,
        device="cuda",
        warmup_runs=1,
        repetitions=2,
        max_batches=3,
        max_workers=1,
    )

    best = next(
        item for item in result.measurements if item.succeeded and item.config == result.best_config
    )
    assert best.gpu_transfer_active
    assert best.summary is not None
    assert best.summary.peak_gpu_memory_bytes is not None
    assert best.summary.peak_gpu_memory_bytes > 0
