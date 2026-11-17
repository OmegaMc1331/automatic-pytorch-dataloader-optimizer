from __future__ import annotations

import os
import time

import pytest
import torch
from torch.utils.data import Dataset, IterableDataset

from apdo.benchmark import (
    _run_cycle,
    _shutdown_loader,
    benchmark_config,
    benchmark_rounds,
    dataset_length,
    plan_batches,
    plan_from_template,
)
from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError
from apdo.loader import LoaderSource, LoaderTemplate
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
    assert full.expected_batches == 3
    assert full.full_epoch
    assert capped.batches == 2
    assert capped.steady_samples == 4
    assert capped.expected_batches == 3
    assert not capped.full_epoch


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


class WorkerSensitiveDataset(Dataset):
    def __init__(self, size: int = 16) -> None:
        self.size = size
        self.parent_pid = os.getpid()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        if os.getpid() != self.parent_pid:
            raise RuntimeError("worker process failure")
        return torch.full((4,), float(index))


def test_benchmark_rounds_records_balanced_rounds_and_actual_samples() -> None:
    dataset = SyntheticDataset(size=16, feature_dim=4)
    template = LoaderTemplate.from_dataset(dataset, 4)
    plan = plan_from_template(template, 3)
    configs = (LoaderConfig(), LoaderConfig(1))
    messages: list[str] = []

    stage = benchmark_rounds(
        template,
        configs,
        phase="test",
        device=torch.device("cpu"),
        plan=plan,
        warmup_rounds=1,
        rounds=2,
        max_ram_bytes=None,
        seed=5,
        progress=messages.append,
    )

    assert stage.measurements and all(item.succeeded for item in stage.measurements)
    for measurement in stage.measurements:
        assert len(measurement.rounds) == 2
        assert len(measurement.passes) == 2
        assert [item.round_index for item in measurement.rounds] == [0, 1]
        assert all(item.pass_metrics.steady_samples == 8 for item in measurement.rounds)
        assert all(item.pass_metrics.batches == 3 for item in measurement.rounds)

    measured = [order for order in stage.orders if order.measured]
    warmup = [order for order in stage.orders if not order.measured]
    assert len(measured) == 2
    assert len(warmup) == 1
    assert all(set(order.configs) == set(configs) for order in measured)
    for position in range(2):
        assert {order.configs[position] for order in measured} == set(configs)
    assert any("round 1/2" in message for message in messages)
    assert any("warmup 1/1" in message for message in messages)


def test_benchmark_rounds_isolates_worker_failure() -> None:
    dataset = WorkerSensitiveDataset()
    template = LoaderTemplate.from_dataset(dataset, 4)
    plan = plan_from_template(template, 2)

    stage = benchmark_rounds(
        template,
        (LoaderConfig(), LoaderConfig(1)),
        phase="test",
        device=torch.device("cpu"),
        plan=plan,
        warmup_rounds=1,
        rounds=2,
        max_ram_bytes=None,
        seed=5,
    )

    by_workers = {item.config.num_workers: item for item in stage.measurements}
    assert by_workers[0].succeeded
    assert not by_workers[1].succeeded
    assert "worker process failure" in (by_workers[1].error_message or "")


@pytest.mark.parametrize(
    ("rounds", "warmup_rounds", "configs"),
    [
        (0, 1, (LoaderConfig(),)),
        (1, -1, (LoaderConfig(),)),
        (1, 0, ()),
    ],
)
def test_benchmark_rounds_validates_arguments(rounds, warmup_rounds, configs) -> None:
    template = LoaderTemplate.from_dataset(SyntheticDataset(size=8), 4)
    plan = plan_from_template(template, 2)

    with pytest.raises(ConfigurationError):
        benchmark_rounds(
            template,
            configs,
            phase="test",
            device=torch.device("cpu"),
            plan=plan,
            warmup_rounds=warmup_rounds,
            rounds=rounds,
            max_ram_bytes=None,
            seed=1,
        )


def test_multi_epoch_reuses_persistent_workers() -> None:
    dataset = SyntheticDataset(size=16, feature_dim=4)
    plan = plan_batches(len(dataset), 4, None)
    measurement = benchmark_config(
        dataset,
        LoaderConfig(num_workers=1, persistent_workers=True),
        phase="test",
        batch_size=4,
        device=torch.device("cpu"),
        plan=plan,
        warmup_runs=0,
        repetitions=1,
        max_ram_bytes=None,
        epochs=3,
    )

    assert measurement.succeeded
    assert measurement.summary is not None
    assert measurement.summary.median_sustained_throughput is not None
    assert measurement.summary.median_worker_startup_overhead_s == 0.0
    cycles = measurement.passes[0].cycles
    assert len(cycles) == 3
    assert all(len(cycle.worker_pids) == 1 for cycle in cycles)
    assert cycles[0].worker_pids == cycles[1].worker_pids == cycles[2].worker_pids
    assert all(cycle.samples == 16 for cycle in cycles)


def test_benchmark_restores_global_rng_state() -> None:
    dataset = SyntheticDataset(size=8, feature_dim=4)
    plan = plan_batches(len(dataset), 4, None)
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()

    benchmark_config(
        dataset,
        LoaderConfig(),
        phase="test",
        batch_size=4,
        device=torch.device("cpu"),
        plan=plan,
        warmup_runs=0,
        repetitions=1,
        max_ram_bytes=None,
    )

    assert torch.equal(torch.get_rng_state(), state)


def test_training_benchmark_records_end_to_end_metrics() -> None:
    dataset = SyntheticDataset(size=16, feature_dim=4)
    plan = plan_batches(len(dataset), 4, None)
    calls = 0

    def step(batch: torch.Tensor) -> None:
        assert batch[0].shape == (4, 4)

    def factory():
        nonlocal calls
        calls += 1
        return step

    measurement = benchmark_config(
        dataset,
        LoaderConfig(),
        phase="training",
        batch_size=4,
        device=torch.device("cpu"),
        plan=plan,
        warmup_runs=0,
        repetitions=1,
        max_ram_bytes=None,
        epochs=2,
        training_step_factory=factory,
    )

    assert measurement.succeeded
    assert measurement.training
    assert measurement.summary is not None
    assert measurement.summary.median_step_latency_s is not None
    assert measurement.summary.median_data_wait_s is not None
    assert calls == 1
    assert len(measurement.passes[0].cycles) == 2
    assert all(len(cycle.step_latencies_s) == 4 for cycle in measurement.passes[0].cycles)


def test_multi_epoch_nonpersistent_workers_record_transitions() -> None:
    dataset = SyntheticDataset(size=16, feature_dim=4)
    plan = plan_batches(len(dataset), 4, 3)
    measurement = benchmark_config(
        dataset,
        LoaderConfig(num_workers=1, persistent_workers=False),
        phase="test",
        batch_size=4,
        device=torch.device("cpu"),
        plan=plan,
        warmup_runs=0,
        repetitions=1,
        max_ram_bytes=None,
        epochs=2,
    )

    assert measurement.succeeded
    pass_metrics = measurement.passes[0]
    assert not pass_metrics.full_epoch
    assert pass_metrics.expected_batches == 4
    assert pass_metrics.worker_startup_overhead_s is not None
    assert pass_metrics.worker_startup_overhead_s > 0.0
    assert all(cycle.worker_pids for cycle in pass_metrics.cycles)


def test_loader_cleanup_failure_is_reported() -> None:
    class FailingIterator:
        def _shutdown_workers(self) -> None:
            raise RuntimeError("worker cleanup failed")

    class Loader:
        _iterator = FailingIterator()

    failure = _shutdown_loader(Loader())  # type: ignore[arg-type]

    assert failure is not None
    assert failure[0] == "WorkerCleanupError"
    assert "worker cleanup failed" in failure[1]


def test_cycle_throughput_excludes_worker_teardown(monkeypatch) -> None:
    dataset = SyntheticDataset(size=16, feature_dim=4)
    template = LoaderTemplate.from_dataset(dataset, 4)
    source = LoaderSource.from_template(template, "dataloader")
    loader = template.build(LoaderConfig())

    def slow_shutdown(iterator):
        time.sleep(1.0)

    monkeypatch.setattr("apdo.benchmark._shutdown_iterator", slow_shutdown)
    started = time.perf_counter()
    cycle = _run_cycle(
        source,
        loader,
        plan=plan_batches(len(dataset), 4, None),
        device=torch.device("cpu"),
        cycle_index=0,
        training_step=None,
    )
    wall = time.perf_counter() - started

    assert wall >= 1.0
    assert cycle.duration_s < 0.9
