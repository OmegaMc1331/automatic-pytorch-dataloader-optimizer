from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError
from apdo.metrics import MetricsSummary, PassMetrics, summarize_passes
from apdo.resources import MemoryMonitor, MemorySnapshot


@dataclass(frozen=True, slots=True)
class BatchPlan:
    batches: int
    steady_samples: int


@dataclass(frozen=True, slots=True)
class ConfigMeasurement:
    config: LoaderConfig
    phase: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    passes: tuple[PassMetrics, ...] = ()
    summary: MetricsSummary | None = None
    cold_start_s: float | None = None
    gpu_transfer_active: bool = False
    memory: MemorySnapshot | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "config": self.config.to_dict(),
            "phase": self.phase,
            "status": self.status,
            "cold_start_s": self.cold_start_s,
            "gpu_transfer_active": self.gpu_transfer_active,
            "passes": [item.to_dict() for item in self.passes],
        }
        if self.summary is not None:
            result["summary"] = self.summary.to_dict()
        if self.memory is not None:
            result["memory"] = self.memory.to_dict()
        if self.error_type is not None:
            result["error_type"] = self.error_type
        if self.error_message is not None:
            result["error_message"] = self.error_message
        return result


def dataset_length(dataset: Dataset) -> int:
    if isinstance(dataset, IterableDataset):
        raise DatasetError(
            "IterableDataset is not supported in v0.1.0; use a map-style dataset "
            "with __len__ and __getitem__"
        )
    if not hasattr(dataset, "__getitem__"):
        raise DatasetError("dataset must implement __getitem__")
    try:
        length = len(dataset)
    except Exception as exc:
        raise DatasetError(f"could not determine dataset length: {exc}") from exc
    if length == 0:
        raise DatasetError("dataset is empty")
    return length


def plan_batches(dataset_size: int, batch_size: int, max_batches: int | None) -> BatchPlan:
    total_batches = math.ceil(dataset_size / batch_size)
    if total_batches < 2:
        raise DatasetError(
            f"dataset has {dataset_size} samples, which produces only {total_batches} batch "
            f"at batch_size={batch_size}; at least two batches are required"
        )
    if max_batches is not None and max_batches < 2:
        raise ConfigurationError("max_batches must be at least 2 or None")
    batches = total_batches if max_batches is None else min(total_batches, max_batches)
    total_samples = dataset_size if batches == total_batches else batches * batch_size
    steady_samples = total_samples - min(batch_size, dataset_size)
    return BatchPlan(batches=batches, steady_samples=steady_samples)


def benchmark_config(
    dataset: Dataset,
    config: LoaderConfig,
    *,
    phase: str,
    batch_size: int,
    device: torch.device,
    plan: BatchPlan,
    warmup_runs: int,
    repetitions: int,
    max_ram_bytes: int | None,
) -> ConfigMeasurement:
    monitor = MemoryMonitor(max_ram_bytes)
    loader: DataLoader | None = None
    measured_passes: list[PassMetrics] = []
    cold_start: float | None = None
    gpu_transfer = False
    failure: tuple[str, str] | None = None
    gpu_stats_reset = False

    try:
        monitor.start()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            gpu_stats_reset = True

        construction_start = time.perf_counter()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            **config.to_dataloader_kwargs(),
        )
        construction_s = time.perf_counter() - construction_start
        monitor.raise_if_exceeded()

        for warmup_index in range(warmup_runs):
            warmup = _run_pass(
                loader,
                plan=plan,
                device=device,
                transfer=device.type == "cuda",
                monitor=monitor,
            )
            gpu_transfer = gpu_transfer or warmup.gpu_transfer_active
            if warmup_index == 0:
                cold_start = construction_s + warmup.startup_s

        for _ in range(repetitions):
            measured = _run_pass(
                loader,
                plan=plan,
                device=device,
                transfer=device.type == "cuda",
                monitor=monitor,
            )
            measured_passes.append(measured)
            gpu_transfer = gpu_transfer or measured.gpu_transfer_active
    except Exception as exc:
        failure = (type(exc).__name__, _describe_exception(exc))
    finally:
        if loader is not None:
            _shutdown_loader(loader)
        snapshot = monitor.stop()
        gpu_peak = _gpu_peak_bytes(device) if gpu_stats_reset else None

    if failure is not None:
        error_type, error_message = failure
        return ConfigMeasurement(
            config=config,
            phase=phase,
            status="failed",
            error_type=error_type,
            error_message=error_message,
            cold_start_s=cold_start,
            gpu_transfer_active=gpu_transfer,
            memory=snapshot,
        )

    summary = summarize_passes(
        measured_passes,
        peak_process_rss_bytes=snapshot.peak_process_rss_bytes,
        peak_system_used_bytes=snapshot.peak_system_used_bytes,
        peak_gpu_memory_bytes=gpu_peak,
    )
    return ConfigMeasurement(
        config=config,
        phase=phase,
        status="ok",
        passes=tuple(measured_passes),
        summary=summary,
        cold_start_s=cold_start,
        gpu_transfer_active=gpu_transfer,
        memory=snapshot,
    )


def _run_pass(
    loader: DataLoader,
    *,
    plan: BatchPlan,
    device: torch.device,
    transfer: bool,
    monitor: MemoryMonitor,
) -> PassMetrics:
    if transfer:
        torch.cuda.synchronize(device)

    startup_start = time.perf_counter()
    iterator = iter(loader)
    first_batch = next(iterator)
    transfer_active = False
    if transfer:
        _, moved = _move_to_device(first_batch, device)
        transfer_active = transfer_active or moved
    startup_s = time.perf_counter() - startup_start
    monitor.raise_if_exceeded()

    latencies: list[float] = []
    steady_start = time.perf_counter()
    for _ in range(1, plan.batches):
        batch_start = time.perf_counter()
        batch = next(iterator)
        if transfer:
            _, moved = _move_to_device(batch, device)
            transfer_active = transfer_active or moved
        latencies.append(time.perf_counter() - batch_start)
        monitor.raise_if_exceeded()
    if transfer:
        torch.cuda.synchronize(device)
    elapsed_s = max(time.perf_counter() - steady_start, 1e-9)

    return PassMetrics(
        startup_s=startup_s,
        steady_elapsed_s=elapsed_s,
        batches=plan.batches,
        steady_samples=plan.steady_samples,
        throughput=plan.steady_samples / elapsed_s,
        latencies_s=tuple(latencies),
        gpu_transfer_active=transfer_active,
    )


def _move_to_device(value: Any, device: torch.device) -> tuple[Any, bool]:
    if isinstance(value, torch.Tensor):
        return value.to(device), True
    if isinstance(value, Mapping):
        moved: dict[Any, Any] = {}
        found = False
        for key, item in value.items():
            moved_item, item_found = _move_to_device(item, device)
            moved[key] = moved_item
            found = found or item_found
        return moved, found
    if isinstance(value, tuple):
        items = []
        found = False
        for item in value:
            moved_item, item_found = _move_to_device(item, device)
            items.append(moved_item)
            found = found or item_found
        return tuple(items), found
    if isinstance(value, list):
        items = []
        found = False
        for item in value:
            moved_item, item_found = _move_to_device(item, device)
            items.append(moved_item)
            found = found or item_found
        return items, found
    return value, False


def _gpu_peak_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return None


def _shutdown_loader(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        with contextlib.suppress(Exception):
            shutdown()


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip() or repr(exc)
    if len(message) > 2000:
        return f"{message[:1997]}..."
    return message
