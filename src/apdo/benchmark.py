from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from apdo.config import LoaderConfig
from apdo.errors import ConfigurationError, DatasetError
from apdo.loader import LoaderTemplate, map_dataset_length
from apdo.metrics import MetricsSummary, PassMetrics, summarize_passes
from apdo.order import RoundOrder, balanced_orders
from apdo.resources import MemoryMonitor, MemorySnapshot


@dataclass(frozen=True, slots=True)
class BatchPlan:
    batches: int
    steady_samples: int


@dataclass(frozen=True, slots=True)
class RoundMeasurement:
    round_index: int
    position: int
    pass_metrics: PassMetrics
    cold_start_s: float | None
    memory: MemorySnapshot
    gpu_peak_bytes: int | None = None
    gpu_transfer_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "position": self.position,
            "pass": self.pass_metrics.to_dict(),
            "cold_start_s": self.cold_start_s,
            "gpu_peak_bytes": self.gpu_peak_bytes,
            "gpu_transfer_active": self.gpu_transfer_active,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ConfigMeasurement:
    config: LoaderConfig
    phase: str
    status: str
    error_type: str | None = None
    error_message: str | None = None
    passes: tuple[PassMetrics, ...] = ()
    rounds: tuple[RoundMeasurement, ...] = ()
    summary: MetricsSummary | None = None
    cold_start_s: float | None = None
    gpu_transfer_active: bool = False
    memory: MemorySnapshot | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def summary_or_raise(self) -> MetricsSummary:
        if self.summary is None:
            raise ValueError(f"no summary is available for {self.config.describe()}")
        return self.summary

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "config": self.config.to_dict(),
            "phase": self.phase,
            "status": self.status,
            "cold_start_s": self.cold_start_s,
            "gpu_transfer_active": self.gpu_transfer_active,
            "passes": [item.to_dict() for item in self.passes],
            "rounds": [item.to_dict() for item in self.rounds],
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


@dataclass(frozen=True, slots=True)
class StageResult:
    measurements: tuple[ConfigMeasurement, ...]
    orders: tuple[RoundOrder, ...]
    budget_exhausted: bool = False


@dataclass(slots=True)
class _Accumulator:
    rounds: list[RoundMeasurement] = field(default_factory=list)
    memories: list[MemorySnapshot] = field(default_factory=list)
    cold_start_s: float | None = None
    gpu_peaks: list[int] = field(default_factory=list)
    gpu_transfer_active: bool = False
    failure: tuple[str, str] | None = None


@dataclass(frozen=True, slots=True)
class _PassExecution:
    pass_metrics: PassMetrics | None
    cold_start_s: float | None
    memory: MemorySnapshot
    gpu_peak_bytes: int | None
    gpu_transfer_active: bool
    error: tuple[str, str] | None


def dataset_length(dataset: Dataset) -> int:
    return map_dataset_length(dataset)


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


def plan_from_template(
    template: LoaderTemplate,
    max_batches: int | None,
) -> BatchPlan:
    total_batches = template.expected_batches()
    if total_batches < 2:
        raise DatasetError(
            f"the loader produces only {total_batches} batch; at least two batches are required"
        )
    if max_batches is not None and max_batches < 2:
        raise ConfigurationError("max_batches must be at least 2 or None")
    batches = total_batches if max_batches is None else min(total_batches, max_batches)
    return BatchPlan(batches=batches, steady_samples=0)


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
    seed: int = 42,
    sample_count_fn: Callable[[Any], int] | None = None,
) -> ConfigMeasurement:
    template = LoaderTemplate.from_dataset(
        dataset,
        batch_size,
        sample_count_fn=sample_count_fn,
    )
    stage = benchmark_rounds(
        template,
        (config,),
        phase=phase,
        device=device,
        plan=plan,
        warmup_rounds=warmup_runs,
        rounds=repetitions,
        max_ram_bytes=max_ram_bytes,
        seed=seed,
    )
    return stage.measurements[0]


def benchmark_rounds(
    template: LoaderTemplate,
    configs: Sequence[LoaderConfig],
    *,
    phase: str,
    device: torch.device,
    plan: BatchPlan,
    warmup_rounds: int,
    rounds: int,
    max_ram_bytes: int | None,
    seed: int,
    deadline: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> StageResult:
    if not configs:
        raise ConfigurationError("at least one configuration is required")
    if warmup_rounds < 0:
        raise ConfigurationError("warmup_rounds must be non-negative")
    if rounds < 1:
        raise ConfigurationError("rounds must be at least 1")

    unique_configs = tuple(dict.fromkeys(configs))
    accumulators = {config: _Accumulator() for config in unique_configs}
    active = set(unique_configs)
    orders: list[RoundOrder] = []

    warmup_orders = balanced_orders(
        unique_configs,
        warmup_rounds,
        seed=seed,
        phase=phase,
        measured=False,
    )
    for warmup_order in warmup_orders:
        executed: list[LoaderConfig] = []
        for config in warmup_order.configs:
            if config not in active:
                continue
            executed.append(config)
            _emit(
                progress,
                f"  warmup {warmup_order.round_index + 1}/{warmup_rounds}: {config.describe()}",
            )
            execution = _execute_pass(
                template,
                config,
                device=device,
                plan=plan,
                max_ram_bytes=max_ram_bytes,
            )
            _record_warmup(accumulators[config], execution)
            if execution.error is not None:
                accumulators[config].failure = execution.error
                active.discard(config)
                _emit(progress, f"    failed: {execution.error[0]}: {execution.error[1]}")
        orders.append(
            RoundOrder(
                phase=phase,
                round_index=warmup_order.round_index,
                configs=tuple(executed),
                measured=False,
            )
        )

    active_configs = tuple(config for config in unique_configs if config in active)
    planned_orders = balanced_orders(
        active_configs,
        rounds,
        seed=seed,
        phase=f"{phase}:measured",
        measured=True,
    )
    budget_exhausted = False
    for measured_order in planned_orders:
        if not active:
            break
        if (
            deadline is not None
            and measured_order.round_index > 0
            and time.perf_counter() >= deadline
        ):
            budget_exhausted = True
            break
        executed = []
        for config in measured_order.configs:
            if config not in active:
                continue
            executed.append(config)
            _emit(
                progress,
                f"  round {measured_order.round_index + 1}/{rounds} "
                f"[{len(executed)}/{len(measured_order.configs)}]: {config.describe()}",
            )
            execution = _execute_pass(
                template,
                config,
                device=device,
                plan=plan,
                max_ram_bytes=max_ram_bytes,
            )
            _record_measured(
                accumulators[config],
                execution,
                round_index=measured_order.round_index,
                position=len(executed) - 1,
            )
            if execution.error is not None:
                accumulators[config].failure = execution.error
                active.discard(config)
                _emit(progress, f"    failed: {execution.error[0]}: {execution.error[1]}")
            elif execution.pass_metrics is not None:
                _emit(progress, f"    {execution.pass_metrics.throughput:,.1f} samples/s")
        orders.append(
            RoundOrder(
                phase=phase,
                round_index=measured_order.round_index,
                configs=tuple(executed),
                measured=True,
            )
        )

    measurements = tuple(
        _finalize_measurement(config, phase, accumulator, max_ram_bytes)
        for config, accumulator in accumulators.items()
    )
    return StageResult(
        measurements=measurements,
        orders=tuple(orders),
        budget_exhausted=budget_exhausted,
    )


def _execute_pass(
    template: LoaderTemplate,
    config: LoaderConfig,
    *,
    device: torch.device,
    plan: BatchPlan,
    max_ram_bytes: int | None,
) -> _PassExecution:
    monitor = MemoryMonitor(max_ram_bytes)
    loader: DataLoader | None = None
    gpu_stats_reset = False
    gpu_peak: int | None = None
    pass_metrics: PassMetrics | None = None
    cold_start: float | None = None
    gpu_transfer = False
    failure: tuple[str, str] | None = None

    try:
        monitor.start()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            gpu_stats_reset = True

        construction_start = time.perf_counter()
        loader = template.build(config)
        construction_s = time.perf_counter() - construction_start
        monitor.raise_if_exceeded()
        pass_metrics = _run_pass(loader, template=template, plan=plan, device=device)
        monitor.raise_if_exceeded()
        gpu_transfer = pass_metrics.gpu_transfer_active
        cold_start = construction_s + pass_metrics.startup_s
    except Exception as exc:
        failure = (type(exc).__name__, _describe_exception(exc))
    finally:
        if loader is not None:
            _shutdown_loader(loader)
        snapshot = monitor.stop()
        if gpu_stats_reset:
            gpu_peak = _gpu_peak_bytes(device)

    return _PassExecution(
        pass_metrics=pass_metrics,
        cold_start_s=cold_start,
        memory=snapshot,
        gpu_peak_bytes=gpu_peak,
        gpu_transfer_active=gpu_transfer,
        error=failure,
    )


def _run_pass(
    loader: DataLoader,
    *,
    template: LoaderTemplate,
    plan: BatchPlan,
    device: torch.device,
) -> PassMetrics:
    transfer = device.type == "cuda"
    if transfer:
        torch.cuda.synchronize(device)

    startup_start = time.perf_counter()
    iterator = iter(loader)
    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        raise DatasetError("the DataLoader produced no batches") from exc
    transfer_active = False
    if transfer:
        moved, moved_any = _move_to_device(first_batch, device)
        first_batch = moved
        transfer_active = transfer_active or moved_any
    template.count_samples(first_batch)
    startup_s = time.perf_counter() - startup_start

    latencies: list[float] = []
    steady_samples = 0
    steady_start = time.perf_counter()
    for _ in range(1, plan.batches):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise DatasetError(
                f"the DataLoader produced fewer than the planned {plan.batches} batches"
            ) from exc
        if transfer:
            moved, moved_any = _move_to_device(batch, device)
            batch = moved
            transfer_active = transfer_active or moved_any
        steady_samples += template.count_samples(batch)
        latencies.append(time.perf_counter() - batch_start)
    if transfer:
        torch.cuda.synchronize(device)
    elapsed_s = max(time.perf_counter() - steady_start, 1e-9)

    return PassMetrics(
        startup_s=startup_s,
        steady_elapsed_s=elapsed_s,
        batches=plan.batches,
        steady_samples=steady_samples,
        throughput=steady_samples / elapsed_s,
        latencies_s=tuple(latencies),
        gpu_transfer_active=transfer_active,
    )


def _record_warmup(accumulator: _Accumulator, execution: _PassExecution) -> None:
    accumulator.memories.append(execution.memory)
    if execution.cold_start_s is not None and accumulator.cold_start_s is None:
        accumulator.cold_start_s = execution.cold_start_s
    if execution.gpu_peak_bytes is not None:
        accumulator.gpu_peaks.append(execution.gpu_peak_bytes)
    accumulator.gpu_transfer_active = (
        accumulator.gpu_transfer_active or execution.gpu_transfer_active
    )


def _record_measured(
    accumulator: _Accumulator,
    execution: _PassExecution,
    *,
    round_index: int,
    position: int,
) -> None:
    _record_warmup(accumulator, execution)
    if execution.pass_metrics is None:
        return
    accumulator.rounds.append(
        RoundMeasurement(
            round_index=round_index,
            position=position,
            pass_metrics=execution.pass_metrics,
            cold_start_s=execution.cold_start_s,
            memory=execution.memory,
            gpu_peak_bytes=execution.gpu_peak_bytes,
            gpu_transfer_active=execution.gpu_transfer_active,
        )
    )


def _finalize_measurement(
    config: LoaderConfig,
    phase: str,
    accumulator: _Accumulator,
    max_ram_bytes: int | None,
) -> ConfigMeasurement:
    memory = _merge_memory(accumulator.memories, max_ram_bytes)
    passes = tuple(item.pass_metrics for item in accumulator.rounds)
    rounds = tuple(accumulator.rounds)
    gpu_peak = max(accumulator.gpu_peaks) if accumulator.gpu_peaks else None
    if accumulator.failure is None and passes:
        summary = summarize_passes(
            passes,
            peak_process_rss_bytes=memory.peak_process_rss_bytes if memory else None,
            peak_system_used_bytes=memory.peak_system_used_bytes if memory else None,
            peak_gpu_memory_bytes=gpu_peak,
        )
        return ConfigMeasurement(
            config=config,
            phase=phase,
            status="ok",
            passes=passes,
            rounds=rounds,
            summary=summary,
            cold_start_s=accumulator.cold_start_s,
            gpu_transfer_active=accumulator.gpu_transfer_active,
            memory=memory,
        )

    error_type, error_message = accumulator.failure or (
        "BenchmarkError",
        "no measured round completed",
    )
    return ConfigMeasurement(
        config=config,
        phase=phase,
        status="failed",
        error_type=error_type,
        error_message=error_message,
        passes=passes,
        rounds=rounds,
        cold_start_s=accumulator.cold_start_s,
        gpu_transfer_active=accumulator.gpu_transfer_active,
        memory=memory,
    )


def _merge_memory(
    snapshots: Sequence[MemorySnapshot],
    limit_bytes: int | None,
) -> MemorySnapshot | None:
    if not snapshots:
        return None
    baseline = next(
        (
            snapshot.baseline_available_bytes
            for snapshot in snapshots
            if snapshot.baseline_available_bytes is not None
        ),
        None,
    )
    process_rss = [
        snapshot.peak_process_rss_bytes
        for snapshot in snapshots
        if snapshot.peak_process_rss_bytes is not None
    ]
    system_used = [
        snapshot.peak_system_used_bytes
        for snapshot in snapshots
        if snapshot.peak_system_used_bytes is not None
    ]
    return MemorySnapshot(
        baseline_available_bytes=baseline,
        peak_process_rss_bytes=max(process_rss) if process_rss else None,
        peak_system_used_bytes=max(system_used) if system_used else None,
        limit_bytes=limit_bytes,
        limit_exceeded=any(snapshot.limit_exceeded for snapshot in snapshots),
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


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
