from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch
from torch.utils.data import Dataset

from apdo.benchmark import (
    BatchPlan,
    ConfigMeasurement,
    benchmark_config,
    dataset_length,
    plan_batches,
)
from apdo.config import BenchmarkSettings, LoaderConfig
from apdo.errors import BenchmarkError, ConfigurationError
from apdo.metrics import MetricsSummary
from apdo.report import OptimizationResult
from apdo.resources import available_cpu_count, collect_environment

MATERIAL_BAND = 0.95
DEFAULT_MAX_WORKERS = 8
FINALIST_WORKERS = 2
PIN_CANDIDATES = (False, True)
PREFETCH_CANDIDATES = (2, 4)
PERSISTENT_CANDIDATES = (False, True)


@dataclass(frozen=True, slots=True)
class _Selection:
    selected: ConfigMeasurement
    fastest: ConfigMeasurement
    equivalent: tuple[ConfigMeasurement, ...]
    reason: str


def optimize(
    dataset: Dataset,
    batch_size: int = 32,
    device: str | torch.device = "auto",
    max_ram_gb: float | None = None,
    *,
    warmup_runs: int = 1,
    repetitions: int = 3,
    max_batches: int | None = 100,
    max_workers: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationResult:
    batch_size = _positive_int(batch_size, "batch_size")
    warmup_runs = _positive_int(warmup_runs, "warmup_runs")
    repetitions = _positive_int(repetitions, "repetitions")
    if repetitions < 2:
        raise ConfigurationError("repetitions must be at least 2")
    if max_batches is not None:
        max_batches = _positive_int(max_batches, "max_batches")
        if max_batches < 2:
            raise ConfigurationError("max_batches must be at least 2 or None")
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    if progress is not None and not callable(progress):
        raise ConfigurationError("progress must be callable or None")
    max_ram_value = _positive_float(max_ram_gb, "max_ram_gb")

    requested_device = str(device)
    resolved_device = resolve_device(device)
    size = dataset_length(dataset)
    plan = plan_batches(size, batch_size, max_batches)

    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    _emit(progress, "Collecting environment information")
    environment = collect_environment()
    cpu_count = available_cpu_count()
    resolved_max_workers = _resolve_max_workers(max_workers, cpu_count)
    settings = BenchmarkSettings(
        batch_size=batch_size,
        requested_device=requested_device,
        device=str(resolved_device),
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        max_batches=max_batches,
        max_workers=resolved_max_workers,
        max_ram_gb=max_ram_value,
        dataset_size=size,
        cpu_count=cpu_count,
    )

    warnings: list[str] = []
    if plan.batches < 4:
        warnings.append(
            f"Only {plan.batches} batches are processed per pass; results may be unstable."
        )
    elif max_batches is not None and max_batches < 10:
        warnings.append(
            f"Only {max_batches} batches are processed per pass; "
            "increase max_batches for steadier results."
        )
    if max_workers is not None and max_workers > cpu_count:
        warnings.append(
            f"max_workers={max_workers} exceeds the {cpu_count} available CPUs; "
            f"the search is limited to {resolved_max_workers} workers."
        )
    if max_ram_value is not None and environment.available_memory_bytes is None:
        warnings.append(
            "Available system memory could not be read; max_ram_gb cannot be enforced "
            "on this platform."
        )

    max_ram_bytes = int(max_ram_value * 1_000_000_000) if max_ram_value is not None else None
    pin_options = _pin_options(resolved_device)
    worker_counts = _build_worker_counts(resolved_max_workers, cpu_count)
    stage_one = _stage_one_configs(worker_counts, pin_options)

    measurements = _run_stage(
        dataset=dataset,
        configs=stage_one,
        phase="worker_scan",
        progress=progress,
        batch_size=batch_size,
        device=resolved_device,
        plan=plan,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
        max_ram_bytes=max_ram_bytes,
        stage_number=1,
        stage_total=2,
    )
    stage_one_successes = [item for item in measurements if item.succeeded]
    if not stage_one_successes:
        detail = "\n".join(_failure_lines(measurements))
        raise BenchmarkError("No DataLoader configuration completed successfully.\n" + detail)

    finalist_workers = _finalist_workers(stage_one_successes)
    seen = {item.config for item in measurements}
    stage_two: list[LoaderConfig] = []
    for workers in finalist_workers:
        for pin_memory in pin_options:
            for prefetch_factor in PREFETCH_CANDIDATES:
                for persistent_workers in PERSISTENT_CANDIDATES:
                    config = LoaderConfig(
                        num_workers=workers,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers,
                    )
                    if config not in seen:
                        seen.add(config)
                        stage_two.append(config)

    if stage_two:
        measurements.extend(
            _run_stage(
                dataset=dataset,
                configs=stage_two,
                phase="flag_sweep",
                progress=progress,
                batch_size=batch_size,
                device=resolved_device,
                plan=plan,
                warmup_runs=warmup_runs,
                repetitions=repetitions,
                max_ram_bytes=max_ram_bytes,
                stage_number=2,
                stage_total=2,
            )
        )
    else:
        _emit(progress, "Stage 2/2: no additional multi-worker configurations to test")

    _emit(progress, "Selecting a recommendation")
    selection = _select_configuration(measurements)
    successes = [item for item in measurements if item.succeeded]
    baseline = next(
        (item for item in successes if item.config == LoaderConfig()),
        None,
    )
    selected_summary = _summary(selection.selected)
    baseline_throughput = _summary(baseline).median_throughput if baseline else None
    improvement_percent: float | None = None
    if baseline_throughput is not None and baseline_throughput > 0:
        improvement_percent = (
            selected_summary.median_throughput / baseline_throughput - 1.0
        ) * 100.0

    failures = [item for item in measurements if not item.succeeded]
    if failures:
        warnings.append(
            f"{len(failures)} of {len(measurements)} configurations failed; "
            "see the report for details."
        )
    if len(successes) == 1:
        warnings.append(
            "Only one configuration completed successfully; no comparative "
            "recommendation could be made."
        )
    if (
        resolved_device.type == "cuda"
        and successes
        and not any(item.gpu_transfer_active for item in successes)
    ):
        warnings.append(
            "Batches contained no tensor values, so host-to-device transfer was not "
            "exercised; pin_memory results may not be representative."
        )

    duration_s = time.perf_counter() - start_time
    _emit(progress, f"Benchmark completed in {duration_s:.2f} s")
    return OptimizationResult(
        best_config=selection.selected.config,
        fastest_config=selection.fastest.config,
        measurements=tuple(measurements),
        environment=environment,
        settings=settings,
        warnings=tuple(warnings),
        selection_reason=selection.reason,
        baseline_throughput=baseline_throughput,
        improvement_percent=improvement_percent,
        started_at=started_at,
        duration_s=duration_s,
    )


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str) and device.strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid device {device!r}: {exc}") from exc
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise ConfigurationError("CUDA was requested but torch.cuda.is_available() is False")
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ConfigurationError(
                f"CUDA device index {resolved.index} is unavailable; "
                f"torch.cuda.device_count() is {torch.cuda.device_count()}"
            )
    elif resolved.type != "cpu":
        raise ConfigurationError(
            f"device type {resolved.type!r} is not supported in v0.1.0; "
            "use 'auto', 'cpu', or 'cuda'"
        )
    return resolved


def _run_stage(
    *,
    dataset: Dataset,
    configs: Sequence[LoaderConfig],
    phase: str,
    progress: Callable[[str], None] | None,
    batch_size: int,
    device: torch.device,
    plan: BatchPlan,
    warmup_runs: int,
    repetitions: int,
    max_ram_bytes: int | None,
    stage_number: int,
    stage_total: int,
) -> list[ConfigMeasurement]:
    total = len(configs)
    _emit(
        progress,
        f"Stage {stage_number}/{stage_total}: {phase} with {total} configurations",
    )
    measurements: list[ConfigMeasurement] = []
    for index, config in enumerate(configs, start=1):
        _emit(progress, f"[{index}/{total}] {config.describe()}")
        measurement = benchmark_config(
            dataset,
            config,
            phase=phase,
            batch_size=batch_size,
            device=device,
            plan=plan,
            warmup_runs=warmup_runs,
            repetitions=repetitions,
            max_ram_bytes=max_ram_bytes,
        )
        measurements.append(measurement)
        if measurement.succeeded:
            summary = _summary(measurement)
            _emit(progress, f"    {summary.median_throughput:,.1f} samples/s")
        else:
            _emit(
                progress,
                f"    failed: {measurement.error_type}: {measurement.error_message}",
            )
    return measurements


def _build_worker_counts(max_workers: int, cpu_count: int) -> list[int]:
    limit = max(1, min(max_workers, cpu_count))
    counts = {0, 1}
    for candidate in (2, 4, 8, limit):
        if candidate <= limit:
            counts.add(candidate)
    return sorted(counts)


def _stage_one_configs(
    worker_counts: Sequence[int],
    pin_options: Sequence[bool],
) -> list[LoaderConfig]:
    configs = [LoaderConfig()]
    for workers in worker_counts:
        if workers == 0:
            continue
        for pin_memory in pin_options:
            configs.append(
                LoaderConfig(
                    num_workers=workers,
                    pin_memory=pin_memory,
                    prefetch_factor=2,
                    persistent_workers=False,
                )
            )
    return configs


def _finalist_workers(measurements: Sequence[ConfigMeasurement]) -> list[int]:
    best_by_worker: dict[int, ConfigMeasurement] = {}
    for measurement in measurements:
        workers = measurement.config.num_workers
        if workers == 0 or measurement.summary is None:
            continue
        current = best_by_worker.get(workers)
        if current is None or (
            _summary(measurement).median_throughput > _summary(current).median_throughput
        ):
            best_by_worker[workers] = measurement
    ranked = sorted(
        best_by_worker.values(),
        key=lambda item: _summary(item).median_throughput,
        reverse=True,
    )
    return [item.config.num_workers for item in ranked[:FINALIST_WORKERS]]


def _select_configuration(
    measurements: Sequence[ConfigMeasurement],
) -> _Selection:
    successful = [item for item in measurements if item.succeeded and item.summary]
    if not successful:
        raise BenchmarkError("no successful configuration is available for selection")

    fastest = max(
        successful,
        key=lambda item: _summary(item).median_throughput,
    )
    fastest_summary = _summary(fastest)
    equivalent = [
        item
        for item in successful
        if _summary(item).median_throughput >= fastest_summary.median_throughput * MATERIAL_BAND
        and _summary(item).ci_low_throughput <= fastest_summary.ci_high_throughput
        and fastest_summary.ci_low_throughput <= _summary(item).ci_high_throughput
    ]
    selected = min(equivalent, key=lambda item: item.config.simplicity_key())

    if selected.config == fastest.config:
        reason = (
            "The fastest measured configuration was selected. No simpler configuration "
            "was both within 5% of its median throughput and had a 95% bootstrap "
            "confidence interval that overlapped the fastest interval."
        )
    else:
        reason = (
            f"The fastest measured configuration was {fastest.config.describe()}. "
            f"{selected.config.describe()} was within 5% of its median throughput and "
            "had a 95% bootstrap confidence interval that overlapped the fastest "
            "interval, so the simpler configuration was preferred."
        )
    return _Selection(
        selected=selected,
        fastest=fastest,
        equivalent=tuple(equivalent),
        reason=reason,
    )


def _resolve_max_workers(requested: int | None, cpu_count: int) -> int:
    if requested is None:
        return max(1, min(DEFAULT_MAX_WORKERS, cpu_count))
    return max(1, min(requested, cpu_count))


def _pin_options(device: torch.device) -> tuple[bool, ...]:
    return PIN_CANDIDATES if device.type == "cuda" else (False,)


def _summary(measurement: ConfigMeasurement | None) -> MetricsSummary:
    if measurement is None or measurement.summary is None:
        raise BenchmarkError("configuration has no completed measurements")
    return measurement.summary


def _failure_lines(measurements: Sequence[ConfigMeasurement]) -> list[str]:
    lines = []
    for measurement in measurements:
        if not measurement.succeeded:
            lines.append(
                f"  {measurement.config.describe()}: "
                f"{measurement.error_type}: {measurement.error_message}"
            )
    return lines


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if value < 1:
        raise ConfigurationError(f"{name} must be at least 1")
    return value


def _positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a positive number or None")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ConfigurationError(f"{name} must be a positive finite number or None")
    return converted


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
