from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CycleMetrics:
    cycle_index: int
    startup_s: float
    duration_s: float
    batches: int
    samples: int
    throughput: float
    steady_elapsed_s: float
    steady_samples: int
    steady_throughput: float
    latencies_s: tuple[float, ...]
    worker_pids: tuple[int, ...] = ()
    rss_bytes: int | None = None
    data_wait_s: float | None = None
    transfer_s: float | None = None
    transfer_bytes: int | None = None
    transfer_active: bool = False
    step_latencies_s: tuple[float, ...] = ()
    first_batch_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_index": self.cycle_index,
            "startup_s": self.startup_s,
            "duration_s": self.duration_s,
            "batches": self.batches,
            "samples": self.samples,
            "throughput": self.throughput,
            "steady_elapsed_s": self.steady_elapsed_s,
            "steady_samples": self.steady_samples,
            "steady_throughput": self.steady_throughput,
            "median_latency_s": median(self.latencies_s) if self.latencies_s else None,
            "p95_latency_s": percentile(self.latencies_s, 0.95) if self.latencies_s else None,
            "worker_pids": list(self.worker_pids),
            "rss_bytes": self.rss_bytes,
            "data_wait_s": self.data_wait_s,
            "transfer_s": self.transfer_s,
            "transfer_bytes": self.transfer_bytes,
            "transfer_active": self.transfer_active,
            "first_batch_samples": self.first_batch_samples,
            "median_step_latency_s": (
                median(self.step_latencies_s) if self.step_latencies_s else None
            ),
        }


@dataclass(frozen=True, slots=True)
class PassMetrics:
    startup_s: float
    steady_elapsed_s: float
    batches: int
    steady_samples: int
    throughput: float
    latencies_s: tuple[float, ...]
    gpu_transfer_active: bool = False
    construction_s: float | None = None
    training_setup_s: float | None = None
    total_elapsed_s: float | None = None
    sustained_throughput: float | None = None
    cycles: tuple[CycleMetrics, ...] = ()
    expected_batches: int | None = None
    full_epoch: bool = True
    training: bool = False
    data_wait_s: float | None = None
    transfer_s: float | None = None
    transfer_bytes: int | None = None
    worker_startup_overhead_s: float | None = None
    average_epoch_transition_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        step_latencies = [value for cycle in self.cycles for value in cycle.step_latencies_s]
        return {
            "startup_s": self.startup_s,
            "steady_elapsed_s": self.steady_elapsed_s,
            "batches": self.batches,
            "steady_samples": self.steady_samples,
            "throughput": self.throughput,
            "median_latency_s": median(self.latencies_s) if self.latencies_s else None,
            "p95_latency_s": percentile(self.latencies_s, 0.95) if self.latencies_s else None,
            "construction_s": self.construction_s,
            "training_setup_s": self.training_setup_s,
            "total_elapsed_s": self.total_elapsed_s,
            "sustained_throughput": self.sustained_throughput,
            "expected_batches": self.expected_batches,
            "full_epoch": self.full_epoch,
            "training": self.training,
            "data_wait_s": self.data_wait_s,
            "transfer_s": self.transfer_s,
            "transfer_bytes": self.transfer_bytes,
            "worker_startup_overhead_s": self.worker_startup_overhead_s,
            "average_epoch_transition_s": self.average_epoch_transition_s,
            "median_step_latency_s": median(step_latencies) if step_latencies else None,
            "cycles": [cycle.to_dict() for cycle in self.cycles],
        }


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    median_throughput: float
    mean_throughput: float
    stdev_throughput: float
    min_throughput: float
    max_throughput: float
    cv_throughput: float
    ci_low_throughput: float
    ci_high_throughput: float
    median_latency_s: float
    p95_latency_s: float
    mean_latency_s: float
    median_startup_s: float
    peak_process_rss_bytes: int | None
    peak_system_used_bytes: int | None
    peak_gpu_memory_bytes: int | None
    median_sustained_throughput: float | None = None
    median_first_cycle_throughput: float | None = None
    median_subsequent_cycle_throughput: float | None = None
    median_worker_startup_overhead_s: float | None = None
    median_epoch_transition_s: float | None = None
    median_step_latency_s: float | None = None
    median_data_wait_s: float | None = None
    median_transfer_s: float | None = None
    total_transfer_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_throughput": self.median_throughput,
            "mean_throughput": self.mean_throughput,
            "stdev_throughput": self.stdev_throughput,
            "min_throughput": self.min_throughput,
            "max_throughput": self.max_throughput,
            "cv_throughput": self.cv_throughput,
            "ci_low_throughput": self.ci_low_throughput,
            "ci_high_throughput": self.ci_high_throughput,
            "median_latency_s": self.median_latency_s,
            "p95_latency_s": self.p95_latency_s,
            "mean_latency_s": self.mean_latency_s,
            "median_startup_s": self.median_startup_s,
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
            "peak_system_used_bytes": self.peak_system_used_bytes,
            "peak_gpu_memory_bytes": self.peak_gpu_memory_bytes,
            "median_sustained_throughput": self.median_sustained_throughput,
            "median_first_cycle_throughput": self.median_first_cycle_throughput,
            "median_subsequent_cycle_throughput": self.median_subsequent_cycle_throughput,
            "median_worker_startup_overhead_s": self.median_worker_startup_overhead_s,
            "median_epoch_transition_s": self.median_epoch_transition_s,
            "median_step_latency_s": self.median_step_latency_s,
            "median_data_wait_s": self.median_data_wait_s,
            "median_transfer_s": self.median_transfer_s,
            "total_transfer_bytes": self.total_transfer_bytes,
        }


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def median(values: Sequence[float]) -> float:
    return percentile(values, 0.5)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(values) / len(values)


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = math.fsum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap_median_ci requires at least one value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if len(values) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        bootstrapped.append(median(sample))

    alpha = (1.0 - confidence) / 2.0
    return percentile(bootstrapped, alpha), percentile(bootstrapped, 1.0 - alpha)


def summarize_passes(
    passes: Sequence[PassMetrics],
    *,
    peak_process_rss_bytes: int | None = None,
    peak_system_used_bytes: int | None = None,
    peak_gpu_memory_bytes: int | None = None,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> MetricsSummary:
    if not passes:
        raise ValueError("summarize_passes requires at least one pass")

    throughputs = [item.throughput for item in passes]
    latencies = [latency for item in passes for latency in item.latencies_s]
    startups = [item.startup_s for item in passes]
    average = mean(throughputs)
    standard_deviation = stdev(throughputs)
    ci_low, ci_high = bootstrap_median_ci(
        throughputs,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    step_latencies = [
        value for item in passes for cycle in item.cycles for value in cycle.step_latencies_s
    ]
    data_waits = [value for item in passes for value in (item.data_wait_s,) if value is not None]
    transfers = [value for item in passes for value in (item.transfer_s,) if value is not None]
    transfer_bytes = [
        value for item in passes for value in (item.transfer_bytes,) if value is not None
    ]
    return MetricsSummary(
        median_throughput=median(throughputs),
        mean_throughput=average,
        stdev_throughput=standard_deviation,
        min_throughput=min(throughputs),
        max_throughput=max(throughputs),
        cv_throughput=standard_deviation / average if average > 0 else 0.0,
        ci_low_throughput=ci_low,
        ci_high_throughput=ci_high,
        median_latency_s=median(latencies),
        p95_latency_s=percentile(latencies, 0.95),
        mean_latency_s=mean(latencies),
        median_startup_s=median(startups),
        peak_process_rss_bytes=peak_process_rss_bytes,
        peak_system_used_bytes=peak_system_used_bytes,
        peak_gpu_memory_bytes=peak_gpu_memory_bytes,
        median_sustained_throughput=_median_optional(
            [item.sustained_throughput for item in passes]
        ),
        median_first_cycle_throughput=_median_optional(
            [item.cycles[0].throughput if item.cycles else item.throughput for item in passes]
        ),
        median_subsequent_cycle_throughput=_median_optional(
            [_subsequent_cycle_throughput(item) for item in passes]
        ),
        median_worker_startup_overhead_s=_median_optional(
            [item.worker_startup_overhead_s for item in passes]
        ),
        median_epoch_transition_s=_median_optional(
            [item.average_epoch_transition_s for item in passes]
        ),
        median_step_latency_s=_median_optional(step_latencies),
        median_data_wait_s=_median_optional(data_waits),
        median_transfer_s=_median_optional(transfers),
        total_transfer_bytes=sum(transfer_bytes) if transfer_bytes else None,
    )


def _subsequent_cycle_throughput(metrics: PassMetrics) -> float | None:
    cycles = metrics.cycles[1:]
    if not cycles:
        return None
    samples = sum(cycle.samples for cycle in cycles)
    duration = sum(cycle.duration_s for cycle in cycles)
    return samples / duration if duration > 0 else None


def _median_optional(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return median(present) if present else None
