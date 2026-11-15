from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PassMetrics:
    startup_s: float
    steady_elapsed_s: float
    batches: int
    steady_samples: int
    throughput: float
    latencies_s: tuple[float, ...]
    gpu_transfer_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "startup_s": self.startup_s,
            "steady_elapsed_s": self.steady_elapsed_s,
            "batches": self.batches,
            "steady_samples": self.steady_samples,
            "throughput": self.throughput,
            "median_latency_s": median(self.latencies_s) if self.latencies_s else None,
            "p95_latency_s": percentile(self.latencies_s, 0.95) if self.latencies_s else None,
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
    )
