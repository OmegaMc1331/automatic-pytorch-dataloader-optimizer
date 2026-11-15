from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from apdo.benchmark import ConfigMeasurement
from apdo.config import BenchmarkSettings, LoaderConfig
from apdo.metrics import PassMetrics, summarize_passes
from apdo.report import OptimizationResult
from apdo.resources import EnvironmentInfo, MemorySnapshot


def make_measurement(
    config: LoaderConfig,
    throughputs: tuple[float, ...],
    *,
    ci_low: float | None = None,
    ci_high: float | None = None,
    phase: str = "test",
) -> ConfigMeasurement:
    passes = tuple(
        PassMetrics(
            startup_s=0.02 + index * 0.001,
            steady_elapsed_s=1.0,
            batches=3,
            steady_samples=throughput,
            throughput=throughput,
            latencies_s=(0.01, 0.02),
            gpu_transfer_active=False,
        )
        for index, throughput in enumerate(throughputs)
    )
    summary = summarize_passes(
        passes,
        peak_process_rss_bytes=64 * 1024**2,
        peak_system_used_bytes=32 * 1024**2,
        peak_gpu_memory_bytes=None,
    )
    if ci_low is not None:
        summary = replace(summary, ci_low_throughput=ci_low)
    if ci_high is not None:
        summary = replace(summary, ci_high_throughput=ci_high)
    return ConfigMeasurement(
        config=config,
        phase=phase,
        status="ok",
        passes=passes,
        summary=summary,
        cold_start_s=0.1,
        memory=MemorySnapshot(
            baseline_available_bytes=8 * 1024**3,
            peak_process_rss_bytes=64 * 1024**2,
            peak_system_used_bytes=32 * 1024**2,
            limit_bytes=2_000_000_000,
            limit_exceeded=False,
        ),
    )


def make_failed_measurement(
    config: LoaderConfig,
    message: str = "worker failed",
) -> ConfigMeasurement:
    return ConfigMeasurement(
        config=config,
        phase="test",
        status="failed",
        error_type="RuntimeError",
        error_message=message,
    )


def make_environment() -> EnvironmentInfo:
    return EnvironmentInfo(
        python_version="3.12.0",
        python_implementation="CPython",
        platform="TestOS-1.0",
        machine="x86_64",
        processor="Test CPU",
        cpu_count=8,
        cpu_affinity_count=8,
        torch_version="2.10.0",
        cuda_available=False,
        cuda_version=None,
        cuda_devices=(),
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=8 * 1024**3,
        torch_num_threads=8,
    )


def make_settings() -> BenchmarkSettings:
    return BenchmarkSettings(
        batch_size=8,
        requested_device="cpu",
        device="cpu",
        warmup_runs=1,
        repetitions=3,
        max_batches=10,
        max_workers=4,
        max_ram_gb=2.0,
        dataset_size=64,
        cpu_count=8,
    )


def make_result(
    measurements: tuple[ConfigMeasurement, ...],
    best_config: LoaderConfig,
    fastest_config: LoaderConfig,
    *,
    warnings: tuple[str, ...] = (),
) -> OptimizationResult:
    return OptimizationResult(
        best_config=best_config,
        fastest_config=fastest_config,
        measurements=measurements,
        environment=make_environment(),
        settings=make_settings(),
        warnings=warnings,
        selection_reason="Test selection reason.",
        baseline_throughput=80.0,
        improvement_percent=25.0,
        started_at=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        duration_s=1.25,
    )
