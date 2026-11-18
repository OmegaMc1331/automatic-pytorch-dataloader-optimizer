from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from apdo.comparability import BatchObservations, batch_fingerprint, batch_structure
from apdo.config import MAX_BENCHMARK_EPOCHS, TIMING_PROFILES, LoaderConfig
from apdo.errors import ConfigurationError, DatasetError, FingerprintError
from apdo.isolation import isolated_rng
from apdo.loader import LoaderSource, LoaderTemplate, map_dataset_length
from apdo.metrics import (
    CycleMetrics,
    MetricsSummary,
    PassMetrics,
    summarize_passes,
)
from apdo.order import RoundOrder, balanced_orders, derive_seed
from apdo.resources import MemoryMonitor, MemorySnapshot, process_rss_bytes
from apdo.training import TrainingStep, TrainingStepFactory, create_training_step

_clock = time.perf_counter


@dataclass(frozen=True, slots=True)
class BatchPlan:
    batches: int
    steady_samples: int
    expected_batches: int = 0
    full_epoch: bool = True


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
    epochs: int = 1
    training: bool = False

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
            "epochs": self.epochs,
            "training": self.training,
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
    return BatchPlan(
        batches=batches,
        steady_samples=steady_samples,
        expected_batches=total_batches,
        full_epoch=batches == total_batches,
    )


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
    return BatchPlan(
        batches=batches,
        steady_samples=0,
        expected_batches=total_batches,
        full_epoch=batches == total_batches,
    )


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
    epochs: int = 1,
    training_step_factory: TrainingStepFactory | None = None,
    timing_profile: str = "standard",
    fingerprint_fn: Callable[[Any], Any] | None = None,
    collect_comparability: bool = True,
) -> ConfigMeasurement:
    template = LoaderTemplate.from_dataset(
        dataset,
        batch_size,
        sample_count_fn=sample_count_fn,
    )
    source = LoaderSource.from_template(template, "dataloader")
    stage = benchmark_source(
        source,
        (config,),
        phase=phase,
        device=device,
        plan=plan,
        warmup_rounds=warmup_runs,
        rounds=repetitions,
        max_ram_bytes=max_ram_bytes,
        seed=seed,
        epochs=epochs,
        training_step_factory=training_step_factory,
        timing_profile=timing_profile,
        fingerprint_fn=fingerprint_fn,
        collect_comparability=collect_comparability,
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
    epochs: int = 1,
    training_step_factory: TrainingStepFactory | None = None,
    timing_profile: str = "standard",
    fingerprint_fn: Callable[[Any], Any] | None = None,
    collect_comparability: bool = True,
) -> StageResult:
    source = LoaderSource.from_template(template, "dataloader")
    return benchmark_source(
        source,
        configs,
        phase=phase,
        device=device,
        plan=plan,
        warmup_rounds=warmup_rounds,
        rounds=rounds,
        max_ram_bytes=max_ram_bytes,
        seed=seed,
        deadline=deadline,
        progress=progress,
        epochs=epochs,
        training_step_factory=training_step_factory,
        timing_profile=timing_profile,
        fingerprint_fn=fingerprint_fn,
        collect_comparability=collect_comparability,
    )


def benchmark_source(
    source: LoaderSource,
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
    epochs: int = 1,
    training_step_factory: TrainingStepFactory | None = None,
    timing_profile: str = "standard",
    fingerprint_fn: Callable[[Any], Any] | None = None,
    collect_comparability: bool = True,
) -> StageResult:
    if not configs:
        raise ConfigurationError("at least one configuration is required")
    if warmup_rounds < 0:
        raise ConfigurationError("warmup_rounds must be non-negative")
    if rounds < 1:
        raise ConfigurationError("rounds must be at least 1")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise ConfigurationError("epochs must be an integer")
    if epochs < 1 or epochs > MAX_BENCHMARK_EPOCHS:
        raise ConfigurationError(f"epochs must be between 1 and {MAX_BENCHMARK_EPOCHS}")
    if timing_profile not in TIMING_PROFILES:
        raise ConfigurationError(f"timing_profile must be one of {TIMING_PROFILES}")
    if fingerprint_fn is not None and not callable(fingerprint_fn):
        raise ConfigurationError("sample_fingerprint_fn must be callable or None")
    if not isinstance(collect_comparability, bool):
        raise ConfigurationError("collect_comparability must be a boolean")

    training = training_step_factory is not None
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
        warmup_seed = derive_seed(
            seed,
            phase,
            "warmup",
            str(warmup_order.round_index),
        )
        for config in warmup_order.configs:
            if config not in active:
                continue
            executed.append(config)
            _emit(
                progress,
                f"  warmup {warmup_order.round_index + 1}/{warmup_rounds}: {config.describe()}",
            )
            execution = _execute_trial(
                source,
                config,
                device=device,
                plan=plan,
                max_ram_bytes=max_ram_bytes,
                epochs=1,
                training_step_factory=training_step_factory,
                seed=warmup_seed,
                timing_profile=timing_profile,
                fingerprint_fn=fingerprint_fn,
                collect_comparability=False,
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
        if deadline is not None and measured_order.round_index > 0 and _clock() >= deadline:
            budget_exhausted = True
            break
        executed = []
        trial_seed = derive_seed(
            seed,
            phase,
            "round",
            str(measured_order.round_index),
        )
        for config in measured_order.configs:
            if config not in active:
                continue
            executed.append(config)
            _emit(
                progress,
                f"  round {measured_order.round_index + 1}/{rounds} "
                f"[{len(executed)}/{len(measured_order.configs)}]: {config.describe()}",
            )
            execution = _execute_trial(
                source,
                config,
                device=device,
                plan=plan,
                max_ram_bytes=max_ram_bytes,
                epochs=epochs,
                training_step_factory=training_step_factory,
                seed=trial_seed,
                timing_profile=timing_profile,
                fingerprint_fn=fingerprint_fn,
                collect_comparability=collect_comparability,
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
        _finalize_measurement(
            config,
            phase,
            accumulator,
            max_ram_bytes,
            epochs=epochs,
            training=training,
        )
        for config, accumulator in accumulators.items()
    )
    return StageResult(
        measurements=measurements,
        orders=tuple(orders),
        budget_exhausted=budget_exhausted,
    )


def _execute_trial(
    source: LoaderSource,
    config: LoaderConfig,
    *,
    device: torch.device,
    plan: BatchPlan,
    max_ram_bytes: int | None,
    epochs: int,
    training_step_factory: TrainingStepFactory | None,
    seed: int,
    timing_profile: str,
    fingerprint_fn: Callable[[Any], Any] | None,
    collect_comparability: bool,
) -> _PassExecution:
    monitor = MemoryMonitor(max_ram_bytes)
    loader: DataLoader | None = None
    gpu_stats_reset = False
    gpu_peak: int | None = None
    pass_metrics: PassMetrics | None = None
    cold_start: float | None = None
    transfer_active = False
    failure: tuple[str, str] | None = None
    cleanup_s: float | None = None
    construction_s = 0.0
    training_setup_s: float | None = None
    trial_start = _clock()

    try:
        with isolated_rng(seed, device):
            monitor.start()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                gpu_stats_reset = True

            construction_start = _clock()
            loader = source.build(config, isolate=True)
            construction_s = _clock() - construction_start
            monitor.raise_if_exceeded()

            training_step: TrainingStep | None = None
            if training_step_factory is not None:
                training_setup_start = _clock()
                training_step = create_training_step(training_step_factory)
                training_setup_s = _clock() - training_setup_start
                monitor.raise_if_exceeded()

            pass_metrics = _run_cycles(
                source,
                loader,
                plan=plan,
                device=device,
                epochs=epochs,
                training_step=training_step,
                timing_profile=timing_profile,
                fingerprint_fn=fingerprint_fn,
                collect_comparability=collect_comparability,
            )
            monitor.raise_if_exceeded()
            transfer_active = pass_metrics.gpu_transfer_active
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        failure = (type(exc).__name__, _describe_exception(exc))
    finally:
        if loader is not None:
            cleanup_start = _clock()
            cleanup_failure = _shutdown_loader(loader)
            cleanup_s = _clock() - cleanup_start
        else:
            cleanup_failure = None
        snapshot = monitor.stop()
        trial_end = _clock()
        if gpu_stats_reset:
            gpu_peak = _gpu_peak_bytes(device)
        if failure is None and cleanup_failure is not None:
            failure = cleanup_failure

    if pass_metrics is not None:
        pass_metrics = _with_setup_timings(
            pass_metrics,
            construction_s=construction_s,
            training_setup_s=training_setup_s,
            loader_cleanup_s=cleanup_s,
            total_elapsed_s=trial_end - trial_start,
        )
        transfer_active = transfer_active or pass_metrics.gpu_transfer_active
        cold_start = construction_s + (training_setup_s or 0.0) + pass_metrics.startup_s

    return _PassExecution(
        pass_metrics=pass_metrics,
        cold_start_s=cold_start,
        memory=snapshot,
        gpu_peak_bytes=gpu_peak,
        gpu_transfer_active=transfer_active,
        error=failure,
    )


def _run_cycles(
    source: LoaderSource,
    loader: DataLoader,
    *,
    plan: BatchPlan,
    device: torch.device,
    epochs: int,
    training_step: TrainingStep | None,
    timing_profile: str = "standard",
    fingerprint_fn: Callable[[Any], Any] | None = None,
    collect_comparability: bool = True,
) -> PassMetrics:
    cycles: list[CycleMetrics] = []
    for cycle_index in range(epochs):
        cycle = _run_cycle(
            source,
            loader,
            plan=plan,
            device=device,
            cycle_index=cycle_index,
            training_step=training_step,
            timing_profile=timing_profile,
            fingerprint_fn=fingerprint_fn,
            collect_comparability=collect_comparability,
        )
        cycles.append(cycle)

    cold_startup_s = cycles[0].startup_s
    training = training_step is not None
    settled_cycles = cycles[:1] if epochs == 1 else cycles[1:]
    steady_samples = sum(item.steady_samples for item in settled_cycles)
    steady_elapsed = sum(item.duration_s for item in settled_cycles)
    throughput = steady_samples / steady_elapsed if steady_elapsed > 0 else 0.0
    transitions = [item.iterator_init_s + item.first_batch_s for item in cycles[1:]]
    worker_startup = 0.0 if loader.persistent_workers else sum(transitions)
    all_latencies = tuple(latency for item in cycles for latency in item.latencies_s)
    step_device_latencies = tuple(
        latency for item in cycles for latency in item.step_device_latencies_s
    )
    transfer_values = [item.transfer_s for item in cycles if item.transfer_s is not None]
    transfer_device_values = [
        item.transfer_device_s for item in cycles if item.transfer_device_s is not None
    ]
    cuda_values = [item.cuda_execution_s for item in cycles if item.cuda_execution_s is not None]
    total_batches = sum(item.batches for item in cycles)
    full_epoch = plan.full_epoch and all(item.full_epoch for item in cycles)
    training_cycles = [item for item in cycles if item.data_wait_s is not None]
    return PassMetrics(
        startup_s=cold_startup_s,
        steady_elapsed_s=steady_elapsed,
        batches=total_batches,
        steady_samples=steady_samples,
        throughput=throughput,
        latencies_s=all_latencies,
        gpu_transfer_active=any(item.transfer_active for item in cycles),
        construction_s=None,
        training_setup_s=None,
        total_elapsed_s=sum(item.cycle_wall_s for item in cycles),
        sustained_throughput=throughput,
        cycles=tuple(cycles),
        expected_batches=plan.expected_batches,
        full_epoch=full_epoch,
        training=training,
        data_wait_s=(
            sum(item.data_wait_s or 0.0 for item in training_cycles) if training_cycles else None
        ),
        transfer_s=sum(transfer_values) if transfer_values else None,
        transfer_bytes=(
            sum(item.transfer_bytes or 0 for item in cycles)
            if any(item.transfer_bytes is not None for item in cycles)
            else None
        ),
        worker_startup_overhead_s=worker_startup,
        average_epoch_transition_s=(sum(transitions) / len(transitions) if transitions else 0.0),
        amortized_throughput=None,
        total_samples=sum(item.samples for item in cycles),
        iterator_init_s=sum(item.iterator_init_s for item in cycles),
        first_batch_s=sum(item.first_batch_s for item in cycles),
        worker_shutdown_s=sum(item.shutdown_s for item in cycles),
        loader_cleanup_s=None,
        cuda_execution_s=sum(cuda_values) if cuda_values else None,
        transfer_device_s=(sum(transfer_device_values) if transfer_device_values else None),
        step_device_latencies_s=step_device_latencies,
        timing_profile=timing_profile,
    )


def _with_setup_timings(
    metrics: PassMetrics,
    *,
    construction_s: float,
    training_setup_s: float | None,
    loader_cleanup_s: float | None,
    total_elapsed_s: float,
) -> PassMetrics:
    amortized = metrics.total_samples / total_elapsed_s if total_elapsed_s > 0 else None
    return replace(
        metrics,
        construction_s=construction_s,
        training_setup_s=training_setup_s,
        loader_cleanup_s=loader_cleanup_s,
        total_elapsed_s=total_elapsed_s,
        amortized_throughput=amortized,
    )


def _run_cycle(
    source: LoaderSource,
    loader: DataLoader,
    *,
    plan: BatchPlan,
    device: torch.device,
    cycle_index: int,
    training_step: TrainingStep | None,
    timing_profile: str = "standard",
    fingerprint_fn: Callable[[Any], Any] | None = None,
    collect_comparability: bool = True,
) -> CycleMetrics:
    transfer = device.type == "cuda"
    detailed = timing_profile == "detailed"
    if transfer:
        torch.cuda.synchronize(device)

    cycle_wall_start = _clock()
    iterator_init_start = _clock()
    iterator = iter(loader)
    iterator_init_s = _clock() - iterator_init_start
    first_batch_start = _clock()
    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        _shutdown_iterator(iterator)
        raise DatasetError("the DataLoader produced no batches") from exc
    first_batch_s = _clock() - first_batch_start
    startup_s = iterator_init_s + first_batch_s
    worker_pids = _worker_pids(iterator)

    cycle_start = _clock()
    cycle_events: tuple[Any, Any] | None = None
    cuda_stream = torch.cuda.current_stream(device) if transfer else None
    if transfer:
        cycle_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        cycle_events[0].record(cuda_stream)

    transfer_events: list[tuple[Any, Any]] = []
    step_events: list[tuple[Any, Any]] = []

    def start_device_pair() -> tuple[Any, Any] | None:
        if not (transfer and detailed):
            return None
        pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        pair[0].record(cuda_stream)
        return pair

    def finish_device_pair(
        pair: tuple[Any, Any] | None,
        collection: list[tuple[Any, Any]],
    ) -> None:
        if pair is not None:
            pair[1].record(cuda_stream)
            collection.append(pair)

    latencies: list[float] = []
    data_wait_s = 0.0
    transfer_s = 0.0
    transfer_bytes = 0
    transfer_active = False
    step_latencies: list[float] = []
    structures: list[str] = []
    fingerprints: list[str] = []
    sample_counts: list[int] = []
    rss = process_rss_bytes()
    shutdown_s = 0.0
    failed = False

    def observe(raw_batch: Any, sample_count: int) -> None:
        if not collect_comparability:
            return
        structures.append(batch_structure(raw_batch))
        sample_counts.append(sample_count)
        if fingerprint_fn is not None:
            try:
                fingerprints.append(batch_fingerprint(fingerprint_fn(raw_batch)))
            except FingerprintError:
                raise
            except Exception as exc:
                raise FingerprintError(
                    "sample_fingerprint_fn failed while processing batch "
                    f"{len(sample_counts)}: {type(exc).__name__}: {exc}"
                ) from exc

    batch = first_batch
    first_samples = source.count_samples(batch)
    observe(batch, first_samples)
    if transfer:
        size = _tensor_bytes(batch)
        pair = start_device_pair()
        move_start = _clock()
        batch, moved_any = _move_to_device(batch, device)
        transfer_s += _clock() - move_start
        finish_device_pair(pair, transfer_events)
        transfer_bytes += size
        transfer_active = transfer_active or moved_any
    if training_step is not None:
        pair = start_device_pair()
        step_start = _clock()
        training_step(batch)
        step_latencies.append(_clock() - step_start)
        finish_device_pair(pair, step_events)

    batches = 1
    samples = first_samples
    try:
        for _ in range(1, plan.batches):
            wait_start = _clock()
            batch_start = wait_start
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise DatasetError(
                    f"the DataLoader produced fewer than the planned {plan.batches} batches"
                ) from exc
            data_wait_s += _clock() - wait_start
            samples_this = source.count_samples(batch)
            observe(batch, samples_this)
            if transfer:
                size = _tensor_bytes(batch)
                pair = start_device_pair()
                move_start = _clock()
                batch, moved_any = _move_to_device(batch, device)
                transfer_s += _clock() - move_start
                finish_device_pair(pair, transfer_events)
                transfer_bytes += size
                transfer_active = transfer_active or moved_any
            if training_step is not None:
                pair = start_device_pair()
                step_start = _clock()
                training_step(batch)
                step_latencies.append(_clock() - step_start)
                finish_device_pair(pair, step_events)
            batches += 1
            samples += samples_this
            latencies.append(_clock() - batch_start)

        cuda_execution_s: float | None = None
        transfer_device_s: float | None = None
        step_device_latencies: tuple[float, ...] = ()
        if transfer:
            assert cycle_events is not None
            cycle_events[1].record(cuda_stream)
            torch.cuda.synchronize(device)
            cuda_execution_s = cycle_events[0].elapsed_time(cycle_events[1]) / 1000.0
            if detailed:
                transfer_device_s = (
                    sum(start.elapsed_time(end) for start, end in transfer_events) / 1000.0
                )
                step_device_latencies = tuple(
                    start.elapsed_time(end) / 1000.0 for start, end in step_events
                )
        duration_s = max(_clock() - cycle_start, 1e-9)
    except BaseException:
        failed = True
        raise
    finally:
        if failed or not loader.persistent_workers:
            shutdown_start = _clock()
            _shutdown_iterator(iterator)
            shutdown_s = _clock() - shutdown_start

    batch_observations = (
        BatchObservations(
            structures=tuple(structures),
            sample_counts=tuple(sample_counts),
            fingerprints=(tuple(fingerprints) if fingerprint_fn is not None else None),
        )
        if collect_comparability
        else None
    )
    training = training_step is not None
    execution_samples = samples if training else max(samples - first_samples, 0)
    throughput = execution_samples / duration_s
    cycle_wall_s = _clock() - cycle_wall_start
    return CycleMetrics(
        cycle_index=cycle_index,
        startup_s=startup_s,
        duration_s=duration_s,
        batches=batches,
        samples=samples,
        throughput=throughput,
        steady_elapsed_s=duration_s,
        steady_samples=execution_samples,
        steady_throughput=throughput,
        latencies_s=tuple(latencies),
        rss_bytes=rss,
        data_wait_s=data_wait_s if training else None,
        transfer_s=transfer_s if transfer else None,
        transfer_bytes=transfer_bytes if transfer else None,
        transfer_active=transfer_active,
        step_latencies_s=tuple(step_latencies),
        first_batch_samples=first_samples,
        iterator_init_s=iterator_init_s,
        first_batch_s=first_batch_s,
        shutdown_s=shutdown_s,
        cycle_wall_s=cycle_wall_s,
        full_epoch=plan.full_epoch and batches >= (plan.expected_batches or plan.batches),
        timing_profile=timing_profile,
        cuda_execution_s=cuda_execution_s,
        transfer_device_s=transfer_device_s,
        step_device_latencies_s=step_device_latencies,
        batch_observations=batch_observations,
        worker_pids=worker_pids,
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
    *,
    epochs: int,
    training: bool,
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
            epochs=epochs,
            training=training,
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
        epochs=epochs,
        training=training,
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


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _gpu_peak_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return None


def _worker_pids(iterator: Any) -> tuple[int, ...]:
    workers = getattr(iterator, "_workers", None)
    if workers is None:
        return ()
    pids: list[int] = []
    for worker in workers:
        pid = getattr(worker, "pid", None)
        if isinstance(pid, int) and pid > 0:
            pids.append(pid)
    return tuple(pids)


def _shutdown_loader(loader: DataLoader) -> tuple[str, str] | None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return None
    return _shutdown_iterator(iterator)


def _shutdown_iterator(iterator: Any) -> tuple[str, str] | None:
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if not callable(shutdown):
        return None
    try:
        shutdown()
    except Exception as exc:
        return ("WorkerCleanupError", _describe_exception(exc))
    return None


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip() or repr(exc)
    if len(message) > 2000:
        return f"{message[:1997]}..."
    return message


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
