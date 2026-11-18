from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from apdo.benchmark import (
    BatchPlan,
    ConfigMeasurement,
    StageResult,
    benchmark_rounds,
    benchmark_source,
)
from apdo.comparability import compare_measurements, validate_comparability_policy
from apdo.config import (
    BENCHMARK_MODES,
    BENCHMARK_PROTOCOL_VERSION,
    MAX_BENCHMARK_EPOCHS,
    SELECTION_METRICS,
    TIMING_PROFILES,
    VALIDATION_POLICIES,
    BenchmarkSettings,
    LoaderConfig,
)
from apdo.errors import BenchmarkError, ConfigurationError, DatasetError
from apdo.loader import (
    LoaderFactory,
    LoaderSource,
    LoaderTemplate,
    initial_loader_config,
)
from apdo.metrics import MetricsSummary
from apdo.order import RoundOrder
from apdo.report import OptimizationResult
from apdo.resources import available_cpu_count, collect_environment
from apdo.training import TrainingStepFactory
from apdo.validation import (
    ValidationObservation,
    ValidationResult,
    analyze_validation,
    final_configuration,
)

MATERIAL_BAND = 0.95
DEFAULT_MAX_WORKERS = 8
DEFAULT_SEED = 42
DEFAULT_ROUNDS = 3
DEFAULT_VALIDATION_ROUNDS = 5
DEFAULT_MIN_IMPROVEMENT = 0.05
DEFAULT_TRAINING_EPOCHS = 3
DEFAULT_TRAINING_ROUNDS = 2
DEFAULT_MAX_TRAINING_CANDIDATES = 3
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
    dataset: Dataset | DataLoader | None = None,
    batch_size: int | None = None,
    device: str | torch.device = "auto",
    max_ram_gb: float | None = None,
    *,
    dataloader: DataLoader | None = None,
    dataloader_factory: LoaderFactory | None = None,
    mode: str = "loader",
    epochs: int | None = None,
    training_step_factory: TrainingStepFactory | None = None,
    training_rounds: int | None = None,
    training_validation_rounds: int | None = None,
    max_training_candidates: int = DEFAULT_MAX_TRAINING_CANDIDATES,
    validation_policy: str = "minimum",
    timing_profile: str = "standard",
    selection_metric: str = "sustained",
    comparability_policy: str = "warn",
    sample_fingerprint_fn: Callable[[Any], Any] | None = None,
    warmup_runs: int = 1,
    repetitions: int = DEFAULT_ROUNDS,
    rounds: int | None = None,
    max_batches: int | None = 100,
    max_workers: int | None = None,
    seed: int = DEFAULT_SEED,
    validation: bool = True,
    validation_rounds: int = DEFAULT_VALIDATION_ROUNDS,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    time_budget_s: float | None = None,
    sample_count_fn: Callable[[Any], int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> OptimizationResult:
    warmup_runs = _positive_int(warmup_runs, "warmup_runs")
    repetitions = _positive_int(repetitions, "repetitions")
    if repetitions < 2:
        raise ConfigurationError("repetitions must be at least 2")
    if rounds is None:
        effective_rounds = repetitions
    else:
        effective_rounds = _positive_int(rounds, "rounds")
        if effective_rounds < 2:
            raise ConfigurationError("rounds must be at least 2")
    _nonnegative_int(seed, "seed")
    if not isinstance(validation, bool):
        raise ConfigurationError("validation must be a boolean")
    validation_rounds = _nonnegative_int(validation_rounds, "validation_rounds")
    if validation and validation_rounds < 2:
        raise ConfigurationError("validation_rounds must be at least 2 when validation is enabled")
    min_improvement = _fraction(min_improvement, "min_improvement")
    time_budget = _positive_float(time_budget_s, "time_budget_s")
    if max_batches is not None:
        max_batches = _positive_int(max_batches, "max_batches")
        if max_batches < 2:
            raise ConfigurationError("max_batches must be at least 2 or None")
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    if sample_count_fn is not None and not callable(sample_count_fn):
        raise ConfigurationError("sample_count_fn must be callable or None")
    if progress is not None and not callable(progress):
        raise ConfigurationError("progress must be callable or None")
    max_ram_value = _positive_float(max_ram_gb, "max_ram_gb")
    if mode not in BENCHMARK_MODES:
        raise ConfigurationError(f"mode must be one of {BENCHMARK_MODES}")
    if validation_policy not in VALIDATION_POLICIES:
        raise ConfigurationError(f"validation_policy must be one of {VALIDATION_POLICIES}")
    if timing_profile not in TIMING_PROFILES:
        raise ConfigurationError(f"timing_profile must be one of {TIMING_PROFILES}")
    if selection_metric not in SELECTION_METRICS:
        raise ConfigurationError(f"selection_metric must be one of {SELECTION_METRICS}")
    validate_comparability_policy(comparability_policy)
    if sample_fingerprint_fn is not None and not callable(sample_fingerprint_fn):
        raise ConfigurationError("sample_fingerprint_fn must be callable or None")
    resolved_epochs = (
        (DEFAULT_TRAINING_EPOCHS if mode == "training" else 1)
        if epochs is None
        else _positive_int(epochs, "epochs")
    )
    if resolved_epochs > MAX_BENCHMARK_EPOCHS:
        raise ConfigurationError(f"epochs must be at most {MAX_BENCHMARK_EPOCHS}")
    training_rounds_value = (
        DEFAULT_TRAINING_ROUNDS
        if training_rounds is None
        else _positive_int(training_rounds, "training_rounds")
    )
    if training_rounds_value < 2:
        raise ConfigurationError("training_rounds must be at least 2")
    training_validation_value = (
        validation_rounds
        if training_validation_rounds is None
        else _positive_int(training_validation_rounds, "training_validation_rounds")
    )
    if validation and training_validation_value < 2:
        raise ConfigurationError(
            "training_validation_rounds must be at least 2 when validation is enabled"
        )
    max_training_candidates = _positive_int(
        max_training_candidates,
        "max_training_candidates",
    )
    if mode == "training" and training_step_factory is None:
        raise ConfigurationError(
            "training mode requires training_step_factory; APDO never starts model "
            "training implicitly"
        )
    if mode == "loader" and training_step_factory is not None:
        raise ConfigurationError("training_step_factory requires mode='training'")
    if training_step_factory is not None and not callable(training_step_factory):
        raise ConfigurationError("training_step_factory must be callable")

    source, initial_config, explicit_batch_size = _prepare_source(
        dataset,
        dataloader,
        dataloader_factory,
        batch_size,
        sample_count_fn,
    )
    if mode == "training" and not source.isolation.training_allowed:
        raise ConfigurationError(
            "training mode requires a reproducible training environment; "
            f"reproducibility status is {source.isolation.status}: {source.isolation.reason}"
        )
    plan = _plan_for_source(source, max_batches)
    dataset_size = source.dataset_size
    requested_device = str(device)
    resolved_device = resolve_device(device)

    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    deadline = start_time + time_budget if time_budget is not None else None
    _emit(progress, "Collecting environment information")
    environment = collect_environment()
    cpu_count = available_cpu_count()
    resolved_max_workers = _resolve_max_workers(max_workers, cpu_count)
    validation_rounds_used = training_validation_value if mode == "training" else validation_rounds
    settings = BenchmarkSettings(
        batch_size=explicit_batch_size,
        requested_device=requested_device,
        device=str(resolved_device),
        warmup_runs=warmup_runs,
        repetitions=effective_rounds,
        max_batches=max_batches,
        max_workers=resolved_max_workers,
        max_ram_gb=max_ram_value,
        dataset_size=dataset_size,
        cpu_count=cpu_count,
        seed=seed,
        rounds=effective_rounds,
        validation_rounds=validation_rounds_used,
        min_improvement=min_improvement,
        validation_enabled=validation,
        time_budget_s=time_budget,
        initial_config=initial_config,
        input_kind=source.input_kind,
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        expected_batches=plan.batches,
        benchmark_mode=mode,
        epochs=resolved_epochs,
        training_rounds=training_rounds_value,
        training_validation_rounds=training_validation_value,
        max_training_candidates=max_training_candidates,
        validation_policy=validation_policy,
        reproducibility_status=source.isolation.status,
        timing_profile=timing_profile,
        selection_metric=selection_metric,
        comparability_policy=comparability_policy,
        sample_fingerprint=sample_fingerprint_fn is not None,
    )

    warnings = _initial_warnings(
        plan=plan,
        max_batches=max_batches,
        max_workers=max_workers,
        resolved_max_workers=resolved_max_workers,
        cpu_count=cpu_count,
        max_ram_value=max_ram_value,
        environment_has_memory=environment.available_memory_bytes is not None,
        initial_config=initial_config,
        source=source,
        mode=mode,
        epochs=resolved_epochs,
    )
    if timing_profile == "detailed":
        warnings.append(
            "Detailed timing adds CUDA-event instrumentation around every transfer and "
            "training step; the extra event records can alter very short-step timings."
        )
    if selection_metric == "amortized":
        warnings.append(
            "Selection uses total-trial amortized throughput, including loader construction, "
            "worker startup, epoch transitions, and cleanup."
        )
    if comparability_policy == "strict" and sample_fingerprint_fn is None:
        warnings.append(
            "Strict comparability requires a sample_fingerprint_fn; without one, sample "
            "identity cannot be verified and a search candidate will not be confirmed."
        )
    if rounds is not None and repetitions != DEFAULT_ROUNDS:
        warnings.append(
            "Both rounds and repetitions were provided; rounds takes precedence for the "
            "number of measured search rounds."
        )

    max_ram_bytes = int(max_ram_value * 1_000_000_000) if max_ram_value is not None else None
    pin_options = _pin_options(resolved_device)
    worker_counts = _build_worker_counts(resolved_max_workers, cpu_count)
    stage_one = _stage_one_configs(worker_counts, pin_options, initial_config)
    stage_total = 3 if mode == "training" else 2
    search_epochs = 1 if mode == "training" else resolved_epochs
    stage_one_result = _run_stage(
        source=source,
        configs=stage_one,
        phase="worker_scan",
        progress=progress,
        device=resolved_device,
        plan=plan,
        warmup_rounds=warmup_runs,
        rounds=effective_rounds,
        max_ram_bytes=max_ram_bytes,
        seed=seed,
        deadline=deadline,
        stage_number=1,
        stage_total=stage_total,
        epochs=search_epochs,
        timing_profile=timing_profile,
        comparability_policy=comparability_policy,
        fingerprint_fn=sample_fingerprint_fn,
    )
    measurements = list(stage_one_result.measurements)
    orders = list(stage_one_result.orders)
    budget_exhausted = stage_one_result.budget_exhausted
    stage_one_successes = [item for item in measurements if item.succeeded]
    if not stage_one_successes:
        detail = "\n".join(_failure_lines(measurements))
        raise BenchmarkError("No DataLoader configuration completed successfully.\n" + detail)

    baseline = next(
        (item for item in stage_one_successes if item.config == initial_config),
        None,
    )
    if baseline is None:
        failed_baseline = next(
            (item for item in measurements if item.config == initial_config),
            None,
        )
        detail = ""
        if failed_baseline is not None:
            detail = f"\n  {failed_baseline.error_type}: {failed_baseline.error_message}"
        raise BenchmarkError(
            "The initial DataLoader configuration failed, so APDO cannot establish a "
            "safe fallback baseline." + detail
        )

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

    if time_budget is not None and time.perf_counter() >= deadline:
        budget_exhausted = True
        _emit(progress, "Stage 2: skipped because the time budget was exhausted")
        warnings.append(
            "The time budget was exhausted before the flag sweep; only stage-one "
            "configurations were considered."
        )
    elif stage_two:
        stage_two_result = _run_stage(
            source=source,
            configs=stage_two,
            phase="flag_sweep",
            progress=progress,
            device=resolved_device,
            plan=plan,
            warmup_rounds=warmup_runs,
            rounds=effective_rounds,
            max_ram_bytes=max_ram_bytes,
            seed=seed,
            deadline=deadline,
            stage_number=2,
            stage_total=stage_total,
            epochs=search_epochs,
            timing_profile=timing_profile,
            comparability_policy=comparability_policy,
            fingerprint_fn=sample_fingerprint_fn,
        )
        measurements.extend(stage_two_result.measurements)
        orders.extend(stage_two_result.orders)
        budget_exhausted = budget_exhausted or stage_two_result.budget_exhausted
        if stage_two_result.budget_exhausted:
            warnings.append("The time budget stopped the flag sweep before all rounds completed.")
    else:
        _emit(progress, "Stage 2: no additional multi-worker configurations to test")

    _emit(progress, "Selecting a loader search recommendation")
    loader_selection = _select_configuration(measurements, selection_metric)
    selection = loader_selection
    if mode == "training":
        baseline = _measurement_for_config(measurements, initial_config)
        training_candidates = _training_candidates(
            baseline=baseline,
            selection=loader_selection,
            measurements=measurements,
            limit=max_training_candidates,
        )
        training_result = _run_stage(
            source=source,
            configs=training_candidates,
            phase="training_finalists",
            progress=progress,
            device=resolved_device,
            plan=plan,
            warmup_rounds=warmup_runs,
            rounds=training_rounds_value,
            max_ram_bytes=max_ram_bytes,
            seed=seed,
            deadline=deadline,
            stage_number=3,
            stage_total=stage_total,
            epochs=resolved_epochs,
            training_step_factory=training_step_factory,
            timing_profile=timing_profile,
            comparability_policy=comparability_policy,
            fingerprint_fn=sample_fingerprint_fn,
        )
        measurements.extend(training_result.measurements)
        orders.extend(training_result.orders)
        budget_exhausted = budget_exhausted or training_result.budget_exhausted
        training_successes = [item for item in training_result.measurements if item.succeeded]
        baseline = next(
            (item for item in training_successes if item.config == initial_config),
            None,
        )
        if baseline is None:
            detail = "\n".join(_failure_lines(training_result.measurements))
            raise BenchmarkError(
                "The initial configuration did not complete a successful training "
                "measurement, so APDO cannot establish a safe fallback.\n" + detail
            )
        selection = _select_configuration(training_successes, selection_metric)
        if training_result.budget_exhausted:
            warnings.append(
                "The time budget stopped training finalists before all planned rounds completed."
            )
        warnings.append(
            "Training throughput includes the user-supplied forward, backward, and "
            "optimizer step on the resolved device."
        )

    baseline_summary = _summary(baseline)
    selected_summary = _summary(selection.selected)
    baseline_value = _metric_value(baseline_summary, selection_metric)
    selected_value = _metric_value(selected_summary, selection_metric)
    search_improvement = (
        selected_value / baseline_value - 1.0 if baseline_value > 0 else 0.0
    ) * 100.0

    failures = [item for item in measurements if not item.succeeded]
    if failures:
        warnings.append(
            f"{len(failures)} of {len(measurements)} configurations failed; "
            "see the report for details."
        )
    if len(stage_one_successes) == 1 and not stage_two:
        warnings.append(
            "Only one configuration completed successfully; no comparative search "
            "recommendation could be made."
        )
    if (
        resolved_device.type == "cuda"
        and measurements
        and not any(item.gpu_transfer_active for item in measurements if item.succeeded)
    ):
        warnings.append(
            "Batches contained no tensor values, so host-to-device transfer was not "
            "exercised; pin_memory results may not be representative."
        )

    validation_result, validation_orders, validation_budget = _run_validation(
        template=source.template,
        source=source,
        baseline_config=initial_config,
        candidate_config=selection.selected.config,
        enabled=validation,
        validation_rounds=validation_rounds_used,
        warmup_rounds=warmup_runs,
        min_improvement=min_improvement,
        device=resolved_device,
        plan=plan,
        max_ram_bytes=max_ram_bytes,
        seed=seed,
        deadline=deadline,
        progress=progress,
        epochs=resolved_epochs,
        training_step_factory=training_step_factory,
        validation_policy=validation_policy,
        timing_profile=timing_profile,
        selection_metric=selection_metric,
        comparability_policy=comparability_policy,
        fingerprint_fn=sample_fingerprint_fn,
    )
    orders.extend(validation_orders)
    budget_exhausted = budget_exhausted or validation_budget
    warnings.extend(validation_result.failures)
    if validation_result.comparability is not None:
        comparison = validation_result.comparability
        if comparison.has_observed_problem or (
            comparability_policy == "strict" and not comparison.sufficient_for_strict
        ):
            warnings.append(
                "Comparability diagnostics did not establish sufficient data identity; "
                f"{comparison.summary}"
            )
    if budget_exhausted:
        warnings.append(
            "The configured time budget was exhausted; some planned measurements were "
            "skipped and the report marks the validation outcome accordingly."
        )

    final_config = final_configuration(initial_config, validation_result)
    final_improvement = (
        validation_result.improvement_percent if validation_result.confirmed else 0.0
    )
    if final_config == initial_config:
        final_improvement = 0.0
    selection_reason = _final_reason(initial_config, selection, validation_result)

    if validation_result.status != "confirmed" and selection.selected.config != initial_config:
        warnings.append(
            "Independent validation did not confirm the search candidate; the initial "
            "configuration is retained as the recommendation."
        )
    if validation_result.status == "regression":
        warnings.append("Validation observed a candidate regression.")

    duration_s = time.perf_counter() - start_time
    _emit(
        progress,
        f"Validation status: {validation_result.status}; recommended {final_config.describe()}",
    )
    _emit(progress, f"Optimization completed in {duration_s:.2f} s")
    return OptimizationResult(
        best_config=final_config,
        fastest_config=selection.fastest.config,
        search_selected_config=selection.selected.config,
        measurements=tuple(measurements),
        environment=environment,
        settings=settings,
        warnings=tuple(dict.fromkeys(warnings)),
        selection_reason=selection_reason,
        baseline_throughput=baseline_summary.median_throughput,
        baseline_amortized_throughput=baseline_summary.median_amortized_throughput,
        improvement_percent=final_improvement,
        started_at=started_at,
        duration_s=duration_s,
        validation=validation_result,
        round_orders=tuple(orders),
        loader_template=source.template if source.factory is None else None,
        loader_source=source,
        search_improvement_percent=search_improvement,
        search_selection_reason=selection.reason,
        budget_exhausted=budget_exhausted,
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
        if resolved.index is not None and (
            resolved.index < 0 or resolved.index >= torch.cuda.device_count()
        ):
            raise ConfigurationError(
                f"CUDA device index {resolved.index} is unavailable; "
                f"torch.cuda.device_count() is {torch.cuda.device_count()}"
            )
    elif resolved.type != "cpu":
        raise ConfigurationError(
            f"device type {resolved.type!r} is not supported in v0.3.0; "
            "use 'auto', 'cpu', or 'cuda'"
        )
    return resolved


def _prepare_source(
    dataset: Dataset | DataLoader | None,
    dataloader: DataLoader | None,
    dataloader_factory: LoaderFactory | None,
    batch_size: int | None,
    sample_count_fn: Callable[[Any], int] | None,
) -> tuple[LoaderSource, LoaderConfig, int | None]:
    provided = sum(value is not None for value in (dataset, dataloader, dataloader_factory))
    if provided == 0:
        raise ConfigurationError("pass a dataset or a dataloader, or provide a dataloader_factory")
    if provided > 1:
        raise ConfigurationError(
            "pass either dataset or dataloader, not both; dataloader_factory is also "
            "mutually exclusive with dataset and dataloader"
        )
    if dataloader_factory is not None:
        if batch_size is not None:
            raise ConfigurationError(
                "batch_size cannot override a dataloader_factory; the factory owns the "
                "batch size and loader construction"
            )
        source = LoaderSource.from_factory(
            dataloader_factory,
            sample_count_fn=sample_count_fn,
        )
        return source, LoaderConfig(), source.batch_size
    if isinstance(dataset, DataLoader):
        if dataloader is not None:
            raise ConfigurationError("pass either dataset or dataloader, not both")
        dataloader = dataset
        dataset = None
    if dataset is not None and dataloader is not None:
        raise ConfigurationError("pass either dataset or dataloader, not both")

    if dataloader is not None:
        if not isinstance(dataloader, DataLoader):
            raise ConfigurationError("dataloader must be a torch.utils.data.DataLoader")
        if batch_size is not None:
            batch_size = _positive_int(batch_size, "batch_size")
            if dataloader.batch_size is None or dataloader.batch_size != batch_size:
                raise ConfigurationError(
                    "batch_size cannot override an existing DataLoader; pass the dataset "
                    "API or create a DataLoader with the intended batch size"
                )
        template = LoaderTemplate.from_dataloader(
            dataloader,
            sample_count_fn=sample_count_fn,
        )
        initial_config = initial_loader_config(dataloader)
        source = LoaderSource.from_template(template, "dataloader")
        return source, initial_config, dataloader.batch_size

    resolved_batch_size = 32 if batch_size is None else _positive_int(batch_size, "batch_size")
    template = LoaderTemplate.from_dataset(
        dataset,
        resolved_batch_size,
        sample_count_fn=sample_count_fn,
    )
    source = LoaderSource.from_template(template, "dataset")
    return source, LoaderConfig(), resolved_batch_size


def _plan_for_source(source: LoaderSource, max_batches: int | None) -> BatchPlan:
    total_batches = source.expected_batches
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


def _training_candidates(
    *,
    baseline: ConfigMeasurement,
    selection: _Selection,
    measurements: Sequence[ConfigMeasurement],
    limit: int,
) -> list[LoaderConfig]:
    ordered = [baseline.config, selection.selected.config, selection.fastest.config]
    ranked = sorted(
        (item for item in measurements if item.succeeded and item.summary is not None),
        key=lambda item: _summary(item).median_throughput,
        reverse=True,
    )
    ordered.extend(item.config for item in ranked)
    unique: list[LoaderConfig] = []
    for config in ordered:
        if config not in unique:
            unique.append(config)
        if len(unique) >= limit:
            break
    return unique


def _measurement_for_config(
    measurements: Sequence[ConfigMeasurement],
    config: LoaderConfig,
) -> ConfigMeasurement:
    measurement = next(
        (
            item
            for item in measurements
            if item.succeeded and item.config == config and item.summary is not None
        ),
        None,
    )
    if measurement is None:
        raise BenchmarkError(f"no successful measurement is available for {config.describe()}")
    return measurement


def _initial_warnings(
    *,
    plan: BatchPlan,
    max_batches: int | None,
    max_workers: int | None,
    resolved_max_workers: int,
    cpu_count: int,
    max_ram_value: float | None,
    environment_has_memory: bool,
    initial_config: LoaderConfig,
    source: LoaderSource,
    mode: str,
    epochs: int,
) -> list[str]:
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
    if initial_config.num_workers > resolved_max_workers:
        warnings.append(
            f"The initial configuration uses {initial_config.num_workers} workers, above "
            f"the search limit of {resolved_max_workers}; it is measured as the baseline "
            "but that worker count is not explored further."
        )
    if max_ram_value is not None and not environment_has_memory:
        warnings.append(
            "Available system memory could not be read; max_ram_gb cannot be enforced "
            "on this platform."
        )
    if epochs > 1:
        if mode == "training":
            warnings.append(
                f"Training trials reuse one DataLoader for {epochs} epochs and exclude the "
                "first epoch from sustained throughput; loader screening uses one epoch."
            )
        else:
            warnings.append(
                f"Loader measurements reuse one DataLoader for {epochs} epochs; sustained "
                "throughput excludes the first epoch."
            )
    if source.isolation.status != "verified":
        warnings.append(source.isolation.reason)
    if source.factory is not None:
        warnings.extend(source.compatibility_warnings())
    if mode == "training":
        warnings.append(
            "Training mode executes the explicit training_step_factory. The factory owns "
            "model, optimizer, loss, and environment reset behavior for every trial."
        )
    return warnings


def _run_stage(
    *,
    source: LoaderSource,
    configs: Sequence[LoaderConfig],
    phase: str,
    progress: Callable[[str], None] | None,
    device: torch.device,
    plan: BatchPlan,
    warmup_rounds: int,
    rounds: int,
    max_ram_bytes: int | None,
    seed: int,
    deadline: float | None,
    stage_number: int,
    stage_total: int,
    epochs: int = 1,
    training_step_factory: TrainingStepFactory | None = None,
    timing_profile: str = "standard",
    comparability_policy: str = "warn",
    fingerprint_fn: Callable[[Any], Any] | None = None,
) -> StageResult:
    _emit(
        progress,
        f"Stage {stage_number}/{stage_total}: {phase} with {len(configs)} configurations, "
        f"{rounds} measured rounds, {epochs} epoch(s)",
    )
    result = benchmark_source(
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
        collect_comparability=comparability_policy != "off",
    )
    for measurement in result.measurements:
        if measurement.succeeded:
            summary = _summary(measurement)
            unit = "samples/s" if measurement.training else "loader samples/s"
            _emit(
                progress,
                f"  {measurement.config.describe()}: "
                f"{summary.median_throughput:,.1f} {unit} across "
                f"{len(measurement.passes)} rounds",
            )
        else:
            _emit(
                progress,
                f"  {measurement.config.describe()}: failed with "
                f"{measurement.error_type}: {measurement.error_message}",
            )
    return result


def _run_validation(
    *,
    template: LoaderTemplate | None,
    baseline_config: LoaderConfig,
    candidate_config: LoaderConfig,
    enabled: bool,
    validation_rounds: int,
    warmup_rounds: int,
    min_improvement: float,
    device: torch.device,
    plan: BatchPlan,
    max_ram_bytes: int | None,
    seed: int,
    deadline: float | None,
    progress: Callable[[str], None] | None,
    source: LoaderSource | None = None,
    epochs: int = 1,
    training_step_factory: TrainingStepFactory | None = None,
    validation_policy: str = "minimum",
    timing_profile: str = "standard",
    selection_metric: str = "sustained",
    comparability_policy: str = "warn",
    fingerprint_fn: Callable[[Any], Any] | None = None,
) -> tuple[ValidationResult, tuple[RoundOrder, ...], bool]:
    if candidate_config == baseline_config:
        result = analyze_validation(
            baseline_config,
            candidate_config,
            (),
            enabled=enabled,
            min_improvement=min_improvement,
            seed=seed,
            policy=validation_policy,
            selection_metric=selection_metric,
        )
        return result, (), False
    if not enabled:
        result = analyze_validation(
            baseline_config,
            candidate_config,
            (),
            enabled=False,
            min_improvement=min_improvement,
            seed=seed,
            policy=validation_policy,
            selection_metric=selection_metric,
        )
        return result, (), False
    if deadline is not None and time.perf_counter() >= deadline:
        result = analyze_validation(
            baseline_config,
            candidate_config,
            (),
            enabled=True,
            min_improvement=min_improvement,
            seed=seed,
            failures=("Independent validation was skipped because the time budget expired.",),
            policy=validation_policy,
            selection_metric=selection_metric,
        )
        return result, (), True
    legacy_template = template if source is None else None
    if source is None:
        if template is None:
            raise ConfigurationError("validation requires a loader source or template")
        source = LoaderSource.from_template(template, "dataloader")

    _emit(
        progress,
        f"Independent validation: alternating {baseline_config.describe()} and "
        f"{candidate_config.describe()} for {validation_rounds} rounds",
    )
    if training_step_factory is None and legacy_template is not None:
        stage = benchmark_rounds(
            legacy_template,
            (baseline_config, candidate_config),
            phase="validation",
            device=device,
            plan=plan,
            warmup_rounds=warmup_rounds,
            rounds=validation_rounds,
            max_ram_bytes=max_ram_bytes,
            seed=seed,
            deadline=deadline,
            progress=progress,
            epochs=epochs,
            timing_profile=timing_profile,
            fingerprint_fn=fingerprint_fn,
            collect_comparability=comparability_policy != "off",
        )
    else:
        stage = benchmark_source(
            source,
            (baseline_config, candidate_config),
            phase="validation",
            device=device,
            plan=plan,
            warmup_rounds=warmup_rounds,
            rounds=validation_rounds,
            max_ram_bytes=max_ram_bytes,
            seed=seed,
            deadline=deadline,
            progress=progress,
            epochs=epochs,
            training_step_factory=training_step_factory,
            timing_profile=timing_profile,
            fingerprint_fn=fingerprint_fn,
            collect_comparability=comparability_policy != "off",
        )
    by_config = {item.config: item for item in stage.measurements}
    baseline_measurement = by_config[baseline_config]
    candidate_measurement = by_config[candidate_config]
    failure_messages: list[str] = []
    for measurement in stage.measurements:
        if not measurement.succeeded:
            failure_messages.append(
                f"{measurement.config.describe()}: {measurement.error_type}: "
                f"{measurement.error_message}"
            )
    if stage.budget_exhausted:
        failure_messages.append(
            "The time budget stopped validation before all planned rounds completed."
        )

    observations: list[ValidationObservation] = []
    if baseline_measurement.succeeded and candidate_measurement.succeeded:
        baseline_rounds = {item.round_index: item for item in baseline_measurement.rounds}
        candidate_rounds = {item.round_index: item for item in candidate_measurement.rounds}
        for round_index in sorted(set(baseline_rounds) & set(candidate_rounds)):
            observations.append(
                ValidationObservation(
                    round_index=round_index,
                    baseline=baseline_rounds[round_index].pass_metrics,
                    candidate=candidate_rounds[round_index].pass_metrics,
                )
            )
    if stage.budget_exhausted:
        observations = []

    comparison = (
        compare_measurements(baseline_measurement, candidate_measurement)
        if comparability_policy != "off"
        else None
    )
    result = analyze_validation(
        baseline_config,
        candidate_config,
        observations,
        enabled=True,
        min_improvement=min_improvement,
        seed=seed,
        failures=failure_messages,
        policy=validation_policy,
        selection_metric=selection_metric,
        comparability=comparison,
        comparability_policy=comparability_policy,
    )
    return result, stage.orders, stage.budget_exhausted


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
    initial_config: LoaderConfig | None = None,
) -> list[LoaderConfig]:
    configs = [initial_config or LoaderConfig()]
    for workers in worker_counts:
        for pin_memory in pin_options:
            config = (
                LoaderConfig(num_workers=0, pin_memory=pin_memory)
                if workers == 0
                else LoaderConfig(
                    num_workers=workers,
                    pin_memory=pin_memory,
                    prefetch_factor=2,
                    persistent_workers=False,
                )
            )
            if config not in configs:
                configs.append(config)
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
    selection_metric: str = "sustained",
) -> _Selection:
    successful = [item for item in measurements if item.succeeded and item.summary]
    if not successful:
        raise BenchmarkError("no successful configuration is available for selection")
    if selection_metric not in SELECTION_METRICS:
        raise ConfigurationError(f"selection_metric must be one of {SELECTION_METRICS}")

    fastest = max(
        successful,
        key=lambda item: _metric_value(_summary(item), selection_metric),
    )
    fastest_summary = _summary(fastest)
    fastest_value = _metric_value(fastest_summary, selection_metric)
    fastest_low, fastest_high = _metric_interval(fastest_summary, selection_metric)
    equivalent = [
        item
        for item in successful
        if _metric_value(_summary(item), selection_metric) >= fastest_value * MATERIAL_BAND
        and _metric_interval(_summary(item), selection_metric)[0] <= fastest_high
        and fastest_low <= _metric_interval(_summary(item), selection_metric)[1]
    ]
    selected = min(equivalent, key=lambda item: item.config.simplicity_key())

    metric_label = (
        "median total-trial amortized throughput"
        if selection_metric == "amortized"
        else "median sustained throughput"
    )
    if selected.config == fastest.config:
        reason = (
            "The fastest measured search configuration was selected. No simpler "
            f"configuration was both within 5% of its {metric_label} and had a 95% "
            "bootstrap confidence interval that overlapped the fastest interval."
        )
    else:
        reason = (
            f"The fastest measured configuration was {fastest.config.describe()}. "
            f"{selected.config.describe()} was within 5% of its {metric_label} and "
            "had a 95% bootstrap confidence interval that overlapped the fastest "
            "interval, so the simpler configuration was preferred."
        )
    return _Selection(
        selected=selected,
        fastest=fastest,
        equivalent=tuple(equivalent),
        reason=reason,
    )


def _metric_value(summary: MetricsSummary, selection_metric: str) -> float:
    if selection_metric == "amortized":
        return summary.median_amortized_throughput or 0.0
    return summary.median_throughput


def _metric_interval(
    summary: MetricsSummary,
    selection_metric: str,
) -> tuple[float, float]:
    if selection_metric == "amortized" and summary.ci_low_amortized_throughput is not None:
        return (
            summary.ci_low_amortized_throughput,
            summary.ci_high_amortized_throughput
            if summary.ci_high_amortized_throughput is not None
            else summary.ci_low_amortized_throughput,
        )
    return summary.ci_low_throughput, summary.ci_high_throughput


def _final_reason(
    initial_config: LoaderConfig,
    selection: _Selection,
    validation: ValidationResult,
) -> str:
    if validation.status == "confirmed":
        return f"{validation.reason} The validated candidate is recommended."
    if selection.selected.config == initial_config:
        if validation.status == "not_needed":
            return "The search selected the initial configuration; no change is recommended."
        return f"The search selected the initial configuration. {validation.reason}"
    return (
        f"The search favored {selection.selected.config.describe()} over "
        f"{initial_config.describe()}, but independent validation did not confirm the "
        f"gain. {validation.reason} The initial configuration is recommended."
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


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if value < 0:
        raise ConfigurationError(f"{name} must be greater than or equal to 0")
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


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a number between 0 and 1")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ConfigurationError(f"{name} must be a finite number between 0 and 1")
    return converted


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)
