# Changelog

All notable changes to this project are documented in this file.

## [0.3.0] - 2026-11-17

### Added

- Explicit end-to-end training mode with `training_step_factory`, forward/backward/optimizer execution, and training-throughput ranking.
- Config-aware `dataloader_factory` input with contract validation and factory-managed reproducibility reporting.
- Multi-epoch loader and training trials with cold-start, first-epoch, sustained, and per-cycle throughput.
- Cycle metrics for worker process IDs, epoch-transition overhead, RSS, data wait, host-to-device transfer, and training-step latency.
- Loader state isolation for supported generators, samplers, and batch samplers, plus Python/Torch/NumPy/CUDA RNG restoration.
- Reproducibility statuses (`verified`, `partial`, `factory_managed`, `not_guaranteed`) and dataset/transform risk diagnostics.
- Progressive training finalists covering the initial, fastest, and highest-ranked loader configurations.
- `validation_policy` with `minimum` (default) and `positive` paired-bootstrap policies.
- `epochs`, `training_rounds`, `training_validation_rounds`, `max_training_candidates`, and `mode` options in the Python API and CLI.
- JSON schema version 3 and protocol version 3.0 with per-cycle, training, reproducibility, factory, and count-field data.
- Dedicated training and reproducibility sections in HTML reports.
- A real MLP training workload in `benchmarks/` and a training example in `examples/`.
- Tests for multi-epoch cycles, worker reuse, RNG restoration, isolation, training mode, factories, validation policies, reports, and CLI training.

### Changed

- Benchmark trials now construct one loader and run multiple epochs instead of measuring only one steady-state pass.
- Loader mode defaults to one epoch, preserving the previous loader benchmark semantics.
- Training mode defaults to three epochs and selects candidates by sustained end-to-end training throughput.
- The default validation policy now requires the lower 95% interval bound to reach `min_improvement`; the previous behavior corresponds to `positive`.
- The CLI now uses mutually exclusive `--dataset` and `--factory` inputs and exposes training and validation-policy options.
- Reports use schema version 3 instead of schema version 2.
- README and workload documentation now cover loader mode, training mode, factories, isolation, and the new report fields.

### Fixed

- Isolated loader construction keeps custom samplers and batch samplers when they cannot be cloned instead of replacing them with `None`.
- Independent validation now uses an explicitly supplied loader source, so factory-backed loaders are not silently reconstructed as templates.
- Partially capped epochs are marked as non-full-epoch measurements.
- Factory-backed results can rebuild a loader through the original factory.
- Loader conflicts report a single actionable dataset/loader/factory error.
- Non-persistent worker teardown is excluded from steady-state throughput instead of being charged to the measured epoch.

### Known limitations

- Map-style datasets only; `IterableDataset` is not supported.
- Training mode is single-device; distributed training and gradient accumulation orchestration are not managed by APDO.
- Arbitrary mutable state hidden inside datasets, transforms, worker initialization, or factories cannot be fully inspected.
- Training mode refuses detected unreproducible loader state and requires fresh training state from the user factory.
- The memory limit is sampled and advisory, not a strict system limit.
- macOS is not covered by CI in this release.

## [0.2.0] - 2026-11-16

### Added

- Seeded balanced execution rounds with recorded execution order and per-round metrics.
- `optimize(dataloader=...)` support for existing `DataLoader` objects, including loader reconstruction through `result.build_dataloader()`.
- Preserved dataset, batch size, sampler, batch sampler, collate function, `drop_last`, timeout, worker initialization, generator, multiprocessing context, `in_order`, and `pin_memory_device`.
- Independent paired validation with paired bootstrap intervals and `confirmed`, `inconclusive`, `regression`, `not_needed`, and `not_possible` outcomes.
- Safe fallback to the initial configuration unless validation confirms the configured minimum improvement.
- `seed`, `rounds`, `validation_rounds`, `min_improvement`, `time_budget_s`, and `sample_count_fn` options.
- Custom sample counting for variable-size and non-tensor batch structures.
- Expanded HTML validation section, loader preservation details, and JSON schema version 2.
- Three deterministic local benchmark workloads for in-memory, disk-backed, and CPU-heavy data pipelines.
- New tests for balanced ordering, loader preservation, sample counting, validation outcomes, custom loaders, budget exhaustion, reports, and workloads.

### Changed

- The search now selects a candidate for independent validation instead of treating search intervals as confirmation.
- The final recommendation is the initial configuration when validation does not confirm a gain.
- `repetitions` remains supported as a legacy alias for measured search rounds.
- JSON reports now use schema version 2 and protocol version 2.0.

### Fixed

- Custom `DataLoader` instances no longer lose their batch size, sampler, or configured batching behavior during reconstruction.
- Worker scans now always include `num_workers=0` even when the initial loader uses workers.
- Balanced benchmark execution order is reproducible from the configured seed.
- Bootstrap intervals are used consistently for search ranking and paired validation.
- Validation truncated by the time budget cannot confirm a candidate; the initial configuration is retained.

### Known limitations

- Map-style datasets only; `IterableDataset` is not supported.
- Loading throughput only; no model training or inference is measured.
- The memory limit is sampled and advisory, not a strict system limit.
- macOS is not covered by CI in this release.

## [0.1.0] - 2026-11-15

### Added

- `apdo.optimize()` API with a recommended `LoaderConfig` and direct `DataLoader` keyword export.
- Two-stage grid search over `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`.
- Warmup passes, repeated steady-state measurements, first-batch startup timing, per-batch latency, and sample throughput.
- Parent-process RSS and system available-memory sampling with a soft `max_ram_gb` guard.
- CUDA host-to-device tensor copies when benchmarking a CUDA device, with peak CUDA allocation reporting.
- Bootstrap confidence intervals and simplicity-aware configuration selection.
- JSON report export and a self-contained HTML report.
- `apdo benchmark` CLI with dataset factories, resource limits, report paths, and Windows multiprocessing support.
- CPU and CUDA integration tests, Ruff configuration, and GitHub Actions CI.
- Packaging metadata for PyPI distribution.

### Known limitations

- Map-style datasets only; `IterableDataset` is not supported.
- Loading throughput only; no model training or inference is measured.
- The memory limit is sampled and advisory, not a strict system limit.
- macOS is not covered by CI in this release.
