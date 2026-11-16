# Changelog

All notable changes to this project are documented in this file.

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
