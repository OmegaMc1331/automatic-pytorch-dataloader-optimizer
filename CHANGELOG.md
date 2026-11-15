# Changelog

All notable changes to this project are documented in this file.

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
