# Changelog

All notable changes to this project are documented in this file.

## [0.4.1] - 2026-11-25

### Added

- Exact one-sided paired sign test in validation, with `SIGN_TEST_EPSILON` handling so a gain exactly at the threshold is not counted as a success.
- Explicit validation payload fields for planned, completed, and valid pairs, rejected observations with reasons, statistical constants, and the sign-test p-value.
- Confirmed-only stability gain statistics, fallback counts, candidate agreement, and warning conditions for insufficient validation pairs, absent confirmed gains, and high agreement without evidence.
- Multi-epoch loader support in the startup-bound workload through `--loader-epochs`.

### Changed

- Validation now requires at least eight valid paired observations and the sign test to pass in addition to the selected policy's observed-median and interval conditions.
- Loader validation observations that are incomparable, incomplete, non-finite, or non-positive are rejected and reported instead of being replaced with zero gains.
- A time budget that truncates validation before all planned pairs complete now retains the completed observations and records the validation as `not_possible` and incomplete.
- Worker finalists, training candidates, search selection, ranking, report values, and HTML output now consistently use the selected `sustained` or `amortized` metric.
- Default validation rounds increased from 5 to 10 for benchmark and stability runs and for the Python API.
- Compute-bound benchmark now performs real MLP training over cheap input; startup-bound now exercises multiple loader epochs.
- Stability reports distinguish confirmed gains from fallback or initial-configuration agreement and clarify that qualification describes validation coverage, not evidence strength.
- Validation policy documentation now describes the exact sign test, invalid-observation handling, and the eight-pair minimum.

### Fixed

- Validation conclusions can no longer be confirmed from fewer than eight paired observations or from a failed sign test.
- Sustained and amortized selection no longer diverge between finalist generation, training candidate selection, reporting, and ranking.
- Budget-truncated validation no longer discards completed observations or substitutes zero gains.
- Stability no longer presents high recommendation agreement as evidence when no run confirmed a gain.
- The compute-bound and startup-bound workloads now measure the behavior their descriptions claim.

### Known limitations

- The sign test is exact and conservative for small samples; eight successes out of ten observations cannot reach `p < 0.05`.
- A single strong outlier can move the observed median gain and bootstrap interval; the sign test counters this but can make the result inconclusive rather than confirmed.
- The memory limit remains sampled and advisory.
- macOS is not covered by CI in this release.

## [0.4.0] - 2026-11-18

### Added

- Explicit lifecycle timing for loader construction, iterator initialization, first-batch acquisition, worker shutdown, loader cleanup, and complete trial wall.
- `sustained` (default) and `amortized` selection metrics. Amortized throughput divides all samples in a trial by the complete trial wall.
- `standard` and `detailed` timing profiles. Standard uses whole-cycle CUDA Events; detailed adds transfer and per-step device events without per-batch synchronization.
- Batch structure and optional sample-fingerprint comparability diagnostics with `off`, `warn`, and `strict` policies.
- `evaluate_stability` API and `apdo stability` CLI command for repeated independent optimizations with partial-failure retention, recommendation agreement, gain intervals, throughput spread, and failure coverage.
- JSON schema version 4 and benchmark protocol version 4.0 with lifecycle, timing, and comparability fields.
- Dedicated timing-breakdown and batch-comparability sections in HTML reports.
- `startup-bound` and `compute-bound` benchmark workloads alongside the existing in-memory, disk-backed, CPU-heavy, and training workloads.
- Tests for fake-clock lifecycle accounting, training first-batch accounting, sample fingerprints, comparability statuses, selection metrics, strict comparability gating, stability reports, and CLI stability output.

### Changed

- Loader cycle throughput still excludes the first loader batch, but the new amortized metric now makes construction, worker startup, transitions, shutdown, and cleanup visible and selectable.
- Validation decisions use the configured selection metric and comparability policy; strict comparisons require verified sample identity.
- Training first-batch samples are counted in total sample accounting, while the first training step remains inside the execution timer.
- README, examples, and workload documentation now describe protocol 4.0, lifecycle timing, comparability, and stability analysis.

### Fixed

- CUDA Events now record against the current CUDA stream, restoring compatibility with PyTorch 2.14.
- Training-mode sample accounting includes first-batch samples without double-counting epoch totals.
- Early validation exits preserve the selected comparability policy and comparison result.

### Known limitations

- Comparability fingerprints detect differences they are given; arbitrary dataset or transform state cannot be proven equivalent.
- Detailed CUDA event timing can perturb very short transfers or training steps.
- The memory limit remains sampled and advisory.
- macOS is not covered by CI in this release.

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
