# Automatic PyTorch DataLoader Optimizer

[![CI](https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`apdo` measures `torch.utils.data.DataLoader` configurations on the machine where it runs and recommends a setting for `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`.

The package name is `apdo`. The command-line interface is also `apdo`.

## Why APDO

DataLoader throughput depends on the dataset, storage, preprocessing cost, CPU topology, operating system, worker process behavior, and the device receiving the batches. Published rules of thumb and configurations tuned on another machine often perform worse than the default.

APDO runs the real loader, records repeated measurements, validates the search recommendation with fresh measurements, and retains the initial configuration when a gain is not confirmed. It does not estimate performance from hardware specifications and does not call a cloud service.

## Features in v0.2.0

- Optimizes an existing map-style dataset with a batch size, or an existing `DataLoader`.
- Preserves the original loader's dataset, batch size, sampler, batch sampler, collate function, `drop_last`, timeout, worker initialization, generator, multiprocessing context, and `in_order` setting.
- Changes only `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`.
- Runs seeded, balanced rounds and records the executed configuration order and every per-round measurement.
- Runs independent paired validation instead of treating the search result as proof.
- Returns `confirmed`, `inconclusive`, `regression`, `not_needed`, or `not_possible` validation outcomes.
- Falls back to the initial configuration unless validation confirms the required minimum gain.
- Supports optional time budgets and custom sample counting for unusual collate functions.
- Generates a schema-v2 JSON report and a self-contained HTML report with validation, ranking, execution order, environment, method, and limitations.
- Provides three deterministic local workloads for serious tuning runs without adding slow benchmarks to CI.

## Installation

v0.2.0 is distributed through GitHub Releases. PyPI publication is not part of this release.

```bash
pip install "git+https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer.git@v0.2.0"
```

For development from a clone:

```bash
git clone https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer.git
cd automatic-pytorch-dataloader-optimizer
python -m pip install -e ".[dev]"
```

Requirements:

- Python 3.10 or newer. Python 3.10 through 3.14 are tested.
- PyTorch 2.10 or newer.
- CPU-only and CUDA systems are supported. CUDA measurements require a working CUDA PyTorch build and driver.
- Map-style datasets only. `IterableDataset` is rejected explicitly.

## Python quick start: dataset

```python
import torch
from torch.utils.data import TensorDataset

from apdo import optimize

features = torch.randn(512, 32)
labels = torch.randint(0, 10, (512,))
dataset = TensorDataset(features, labels)

result = optimize(
    dataset,
    batch_size=32,
    device="cpu",
    rounds=3,
    validation_rounds=5,
    max_batches=50,
)
print(result.summary())

optimized_loader = result.build_dataloader()
batch = next(iter(optimized_loader))
```

## Python quick start: existing DataLoader

```python
from torch.utils.data import DataLoader

from apdo import optimize

original_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)
result = optimize(
    dataloader=original_loader,
    device="auto",
    rounds=3,
    validation_rounds=5,
)

print(result.validation.status)
print(result.best_config.describe())
optimized_loader = result.build_dataloader()
```

The original loader is not modified. APDO keeps the existing batch size and the loader behavior outside the four optimized options. A custom sampler, batch sampler, generator, or `worker_init_fn` is reused by reference, so mutable state can be shared with the original loader; the result includes a warning in that case.

`optimize` also accepts the loader as its first positional argument:

```python
result = optimize(original_loader, device="cpu")
```

Dataset and loader inputs are mutually exclusive. A `batch_size` override is rejected for a loader with an existing batch size. `DataLoader` subclasses are rejected because APDO cannot verify that their behavior is preserved.

### Preserved loader options

For an existing `DataLoader`, APDO preserves:

- `dataset`
- `batch_size`
- `sampler`
- `batch_sampler`
- `collate_fn`
- `drop_last`
- `timeout`
- `worker_init_fn`
- `generator`
- `multiprocessing_context` when workers are enabled
- `in_order`
- `pin_memory_device` when pinning is enabled

Only `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers` vary between candidates.

## How the recommendation is produced

1. A map-style dataset or loader is validated. The initial loader configuration becomes the safe fallback baseline.
2. Stage 1 scans `num_workers` values with default prefetch settings and applicable `pin_memory` choices.
3. Stage 2 varies `prefetch_factor` and `persistent_workers` for the two best multi-worker counts.
4. Configurations are measured in seeded, balanced rounds. A discarded warmup round is followed by one measured pass per configuration per round. The executed order is recorded.
5. The search selects the simplest configuration that is within 5% of the fastest median throughput and whose bootstrap interval overlaps the fastest interval.
6. The initial and search-candidate configurations are measured again in balanced paired rounds. Search measurements are not reused.
7. A paired bootstrap interval of relative gains determines the validation status.
8. The candidate is recommended only when the observed median gain meets `min_improvement` and the 95% interval is above zero. Otherwise the initial configuration is retained.
9. Validation that is stopped by the time budget before all planned rounds complete cannot confirm the candidate; the initial configuration is retained.

### Validation statuses

| Status | Meaning |
| --- | --- |
| `confirmed` | The candidate met the minimum gain and the paired interval is above zero. |
| `inconclusive` | The candidate did not establish the required gain. The initial configuration is retained. |
| `regression` | The candidate was slower in validation. The initial configuration is retained. |
| `not_needed` | The search selected the initial configuration. |
| `not_possible` | Validation was disabled, skipped by the time budget, failed, or had fewer than two paired rounds. The initial configuration is retained. |

## Important `optimize()` parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `dataset` | `None` | Map-style dataset or existing `DataLoader`. |
| `batch_size` | `32` for datasets | Required for dataset input; must match an existing loader if provided. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, or `"cuda:N"`. |
| `warmup_runs` | `1` | Discarded warmup rounds per configuration. Must be at least 1. |
| `repetitions` | `3` | Legacy number of measured search rounds. |
| `rounds` | `None` | Measured balanced search rounds; overrides `repetitions`. |
| `max_batches` | `100` | Batches per pass, or `None` for the full dataset. |
| `max_workers` | available CPUs, up to 8 | Upper bound for the worker search. |
| `seed` | `42` | Seed for execution order and bootstrap resampling. |
| `validation` | `True` | Enables independent paired validation. |
| `validation_rounds` | `5` | Paired rounds for validation. |
| `min_improvement` | `0.05` | Required relative gain, as a fraction. |
| `time_budget_s` | `None` | Approximate search budget, checked at round boundaries. |
| `sample_count_fn` | `None` | Callable that returns the number of samples in a batch. |
| `progress` | `None` | Callable receiving progress messages. |

## CLI

Dataset factory:

```bash
apdo benchmark \
  --dataset my_project.data:build_dataset \
  --batch-size 32 \
  --device auto \
  --rounds 3 \
  --validation-rounds 5 \
  --json report.json \
  --html report.html
```

DataLoader factory:

```bash
apdo benchmark \
  --dataset my_project.data:build_loader \
  --device auto \
  --no-validation \
  --json loader-report.json
```

The factory must be a zero-argument callable in an importable module. The current working directory is added to `sys.path`. On Windows, dataset classes must be defined at module level and guarded by the usual `if __name__ == "__main__"` pattern in scripts that start worker processes.

Useful options:

- `--seed`
- `--rounds`
- `--repetitions`
- `--validation-rounds`
- `--min-improvement`
- `--no-validation`
- `--time-budget-s`
- `--sample-count-fn MODULE:FACTORY`
- `--max-batches` or `--full-dataset`
- `--max-workers`
- `--json PATH`
- `--html PATH`

## Reports

JSON reports use schema version 2 and include:

- APDO version, start time, duration, protocol version, seed, rounds, budget state, and executed search/validation orders.
- Environment, benchmark settings, loader preservation details, method, and limitations.
- Baseline, search recommendation, fastest configuration, final recommendation, validation status, and final gain.
- Per-configuration per-round metrics, aggregate summaries, and failures.
- Complete validation observations and bootstrap results.

HTML reports are self-contained, contain no remote assets, escape benchmark data, and include a dedicated `Recommendation Validation` section.

## When the candidate is not confirmed

The final recommendation is safe by default. If validation is inconclusive, regresses, is disabled, or cannot run, `result.best_config` remains the initial configuration. The search candidate and search gain remain available in `result.search_candidate_config` and `result.search_improvement_percent` for inspection.

```python
if result.validation.status != "confirmed":
    print("Using the original configuration:", result.best_config.describe())
    print("Search candidate was:", result.search_candidate_config.describe())
```

## Benchmark workloads

The repository includes three deterministic local workloads designed for tuning sessions rather than CI:

- `in-memory`: preloaded tensor features with light preprocessing.
- `disk-backed`: generated local binary records with deterministic random access.
- `cpu-heavy`: per-sample matrix operations that make worker parallelism visible.

Run all workloads:

```bash
python benchmarks/run_workloads.py
```

Fast smoke run:

```bash
python benchmarks/run_workloads.py --workload in-memory --rounds 2 --validation-rounds 2 --max-batches 10
```

Generated data and reports are written to ignored directories under `benchmarks/.cache/` and `benchmarks/results/`. The workloads require no network access.

## Development

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests benchmarks examples
python -m ruff format --check src tests benchmarks examples
python -m pytest
```

CUDA tests are marked and can be deselected on CPU-only machines:

```bash
python -m pytest -m "not cuda"
```

Build and inspect the release artifacts:

```bash
python -m build
python -m twine check dist/*
```

## Compatibility and limitations

- v0.1.0 calls to `optimize(dataset, batch_size=...)` remain supported.
- The two-stage search, bootstrap selection, memory guard, CUDA transfer measurement, CLI, and reports remain available.
- v0.2.0 changes the JSON schema to version 2 and adds validation fields.
- Only map-style datasets with `__len__` and `__getitem__` are supported.
- Validation confirms a gain for the measured workload; it is not a universal guarantee.
- Measurements cover data loading and CUDA host-to-device copies, not model training.
- `max_ram_gb` is a sampled guard, not a hard memory limit.
- APDO does not reset dataset, sampler, transform, global RNG state, or GPU memory pools.
- macOS is not covered by CI in this release.

See `CHANGELOG.md` for release history. Licensed under MIT.
