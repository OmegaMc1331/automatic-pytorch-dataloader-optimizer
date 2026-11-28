# Automatic PyTorch DataLoader Optimizer

[![CI](https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`apdo` measures `torch.utils.data.DataLoader` configurations on the machine where it runs and recommends a setting for `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`.

In loader mode it reports sustained data-loading throughput. In training mode it runs the user's forward, backward, and optimizer step and recommends the loader that delivers the highest end-to-end training throughput.

The package name is `apdo`. The command-line interface is also `apdo`.

## Why APDO

DataLoader throughput depends on the dataset, storage, preprocessing cost, CPU topology, operating system, worker process behavior, and the device receiving the batches. Published rules of thumb and configurations tuned on another machine often perform worse than the default.

APDO runs the real loader, records repeated measurements, validates the search recommendation with fresh measurements, and retains the initial configuration when a gain is not confirmed. It does not estimate performance from hardware specifications and does not call a cloud service.

## Features in v0.5.0

- Optimizes a map-style dataset with a batch size, an existing `DataLoader`, or a config-aware loader factory.
- Measures repeated epochs in one trial and reports iterator initialization, first-batch acquisition, worker shutdown, loader cleanup, epoch transitions, sustained throughput, and total trial wall separately.
- Supports a detailed timing profile with CUDA Events around transfers and training steps without per-batch synchronization.
- Supports an explicit amortized metric that divides all trial samples by the complete trial wall, including construction, startup, transitions, shutdown, and cleanup.
- Records batch structure and optional sample fingerprints for paired comparability diagnostics under `off`, `warn`, or `strict` policies.
- Provides repeated independent optimizations through `evaluate_stability` and `apdo stability`, retaining partial failures, separating confirmed gains from fallbacks, and summarizing recommendation agreement.
- Supports end-to-end training mode with a user-supplied training-step factory covering forward, backward, and optimizer work.
- Preserves the original loader's dataset, batch size, sampler, batch sampler, collate function, `drop_last`, timeout, worker initialization, generator, multiprocessing context, and `in_order` setting.
- Changes only `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`.
- Isolates supported loader generators and samplers, restores Python/Torch/NumPy/CUDA RNG state after each trial, and reports a reproducibility status.
- Runs seeded, balanced rounds and records the executed configuration order and every per-round measurement.
- Reports worker PIDs, process RSS, data-wait time, host-to-device transfer time, training-step latency, and epoch-transition overhead.
- Reports effective CPU count, CPU affinity, cgroup memory/CPU limits, and the multiprocessing start method; caps worker search by effective CPU availability.
- Runs independent paired validation with a configurable `minimum` or `positive` policy instead of treating the search result as proof.
- Requires at least eight valid paired observations, an exact one-sided paired sign test, and the selected policy's interval condition before confirming a gain.
- Uses the selected metric consistently for worker finalists, training candidates, the search recommendation, and the report.
- Falls back to the initial configuration unless validation confirms the required minimum gain.
- Supports cooperative time budgets with validation reservation and explicit skipped-trial accounting.
- Offers grid search by default and an adaptive screening/refinement strategy that records search-stage decisions.
- Estimates profitability for a planned epoch and session horizon, with conservative and optimistic bounds.
- Supports an opt-in local JSON cache (`off`, `hint`, `revalidate`) whose hits never replace fresh validation.
- Supports custom sample counting for unusual collate functions.
- Generates a schema-v5 JSON report and a self-contained HTML report with lifecycle timing, comparability, search trace, budget, profitability, cache, training metrics, reproducibility, validation, ranking, execution order, environment, method, and limitations.
- Provides six deterministic local workloads, including startup-bound, compute-bound, and real training workloads, without adding slow benchmarks to CI.

## Installation

v0.5.0 is distributed through GitHub Releases. PyPI publication is not part of this release.

```bash
pip install "git+https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer.git@v0.5.0"
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
    validation_rounds=10,
    max_batches=50,
)
print(result.summary())

optimized_loader = result.build_dataloader()
batch = next(iter(optimized_loader))
```

## Adaptive search, budgets, cache, and profitability

Grid search remains the default. Adaptive search screens worker counts with fewer rounds and refines one worker finalist before the same independent validation gate:

```python
result = optimize(
    dataset,
    batch_size=32,
    search_strategy="adaptive",
    time_budget_s=120,
    planned_epochs=50,
    planned_sessions=1,
    cache_policy="hint",
)

print(result.search_trace.summary())
print(result.budget["status"])
print(result.profitability.summary())
```

A budget is cooperative: APDO checks it at measured round boundaries, reserves capacity for validation, and never treats partial evidence as confirmation. `result.estimate_profitability(planned_epochs=100, planned_sessions=2)` can be called later with a different horizon.

For one-epoch or startup-sensitive workloads, use `selection_metric="amortized"`. A candidate can pass sustained-throughput validation while making the complete trial slower; the report warns when that divergence is material, and the profitability estimate can mark the recommendation `not_profitable`.

The cache is opt-in and local. `hint` uses a matching recommendation to focus search but still validates fresh; `revalidate` behaves similarly with explicit revalidation semantics; `off` disables cache access. Inspect or clear it with:

```bash
apdo cache list
apdo cache clear --yes
```

Cache entries are JSON only, bounded, and include protocol, package, environment, source, and validation evidence. Corrupt or stale entries are ignored.

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
    validation_rounds=10,
)

print(result.validation.status)
print(result.best_config.describe())
optimized_loader = result.build_dataloader()
```

The original loader is not modified. APDO keeps the existing batch size and the loader behavior outside the four optimized options. A standard generator, sampler, and batch sampler are reconstructed for each trial. If a custom sampler or batch sampler cannot be reconstructed safely, APDO marks reproducibility as not guaranteed and refuses training mode.

`optimize` also accepts the loader as its first positional argument:

```python
result = optimize(original_loader, device="cpu")
```

Dataset and loader inputs are mutually exclusive. A `batch_size` override is rejected for a loader with an existing batch size. `DataLoader` subclasses are rejected because APDO cannot verify that their behavior is preserved.

## Python quick start: end-to-end training

Training mode is explicit. APDO never starts model training unless `mode="training"` and a `training_step_factory` are provided.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from apdo import optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TensorDataset(torch.randn(2048, 32), torch.randint(0, 10, (2048,)))


def create_training_step():
    model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def step(batch):
        features, labels = batch
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), labels)
        loss.backward()
        optimizer.step()

    return step


result = optimize(
    dataset,
    batch_size=64,
    device=device,
    mode="training",
    epochs=3,
    training_step_factory=create_training_step,
    rounds=3,
    training_rounds=2,
    validation_rounds=10,
)
print(result.to_dict()["training"])
print(result.best_config.describe())
```

The factory takes no arguments and is called once per measured trial. It must create a fresh model, optimizer, and any other mutable training state on each call. APDO transfers supported batch tensor structures to the resolved device before calling the returned `step(batch)`. Training mode is refused when the loader or dataset cannot be isolated sufficiently for comparable trials.

## Config-aware loader factory

Use a loader factory when loader or dataset state must be created fresh for every trial:

```python
from torch.utils.data import DataLoader

from apdo import LoaderConfig, optimize


def build_loader(config: LoaderConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        **config.to_dataloader_kwargs(),
    )


result = optimize(
    dataloader_factory=build_loader,
    device="auto",
    mode="training",
    training_step_factory=create_training_step,
)
```

The factory must return a standard `DataLoader` that honors the requested `LoaderConfig`. The initial configuration is `LoaderConfig()`. Factory-managed state cannot be inspected by APDO, so the result reports reproducibility status `factory_managed`.

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

1. A map-style dataset, loader, or config-aware factory is validated. The initial loader configuration becomes the safe fallback baseline.
2. Loader search stage 1 scans `num_workers` values with default prefetch settings and applicable `pin_memory` choices.
3. Loader search stage 2 varies `prefetch_factor` and `persistent_workers` for the two best multi-worker counts.
4. In training mode, stage 1 and stage 2 measure data loading with one epoch. The initial configuration, the fastest loader configuration, and the highest-ranked alternatives then run the user's training step for the requested number of epochs.
5. Configurations are measured in seeded, balanced rounds. A discarded warmup round is followed by one measured pass per configuration per round. The executed order is recorded.
6. Every cycle records iterator initialization, first-batch acquisition, execution, and worker shutdown; the complete trial wall also includes loader construction and final cleanup. Sustained throughput excludes the first loader batch, while the amortized metric uses total samples divided by the complete trial wall.
7. Training mode ranks candidates by the selected end-to-end training metric, sustained by default. The search recommendation is not based on loader-only throughput.
8. The initial and selected configurations are measured again in balanced paired rounds. Search measurements are not reused.
9. Observations that are structurally incomparable, incomplete, non-finite, or non-positive are excluded and reported; they are never replaced with zeros.
10. A paired bootstrap interval and an exact one-sided paired sign test determine the validation status under the selected policy. At least eight valid pairs are required.
11. Validation that is stopped by the time budget before all planned rounds complete cannot confirm the candidate; the initial configuration is retained. Validation that produces fewer than eight valid pairs is `not_possible`.

### Validation policies

- `minimum` requires the observed median gain and the lower 95% interval bound to reach `min_improvement`. This is the default and guarantees the configured threshold at the lower interval bound.
- `positive` requires the observed median gain to reach `min_improvement` and the interval to be strictly above zero. It allows a statistically positive gain that may not guarantee the full threshold.
- Both policies also require the exact one-sided paired sign test to reject the null hypothesis at `alpha = 0.05`. A gain exactly at the threshold is counted as a failure for the sign test.

### Validation statuses

| Status | Meaning |
| --- | --- |
| `confirmed` | The candidate met the selected policy. |
| `inconclusive` | The candidate did not establish the required gain. The initial configuration is retained. |
| `regression` | The candidate was slower in validation. The initial configuration is retained. |
| `not_needed` | The search selected the initial configuration. |
| `not_possible` | Validation was disabled, failed, had fewer than eight valid paired observations, or was stopped by the time budget before all planned pairs completed. The initial configuration is retained. |

## Important `optimize()` parameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `dataset` | `None` | Map-style dataset or existing `DataLoader`. |
| `batch_size` | `32` for datasets | Required for dataset input; must match an existing loader if provided. |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, or `"cuda:N"`. |
| `dataloader` | `None` | Existing loader; mutually exclusive with `dataset` and `dataloader_factory`. |
| `dataloader_factory` | `None` | Callable taking `LoaderConfig` and returning a standard `DataLoader`. |
| `mode` | `"loader"` | `"loader"` or `"training"`. |
| `epochs` | `1` loader, `3` training | Epochs per measured trial; at most 20. |
| `training_step_factory` | `None` | Zero-argument callable returning a fresh `step(batch)` for training mode. |
| `training_rounds` | `2` | Measured balanced rounds for training finalists. |
| `training_validation_rounds` | `validation_rounds` | Paired validation rounds in training mode. |
| `max_training_candidates` | `3` | Maximum configurations promoted to the training stage. |
| `validation_policy` | `"minimum"` | `"minimum"` or `"positive"`. |
| `timing_profile` | `"standard"` | `"standard"` whole-cycle CUDA events or `"detailed"` transfer/step device events. |
| `selection_metric` | `"sustained"` | `"sustained"` execution throughput or `"amortized"` total-trial throughput. |
| `comparability_policy` | `"warn"` | `"off"`, `"warn"`, or `"strict"` batch/sample comparison policy. |
| `sample_fingerprint_fn` | `None` | Callable returning stable scalar or collection sample identifiers for comparability. |
| `warmup_runs` | `1` | Discarded warmup rounds per configuration. Must be at least 1. |
| `repetitions` | `3` | Legacy number of measured search rounds. |
| `rounds` | `None` | Measured balanced search rounds; overrides `repetitions`. |
| `max_batches` | `100` | Batches per epoch, or `None` for the full dataset. |
| `max_workers` | available CPUs, up to 8 | Upper bound for the worker search. |
| `seed` | `42` | Seed for execution order, RNG isolation, and bootstrap resampling. |
| `validation` | `True` | Enables independent paired validation. |
| `validation_rounds` | `10` | Planned paired rounds for validation; at least eight valid pairs are required to confirm a gain. |
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
  --validation-rounds 10 \
  --json report.json \
  --html report.html
```

Config-aware loader factory:

```bash
apdo benchmark \
  --factory my_project.data:build_loader \
  --device auto \
  --rounds 3 \
  --validation-rounds 10 \
  --json loader-report.json
```

End-to-end training with a config-aware factory:

```bash
apdo benchmark \
  --factory my_project.data:build_train_loader \
  --mode training \
  --epochs 3 \
  --training-step-factory my_project.train:create_training_step \
  --training-rounds 2 \
  --validation-policy minimum \
  --device cuda \
  --json training-report.json \
  --html training-report.html
```

`--dataset MODULE:FACTORY` accepts a zero-argument factory returning a dataset or an existing `DataLoader`. `--factory MODULE:FACTORY` accepts a factory that takes a `LoaderConfig`. The current working directory is added to `sys.path`. On Windows, dataset and model classes must be defined at module level, and scripts that start worker processes need the usual `if __name__ == "__main__"` guard.

Useful options:

- `--mode loader|training`
- `--epochs N`
- `--timing-profile standard|detailed`
- `--selection-metric sustained|amortized`
- `--comparability-policy off|warn|strict`
- `--sample-fingerprint MODULE:FACTORY`
- `--training-step-factory MODULE:FACTORY`
- `--training-rounds N`
- `--training-validation-rounds N`
- `--max-training-candidates N`
- `--validation-policy minimum|positive`
- `--seed`
- `--rounds`
- `--repetitions`
- `--validation-rounds`
- `--min-improvement`
- `--no-validation`
- `--time-budget-s`
- `--search-strategy grid|adaptive`
- `--planned-epochs N`
- `--planned-sessions N`
- `--cache-policy off|hint|revalidate`
- `--cache-key KEY`
- `--cache-dir PATH`
- `--sample-count-fn MODULE:FACTORY`
- `--max-batches` or `--full-dataset`
- `--max-workers`
- `--json PATH`
- `--html PATH`

## Reports

JSON reports use schema version 5 and include:

- APDO version, start time, duration, protocol version 5.0, benchmark mode, epochs, seed, rounds, timing profile, selection metric, comparability policy, search strategy, cache policy, budget state, and executed search/validation orders.
- Lifecycle timing for construction, iterator initialization, first batch, worker shutdown, loader cleanup, total trial wall, sustained throughput, and amortized throughput.
- Search strategy, per-stage candidate statuses, budget status and category timing, profitability bounds, and cache status.
- Batch structure and optional sample-fingerprint comparability results.
- Environment, benchmark settings, loader preservation details, reproducibility status and risks, method, and limitations.
- Baseline, search recommendation, fastest configuration, final recommendation, validation status, policy, and final gain.
- Per-configuration per-round metrics, per-epoch cycle metrics, aggregate summaries, and failures.
- Training details with end-to-end throughput, step latency, data wait, transfer time, and selected configuration summaries.
- Complete validation observations, rejected-observation reasons, planned and completed pair counts, bootstrap results, statistical constants, sign-test p-value, and sustained and amortized selection values.

HTML reports are self-contained, contain no remote assets, escape benchmark data, and include dedicated recommendation-validation, comparability, timing-breakdown, training, and reproducibility sections.

## When the candidate is not confirmed

The final recommendation is safe by default. If validation is inconclusive, regresses, is disabled, or cannot run, `result.best_config` remains the initial configuration. The search candidate and search gain remain available in `result.search_candidate_config` and `result.search_improvement_percent` for inspection.

```python
if result.validation.status != "confirmed":
    print("Using the original configuration:", result.best_config.describe())
    print("Search candidate was:", result.search_candidate_config.describe())
```

## Benchmark workloads

The repository includes six deterministic local workloads designed for tuning sessions rather than CI:

- `in-memory`: preloaded tensor features with light preprocessing.
- `disk-backed`: generated local binary records with deterministic random access.
- `cpu-heavy`: per-sample matrix operations that make worker parallelism visible.
- `startup-bound`: trivial per-sample work with multiple loader epochs so construction and worker startup dominate the trial.
- `compute-bound`: a deterministic MLP forward pass, cross-entropy loss, backward pass, and optimizer step over cheap data, so compute rather than loading dominates.
- `training`: a real MLP forward pass, cross-entropy loss, backward pass, and optimizer step.

Run all workloads:

```bash
python benchmarks/run_workloads.py
```

Fast smoke run:

```bash
python benchmarks/run_workloads.py --workload in-memory --rounds 2 --validation-rounds 2 --max-batches 10
```

Training workload:

```bash
python benchmarks/run_workloads.py --workload training --device cuda --epochs 3 --training-rounds 2
```

Startup-sensitive run with multiple loader epochs:

```bash
python benchmarks/run_workloads.py --workload startup-bound --epochs 1 --loader-epochs 4
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

- v0.1.0 and v0.2.0 calls to `optimize(dataset, batch_size=...)` remain supported.
- v0.3.0 adds training mode, loader factories, multi-epoch metrics, reproducibility reporting, and JSON schema version 3 with protocol version 3.0.
- v0.4.0 adds explicit lifecycle timing, amortized selection, detailed CUDA timing, batch comparability diagnostics, stability reports, and JSON schema version 4 with protocol version 4.0.
- v0.4.1 adds metric-consistent selection and ranking, the eight-pair validation minimum with an exact paired sign test, explicit rejection of invalid validation observations, complete budget-truncated validation evidence, corrected startup-bound and compute-bound workloads, and confirmed-versus-fallback stability statistics with protocol version 4.1.
- v0.5.0 adds adaptive screening, cooperative budget categories and validation reservation, profitability estimates, an opt-in local JSON cache, search traces, schema version 5, and protocol version 5.0.
- The grid or adaptive search, bootstrap selection, memory guard, CUDA transfer measurement, CLI, and reports remain available.
- Only map-style datasets with `__len__` and `__getitem__` are supported.
- Loader mode measures data loading; it does not measure model training. Training mode measures the user-supplied model step, so model architecture and device work affect the result.
- Validation confirms a gain for the measured workload and policy; it is not a universal guarantee.
- APDO isolates supported loader generators and samplers and restores supported global RNG state. Arbitrary mutable state hidden inside dataset, transform, worker-initialization, or factory code cannot be fully inspected.
- Training mode refuses loaders whose detected isolation status is not reproducible enough. Factory-managed state is the user's responsibility.
- `max_ram_gb` is a sampled guard, not a hard memory limit.
- macOS is not covered by CI in this release.

See `CHANGELOG.md` for release history. Licensed under MIT.

## Stability analysis

Run the same optimization repeatedly with independent seeds:

```python
from apdo import evaluate_stability, optimize

report = evaluate_stability(
    lambda seed: optimize(
        dataset,
        batch_size=32,
        device="cpu",
        rounds=3,
        validation_rounds=10,
        seed=seed,
    ),
    runs=5,
    base_seed=42,
)
print(report.summary_text())
report.to_json("stability.json")
report.to_html("stability.html")
```

The stability report records each run's recommended configuration, search candidate, validation status, gain, throughput, comparability status, and any failure. It summarizes coverage, recommendation agreement, candidate agreement, best-configuration frequency, confirmed gain statistics, fallback counts, validation pair counts, throughput spread, and failure counts. Raw batch values are never stored.

An agreement rate alone is not evidence of a speedup: a run that retains the initial configuration or falls back after failed validation still counts toward recommendation agreement. Confirmed gains are counted separately, fallback runs retain their search-candidate gain in `improvement_percent`, and confirmed-only statistics use `median_confirmed_gain_percent`. The report warns when successful runs had too few valid validation pairs, when no run confirmed a gain, or when high agreement comes from fallback or initial-configuration choices. Qualification `complete` means every successful run contributed at least eight valid validation pairs; it says nothing about the size or usefulness of those gains.

CLI equivalent:

```bash
apdo stability \
  --dataset my_project.data:build_dataset \
  --runs 5 \
  --base-seed 42 \
  --json stability.json \
  --html stability.html
```
