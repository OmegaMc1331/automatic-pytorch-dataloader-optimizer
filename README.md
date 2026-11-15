# Automatic PyTorch DataLoader Optimizer

[![CI](https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

`apdo` measures `torch.utils.data.DataLoader` configurations on the machine where it runs and recommends a setting for `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers`.

The package name is `apdo`. The command-line interface is also `apdo`.

## The problem

DataLoader throughput depends on the dataset, storage, CPU topology, operating system, worker process behavior, and the device receiving the batches. Published rules of thumb and configurations tuned on another machine often perform worse than the default.

`apdo` runs the real loader, records repeated measurements, and selects a configuration using those measurements. It does not estimate performance from hardware specifications, and it does not call a cloud service.

## Features in v0.1.0

- Benchmarks map-style PyTorch datasets on CPU or CUDA.
- Searches `num_workers`, `pin_memory`, `prefetch_factor`, and `persistent_workers` with a two-stage grid.
- Measures startup latency, steady-state throughput, per-batch latency, parent-process RSS and system memory growth on Windows and Linux, and CUDA allocation during tensor copies.
- Excludes failed configurations instead of treating them as valid results.
- Selects a stable configuration with bootstrap confidence intervals and a preference for simpler settings.
- Exports a structured JSON report and a self-contained HTML report.
- Provides a Python API and an `apdo benchmark` command.
- Runs locally without authentication or external APIs.

## Installation

v0.1.0 is released on GitHub. PyPI publication is not part of this release.

```bash
pip install "git+https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer.git@v0.1.0"
```

For development from a clone:

```bash
git clone https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer.git
cd automatic-pytorch-dataloader-optimizer
python -m pip install -e ".[dev]"
```

Requirements:

- Python 3.10 or newer. Python 3.10 through 3.14 are tested.
- PyTorch 2.10 or newer. PyTorch 2.10 and 2.15 are tested.
- CPU-only and CUDA systems are supported. CUDA measurements require a working CUDA PyTorch build and driver.

## Python quick start

```python
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from apdo import optimize


def main() -> None:
    features = torch.randn(512, 32)
    labels = torch.randint(0, 10, (512,))
    dataset = TensorDataset(features, labels)

    result = optimize(
        dataset,
        batch_size=32,
        device="cpu",
        max_ram_gb=2,
        max_batches=50,
    )
    print(result.summary())

    loader = DataLoader(
        dataset,
        batch_size=32,
        **result.best_config.to_dataloader_kwargs(),
    )
    batch = next(iter(loader))
    print(len(batch), batch[0].shape)

    result.to_json(Path("reports/benchmark.json"))
    result.to_html(Path("reports/benchmark.html"))


if __name__ == "__main__":
    main()
```

On Windows, keep the `if __name__ == "__main__":` guard. DataLoader workers use the `spawn` start method there, and the guard prevents the benchmark from being re-executed in child processes.

`device="auto"` resolves to CUDA when CUDA is available and to CPU otherwise. On CUDA, throughput includes blocking host-to-device copies of tensor values in each batch. It does not include a model forward or backward pass.

The recommended configuration can be passed directly to a new loader:

```python
config = result.best_config
loader = DataLoader(train_dataset, batch_size=32, **config.to_dataloader_kwargs())
```

`result.fastest_config` contains the empirically fastest configuration. It can differ from `result.best_config` when the fastest result's bootstrap interval overlaps that of a simpler configuration within the 5% materiality band.

## Command line

```bash
apdo --help
apdo --version
apdo benchmark --help
```

A dataset factory is an importable zero-argument callable that returns a map-style dataset. The repository includes one at `examples/synthetic_dataset.py`:

```bash
apdo benchmark \
  --dataset examples.synthetic_dataset:create_dataset \
  --batch-size 32 \
  --device auto \
  --max-ram-gb 4 \
  --max-batches 100 \
  --max-workers 8 \
  --repetitions 3 \
  --json benchmark.json \
  --html benchmark.html
```

The CLI adds the current working directory to the import path when it loads the factory. Modules are imported only when named explicitly on the command line; `apdo` does not evaluate configuration files or expressions.

The factory module must be importable by worker processes as well. Define dataset and transform classes at module level, not inside a function or inside `if __name__ == "__main__":`.

Benchmark options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset MODULE:FACTORY` | required | Import path of a zero-argument dataset factory. |
| `--batch-size N` | `32` | Batch size used for every configuration. |
| `--device DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`. |
| `--max-ram-gb GB` | none | Soft limit on observed system memory growth during a configuration. |
| `--warmup-runs N` | `1` | Discarded passes before measured passes. |
| `--repetitions N` | `3` | Measured passes per configuration. Minimum 2. |
| `--max-batches N` | `100` | Maximum batches in each pass, including the first. Minimum 2. |
| `--full-dataset` | off | Iterate every batch instead of applying `--max-batches`. |
| `--max-workers N` | `min(8, available CPUs)` | Upper bound for the worker search. |
| `--json PATH` | none | Write the JSON report. Parent directories are created. |
| `--html PATH` | none | Write the standalone HTML report. Parent directories are created. |

Progress and warnings are written to stderr. The recommendation summary is written to stdout, so it can be redirected independently.

## Parameters under test

### `num_workers`

Candidate counts are drawn from `0, 1, 2, 4, 8`, plus the configured worker limit when it differs, capped by the available CPU count. The default limit is `min(8, available CPUs)`. Allowing more workers than CPUs is usually counterproductive and is not explored by default.

### `pin_memory`

`pin_memory` is explored only when the resolved device is CUDA. Pinning on a CPU-only run has no transfer path to improve, so only `False` is tested. CUDA configurations with and without pinning are compared using actual host-to-device copies.

### `prefetch_factor`

`prefetch_factor` is only legal when `num_workers > 0`. The search compares values `2` and `4`.

### `persistent_workers`

`persistent_workers` is only legal when `num_workers > 0`. Both values are tested for the selected worker counts in the second stage.

The default PyTorch configuration, `num_workers=0` with `pin_memory=False`, is always included and serves as the baseline for the reported improvement.

## How the benchmark works

1. The dataset is checked for map-style access and a positive length. A run needs at least two batches. Fewer than four batches produces a warning because startup effects and noise are difficult to separate.
2. A first-stage grid scans worker counts with `prefetch_factor=2`, `persistent_workers=False`, and both applicable pinning values. The default configuration is included.
3. The two best multi-worker counts by median throughput are refined in a second stage over `prefetch_factor` and `persistent_workers`.
4. Each configuration receives one or more discarded warmup passes, followed by at least two measured passes.
5. Cold start includes loader construction and the first worker startup. The startup statistic in each measured pass begins at `iter(loader)`, so it includes worker restarts for non-persistent loaders and iterator setup for persistent and single-process loaders. Steady-state throughput excludes the first batch, so initialization is reported separately from sustained loading.
6. Batch latency is measured around each steady-state `next(iterator)` call, including host-to-device copies on CUDA.
7. A background sampler records parent-process RSS and system-wide available memory every 50 ms on Windows and Linux. `max_ram_gb` compares observed system memory growth against the limit and stops the affected configuration when it is exceeded.
8. Configurations are benchmarked sequentially in the same process. Failed configurations are retained in the report with their error details and are excluded from selection.
9. The recommendation is the simplest configuration whose median throughput is within 5% of the fastest median and whose 95% bootstrap interval overlaps the fastest interval. Simplicity is ordered by fewer workers, no pinning, non-persistent workers, and lower prefetch.

All timings use `time.perf_counter()`. Throughput is reported in samples per second, not batches per second and not training iterations per second.

## Interpreting the result

`result.summary()` prints the recommended configuration, measured throughput, batch latencies, startup cost, observed memory, baseline improvement, and the reason for the selection. `result.to_dict()` exposes the same data in structured form.

A recommended configuration with fewer workers than `result.fastest_config` is not a mistake. A simpler configuration was also within 5% of the fastest median throughput and its 95% bootstrap interval overlapped the fastest interval, so the simpler configuration is preferred.

A warning that only one configuration succeeded means no comparison was possible. A warning about too few batches means the measurement window was short. Treat both cases as weak evidence.

The numbers are specific to the current machine, dataset, storage state, batch size, and system load. They are useful for choosing settings on that machine; they are not a general benchmark of PyTorch or of a model.

## Limitations in v0.1.0

- Only map-style datasets with `__len__` and `__getitem__` are supported. `IterableDataset` is rejected.
- Custom samplers, `collate_fn`, `worker_init_fn`, `drop_last`, `timeout`, and transforms that depend on loader options are not part of the search. The benchmark uses the dataset as provided and PyTorch's default collation.
- No model is executed. Throughput covers loading and, on CUDA, tensor copies to the device. It says nothing about training or inference speed.
- The grid is not exhaustive. Worker counts above `max_workers` are not considered, and interactions outside the staged search are not tested.
- `max_ram_gb` is a soft, sampled guard, not a reservation or a hard cgroup limit. Allocations between samples, memory held by other processes, and page-cache behavior affect what is observed. Parent-process RSS does not include worker processes; system-wide available memory is used to observe their effect indirectly.
- Parent-process RSS is absolute and can include memory retained from earlier configurations. Memory sampling is implemented for Windows and Linux.
- The same dataset object is reused across configurations. Dataset caching, state changes, or storage warm-up can make later configurations appear faster.
- Configurations run sequentially and in a deterministic order. Machine load that changes during the benchmark can favor earlier or later configurations.
- Bootstrap intervals describe variation across the measured passes. Interval overlap is an equivalence heuristic, not a formal hypothesis test. Intervals do not model variation from random transforms, changing storage caches, or other long-running system effects.
- The search is single-device. Multi-GPU transfers and distributed data loading are not measured.
- Only CPU and CUDA devices are accepted. MPS and other accelerator types are not supported in v0.1.0. On macOS, only total physical memory is reported and `max_ram_gb` cannot be enforced.
- `pin_memory` benefits are only observable when batches contain tensors and the resolved device is CUDA. A warning is reported when no tensors were transferred.
- Windows is supported and tested with multiprocessing `spawn`, but datasets and transforms must be importable from a module so worker processes can reconstruct them.
- CI covers lint, formatting, packaging, and CPU tests on Linux and Windows. The CUDA integration test runs locally on CUDA hardware and is not part of CI. macOS is not covered by CI in v0.1.0.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
pytest -m cuda          # only where a CUDA device is available
python -m build
twine check dist/*
```

The CUDA integration test is marked `cuda` and skipped when `torch.cuda.is_available()` is false. CPU tests exercise real `DataLoader` processes, including a failing worker, and a full `optimize()` run on a synthetic dataset.

The CI workflow runs lint, format checks, CPU tests, a Windows test job, and package build checks. The test matrix covers Python 3.10 through 3.14, using PyTorch 2.10 on Python 3.10 through 3.13 and PyTorch 2.15 on Python 3.14.

## Contributing

Bug reports and pull requests are welcome at
<https://github.com/OmegaMc1331/automatic-pytorch-dataloader-optimizer/issues>.

Please include the operating system, Python version, PyTorch version, CPU model, dataset characteristics, and command line with any performance report. Keep changes covered by tests and run the lint and test commands above before opening a pull request.

## License

MIT. See [LICENSE](LICENSE).
