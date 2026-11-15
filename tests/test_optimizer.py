from __future__ import annotations

import pytest
import torch

from apdo import LoaderConfig, optimize
from apdo.errors import BenchmarkError, ConfigurationError
from apdo.optimizer import (
    _build_worker_counts,
    _finalist_workers,
    _select_configuration,
    _stage_one_configs,
    resolve_device,
)
from tests.datasets import SyntheticDataset
from tests.helpers import make_failed_measurement, make_measurement


def test_build_worker_counts_is_bounded() -> None:
    assert _build_worker_counts(8, 16) == [0, 1, 2, 4, 8]
    assert _build_worker_counts(3, 16) == [0, 1, 2, 3]
    assert _build_worker_counts(1, 16) == [0, 1]
    assert _build_worker_counts(8, 1) == [0, 1]


def test_stage_one_configs_cover_pin_variants() -> None:
    configs = _stage_one_configs([0, 1, 4], [False, True])

    assert LoaderConfig() in configs
    assert LoaderConfig(1, False, 2, False) in configs
    assert LoaderConfig(1, True, 2, False) in configs
    assert LoaderConfig(4, False, 2, False) in configs
    assert LoaderConfig(4, True, 2, False) in configs
    assert len(configs) == 5


def test_finalist_workers_are_ranked_and_exclude_zero() -> None:
    measurements = [
        make_measurement(LoaderConfig(), (50.0, 51.0, 50.0)),
        make_measurement(LoaderConfig(1), (80.0, 81.0, 80.0)),
        make_measurement(LoaderConfig(2), (100.0, 101.0, 100.0)),
        make_measurement(LoaderConfig(4), (90.0, 91.0, 90.0)),
        make_failed_measurement(LoaderConfig(8)),
    ]

    assert _finalist_workers(measurements) == [2, 4]


def test_selection_prefers_simple_when_intervals_overlap() -> None:
    fast = make_measurement(
        LoaderConfig(4),
        (100.0, 101.0, 100.0),
        ci_low=95.0,
        ci_high=105.0,
    )
    simple = make_measurement(
        LoaderConfig(1),
        (99.0, 100.0, 99.0),
        ci_low=94.0,
        ci_high=104.0,
    )

    selection = _select_configuration([fast, simple])

    assert selection.fastest.config == fast.config
    assert selection.selected.config == simple.config
    assert selection.equivalent == (fast, simple)


def test_selection_keeps_clearly_faster_configuration() -> None:
    fast = make_measurement(
        LoaderConfig(4),
        (100.0, 101.0, 100.0),
        ci_low=99.0,
        ci_high=101.0,
    )
    slow = make_measurement(
        LoaderConfig(1),
        (50.0, 51.0, 50.0),
        ci_low=49.0,
        ci_high=51.0,
    )

    selection = _select_configuration([fast, slow])

    assert selection.selected.config == fast.config
    assert selection.equivalent == (fast,)


def test_selection_requires_material_difference_in_addition_to_overlap() -> None:
    fast = make_measurement(
        LoaderConfig(4),
        (100.0, 100.0, 100.0),
        ci_low=95.0,
        ci_high=105.0,
    )
    slower = make_measurement(
        LoaderConfig(1),
        (90.0, 90.0, 90.0),
        ci_low=85.0,
        ci_high=95.0,
    )

    selection = _select_configuration([fast, slower])

    assert selection.selected.config == fast.config
    assert slower not in selection.equivalent


def test_selection_handles_single_configuration() -> None:
    only = make_measurement(LoaderConfig(), (75.0, 76.0, 75.0))

    selection = _select_configuration([only])

    assert selection.selected.config == only.config
    assert selection.fastest.config == only.config


def test_selection_without_success_raises() -> None:
    with pytest.raises(BenchmarkError):
        _select_configuration([make_failed_measurement(LoaderConfig())])


def test_resolve_cpu_and_reject_unknown_device() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ConfigurationError):
        resolve_device("tpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="no CUDA-only path to exercise")
def test_resolve_cuda_without_gpu_raises() -> None:
    with pytest.raises(ConfigurationError):
        resolve_device("cuda")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": 1.5},
        {"repetitions": 1},
        {"warmup_runs": 0},
        {"max_batches": 1},
        {"max_workers": 0},
        {"max_ram_gb": 0},
        {"max_ram_gb": float("inf")},
        {"progress": 1},
    ],
)
def test_optimize_rejects_invalid_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        optimize(SyntheticDataset(size=16), **kwargs)
