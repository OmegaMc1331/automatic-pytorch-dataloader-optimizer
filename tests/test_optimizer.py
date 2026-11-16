from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.utils.data import DataLoader

import apdo.optimizer as optimizer_module
from apdo import LoaderConfig, optimize
from apdo.benchmark import BatchPlan, RoundMeasurement, StageResult
from apdo.errors import BenchmarkError, ConfigurationError
from apdo.loader import LoaderTemplate, initial_loader_config
from apdo.optimizer import (
    _build_worker_counts,
    _finalist_workers,
    _select_configuration,
    _Selection,
    _stage_one_configs,
    resolve_device,
)
from tests.datasets import RaisingDataset, SyntheticDataset
from tests.helpers import make_failed_measurement, make_measurement, make_pass_metrics


def test_build_worker_counts_is_bounded() -> None:
    assert _build_worker_counts(8, 16) == [0, 1, 2, 4, 8]
    assert _build_worker_counts(3, 16) == [0, 1, 2, 3]
    assert _build_worker_counts(1, 16) == [0, 1]
    assert _build_worker_counts(8, 1) == [0, 1]


def test_stage_one_configs_cover_pin_variants() -> None:
    configs = _stage_one_configs([0, 1, 4], [False, True])

    assert LoaderConfig() in configs
    assert LoaderConfig(0, True) in configs
    assert LoaderConfig(1, False, 2, False) in configs
    assert LoaderConfig(1, True, 2, False) in configs
    assert LoaderConfig(4, False, 2, False) in configs
    assert LoaderConfig(4, True, 2, False) in configs
    assert len(configs) == 6


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


def test_optimize_preserves_a_custom_dataloader(tmp_path) -> None:
    dataset = SyntheticDataset(size=64, feature_dim=4)
    generator = torch.Generator().manual_seed(11)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=True,
        generator=generator,
        num_workers=0,
    )
    initial = initial_loader_config(loader)
    messages: list[str] = []

    result = optimize(
        loader,
        device="cpu",
        warmup_runs=1,
        repetitions=2,
        validation_rounds=2,
        max_batches=4,
        max_workers=1,
        seed=3,
        progress=messages.append,
    )

    assert result.settings.input_kind == "dataloader"
    assert result.settings.batch_size == 8
    assert result.settings.initial_config == initial
    assert result.validation.baseline_config == initial
    if result.validation.confirmed:
        assert result.best_config == result.validation.candidate_config
    else:
        assert result.best_config == initial
        assert result.improvement_percent == 0.0

    payload = result.to_dict()
    assert payload["loader"]["batch_size"] == 8
    assert payload["loader"]["drop_last"] is True
    assert payload["loader"]["generator"] is True
    html_path = tmp_path / "custom-loader.html"
    result.to_html(html_path)
    assert "Loader batch sampler" in html_path.read_text(encoding="utf-8")

    rebuilt = result.build_dataloader()
    assert rebuilt.dataset is dataset
    assert rebuilt.batch_size == 8
    assert rebuilt.drop_last is True
    assert rebuilt.generator is generator
    assert loader.batch_size == 8
    assert loader.num_workers == 0
    assert next(iter(loader))[0].shape == (8, 4)

    assert any(order.phase == "worker_scan" for order in result.round_orders)
    if result.validation.candidate_config != initial:
        assert any(order.phase == "validation" for order in result.round_orders)
    assert any("Validation status:" in message for message in messages)


def test_optimize_runs_validation_for_a_noninitial_candidate(monkeypatch) -> None:
    original = optimizer_module._select_configuration

    def force_worker_candidate(measurements):
        selection = original(measurements)
        candidate = next(
            (item for item in measurements if item.succeeded and item.config.num_workers > 0),
            None,
        )
        if candidate is None:
            return selection
        return _Selection(
            selected=candidate,
            fastest=selection.fastest,
            equivalent=selection.equivalent,
            reason=selection.reason,
        )

    monkeypatch.setattr(optimizer_module, "_select_configuration", force_worker_candidate)

    result = optimize(
        SyntheticDataset(size=64, feature_dim=4, delay_s=0.0004),
        batch_size=8,
        device="cpu",
        warmup_runs=1,
        repetitions=2,
        validation_rounds=2,
        max_batches=4,
        max_workers=1,
        seed=3,
    )

    assert result.validation.candidate_config != result.settings.initial_config
    assert any(order.phase == "validation" for order in result.round_orders)
    if result.validation.confirmed:
        assert result.best_config == result.validation.candidate_config
    else:
        assert result.best_config == result.settings.initial_config


def test_optimize_rejects_conflicting_inputs() -> None:
    dataset = SyntheticDataset(size=16)
    loader = DataLoader(dataset, batch_size=4)

    with pytest.raises(ConfigurationError, match="not both"):
        optimize(dataset, dataloader=loader)
    with pytest.raises(ConfigurationError, match="batch_size cannot override"):
        optimize(loader, batch_size=8)
    with pytest.raises(ConfigurationError, match="dataset or a dataloader"):
        optimize()


def test_optimize_fails_when_initial_configuration_cannot_run() -> None:
    with pytest.raises(BenchmarkError):
        optimize(
            RaisingDataset(size=16),
            batch_size=4,
            device="cpu",
            warmup_runs=1,
            repetitions=2,
            max_batches=2,
            max_workers=1,
        )


def _failing_worker_init(worker_id: int) -> None:
    raise RuntimeError(f"worker {worker_id} failed during initialization")


def test_optimize_requires_a_working_initial_loader_for_fallback() -> None:
    loader = DataLoader(
        SyntheticDataset(size=16, feature_dim=4),
        batch_size=4,
        num_workers=2,
        worker_init_fn=_failing_worker_init,
    )

    with pytest.raises(BenchmarkError, match="safe fallback baseline"):
        optimize(
            loader,
            device="cpu",
            warmup_runs=1,
            repetitions=2,
            validation=False,
            max_batches=2,
            max_workers=1,
        )


def test_partial_validation_budget_cannot_confirm_a_candidate(monkeypatch) -> None:
    baseline = LoaderConfig()
    candidate = LoaderConfig(num_workers=1, prefetch_factor=2)
    base_measurement = make_measurement(baseline, (100.0, 100.0, 100.0))
    candidate_measurement = make_measurement(candidate, (200.0, 200.0, 200.0))

    def with_rounds(measurement, throughput: float):
        rounds = tuple(
            RoundMeasurement(
                round_index=index,
                position=index,
                pass_metrics=make_pass_metrics(throughput),
                cold_start_s=measurement.cold_start_s,
                memory=measurement.memory,
            )
            for index in range(3)
        )
        return replace(measurement, rounds=rounds)

    def partial_stage(*args, **kwargs):
        return StageResult(
            measurements=(
                with_rounds(base_measurement, 100.0),
                with_rounds(candidate_measurement, 200.0),
            ),
            orders=(),
            budget_exhausted=True,
        )

    monkeypatch.setattr(optimizer_module, "benchmark_rounds", partial_stage)
    template = LoaderTemplate.from_dataset(SyntheticDataset(size=16, feature_dim=4), 4)

    result, orders, budget = optimizer_module._run_validation(
        template=template,
        baseline_config=baseline,
        candidate_config=candidate,
        enabled=True,
        validation_rounds=5,
        warmup_rounds=0,
        min_improvement=0.05,
        device=torch.device("cpu"),
        plan=BatchPlan(batches=4, steady_samples=12),
        max_ram_bytes=None,
        seed=1,
        deadline=None,
        progress=None,
    )

    assert budget is True
    assert orders == ()
    assert result.status == "not_possible"
    assert not result.confirmed
    assert result.observations == ()


def test_optimize_keeps_initial_configuration_without_validation() -> None:
    result = optimize(
        SyntheticDataset(size=32, feature_dim=4, delay_s=0.0002),
        batch_size=8,
        device="cpu",
        warmup_runs=1,
        repetitions=2,
        validation=False,
        max_batches=3,
        max_workers=1,
        seed=4,
    )

    assert result.validation.status in {"not_needed", "not_possible"}
    assert result.best_config == result.settings.initial_config
    assert result.improvement_percent == 0.0


def test_optimize_reports_exhausted_time_budget() -> None:
    result = optimize(
        SyntheticDataset(size=32, feature_dim=4),
        batch_size=8,
        device="cpu",
        warmup_runs=1,
        repetitions=2,
        validation_rounds=2,
        max_batches=3,
        max_workers=1,
        time_budget_s=1e-9,
        seed=4,
    )

    assert result.budget_exhausted
    assert result.best_config == result.settings.initial_config
    assert any("time budget" in warning.lower() for warning in result.warnings)
