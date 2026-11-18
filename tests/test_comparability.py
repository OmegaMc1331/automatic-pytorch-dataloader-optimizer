from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from apdo.benchmark import ConfigMeasurement, RoundMeasurement
from apdo.comparability import (
    BatchObservations,
    batch_fingerprint,
    batch_structure,
    compare_measurements,
)
from apdo.config import LoaderConfig
from apdo.errors import FingerprintError
from apdo.metrics import CycleMetrics
from tests.helpers import make_measurement


def _measurement(observations: BatchObservations | None) -> ConfigMeasurement:
    base = make_measurement(LoaderConfig(), (100.0, 101.0, 100.0))
    if observations is None:
        cycles: tuple[CycleMetrics, ...] = ()
    else:
        cycle = CycleMetrics(
            cycle_index=0,
            startup_s=0.01,
            duration_s=1.0,
            batches=len(observations.structures),
            samples=sum(observations.sample_counts),
            throughput=100.0,
            steady_elapsed_s=1.0,
            steady_samples=sum(observations.sample_counts),
            steady_throughput=100.0,
            latencies_s=(0.01,),
            batch_observations=observations,
        )
        cycles = (cycle,)
    passes = (
        replace(
            base.passes[0],
            cycles=cycles,
            total_samples=sum(observations.sample_counts) if observations else 0,
        ),
    )
    return replace(
        base, passes=passes, rounds=(RoundMeasurement(0, 0, passes[0], 0.1, base.memory),)
    )


def test_batch_structure_tracks_dtype_and_shape() -> None:
    left = batch_structure((torch.zeros(2, 3), [1, 2]))
    right = batch_structure((torch.zeros(2, 3), [1, 2]))
    wrong_shape = batch_structure((torch.zeros(3, 2), [1, 2]))
    wrong_dtype = batch_structure((torch.zeros(2, 3, dtype=torch.float64), [1, 2]))

    assert left == right
    assert left != wrong_shape
    assert left != wrong_dtype


def test_batch_fingerprint_hashes_values_and_rejects_unsupported_values() -> None:
    assert batch_fingerprint((1, 2.5, "x")) == batch_fingerprint((1, 2.5, "x"))
    assert batch_fingerprint((1, 2)) != batch_fingerprint((2, 1))
    assert batch_fingerprint(torch.tensor(3)) == batch_fingerprint(torch.tensor(3))

    with pytest.raises(FingerprintError):
        batch_fingerprint(object())


def test_compare_measurements_reports_same_order_and_different_samples() -> None:
    same_left = _measurement(BatchObservations(("s1", "s2"), (2, 2), ("f1", "f2")))
    same_right = _measurement(BatchObservations(("s1", "s2"), (2, 2), ("f1", "f2")))
    same = compare_measurements(same_left, same_right)
    assert same.status == "same_order"
    assert same.structural_verified
    assert same.sample_identity_verified
    assert same.sufficient_for_strict

    different = compare_measurements(
        same_left,
        _measurement(BatchObservations(("s1", "s2"), (2, 2), ("f1", "f9"))),
    )
    assert different.status == "different_samples"
    assert different.has_observed_problem
    assert not different.sufficient_for_strict


def test_compare_measurements_reports_unknown_without_fingerprints() -> None:
    left = _measurement(BatchObservations(("s1", "s2"), (2, 2), None))
    right = _measurement(BatchObservations(("s1", "s2"), (2, 2), None))

    comparison = compare_measurements(left, right)

    assert comparison.status == "unknown"
    assert comparison.structural_verified
    assert not comparison.sample_identity_verified
    assert not comparison.sufficient_for_strict
    assert comparison.confidence == "partial"
