from __future__ import annotations

from importlib.resources import files

import apdo


def test_version_is_exposed() -> None:
    assert apdo.__version__ == "0.4.1"


def test_py_typed_marker_is_installed() -> None:
    assert files("apdo").joinpath("py.typed").is_file()


def test_stability_and_comparability_apis_are_exported() -> None:
    assert callable(apdo.evaluate_stability)
    assert apdo.StabilityReport is not None
    assert apdo.ComparabilityResult is not None
    assert callable(apdo.compare_measurements)
    assert apdo.TIMING_PROFILES == ("standard", "detailed")
    assert apdo.SELECTION_METRICS == ("sustained", "amortized")
    assert apdo.COMPARABILITY_POLICIES == ("off", "warn", "strict")
