from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from apdo import LoaderConfig
from apdo.search import SearchCandidate, SearchStage, SearchTrace
from tests.helpers import (
    make_failed_measurement,
    make_measurement,
    make_result,
    make_settings,
)


@pytest.fixture
def result():
    best = LoaderConfig(num_workers=1, prefetch_factor=2)
    fastest = LoaderConfig(num_workers=2, prefetch_factor=4)
    measurements = (
        make_measurement(best, (100.0, 101.0, 100.0), ci_low=99.0, ci_high=102.0),
        make_measurement(fastest, (105.0, 106.0, 105.0), ci_low=104.0, ci_high=107.0),
        make_measurement(LoaderConfig(), (80.0, 81.0, 80.0)),
        make_failed_measurement(LoaderConfig(4), "<script>worker failed</script>"),
    )
    return make_result(
        measurements,
        best,
        fastest,
        warnings=("Dataset warning <unsafe>",),
    )


def test_summary_is_human_readable(result) -> None:
    text = result.summary()

    assert "Recommended DataLoader configuration" in text
    assert "num_workers=1" in text
    assert "100.0 samples/s" in text
    assert "fastest measured configuration" in text
    assert "Test selection reason." in text
    assert "Recommendation validation" in text
    assert "status: confirmed" in text
    assert "Dataset warning <unsafe>" in text
    assert result.best_summary.median_throughput == 100.0


def test_json_export(result, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"

    result.to_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 5
    assert payload["apdo_version"] == "0.5.0"
    assert payload["protocol"]["seed"] == 42
    assert payload["protocol"]["search_rounds"] == 3
    assert payload["recommended"]["config"]["num_workers"] == 1
    assert "median_amortized_throughput" in payload["recommended"]
    assert payload["protocol"]["timing_profile"] == "standard"
    assert payload["protocol"]["selection_metric"] == "sustained"
    assert payload["protocol"]["comparability_policy"] == "warn"
    assert payload["protocol"]["search_strategy"] == "grid"
    assert payload["protocol"]["cache_policy"] == "off"
    assert payload["budget"] == {}
    assert payload["search_trace"] is None
    assert payload["profitability"] is None
    assert payload["cache"] == {}
    assert payload["timing"]["timing_profile"] == "standard"
    assert payload["comparability"] is None
    assert payload["recommended"]["validation_status"] == "confirmed"
    assert payload["fastest"]["config"]["num_workers"] == 2
    assert payload["settings"]["batch_size"] == 8
    assert payload["validation"]["status"] == "confirmed"
    assert payload["validation"]["candidate_config"]["num_workers"] == 1
    assert len(payload["validation"]["observations"]) == 10
    assert payload["validation"]["valid_pairs"] == 10
    assert payload["validation"]["minimum_required_pairs"] == 8
    assert payload["protocol"]["minimum_confirmation_pairs"] == 8
    assert payload["recommended"]["selection_metric"] == "sustained"
    assert payload["recommended"]["selection_value"] == 100.0
    assert payload["failures"][0]["error_message"] == "<script>worker failed</script>"
    assert payload["warnings"] == ["Dataset warning <unsafe>"]
    assert len(payload["configurations"]) == 4
    recommended = [item for item in payload["configurations"] if item["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["rank"] == 2
    assert "http://" not in target.read_text(encoding="utf-8")


def test_report_exposes_v05_sections(result, tmp_path: Path) -> None:
    trace = SearchTrace(
        strategy="adaptive",
        stages=(
            SearchStage(
                name="screening",
                planned_rounds=2,
                candidates=(
                    SearchCandidate(
                        config=result.search_candidate_config,
                        status="finalist",
                        phase="worker_scan",
                        measured_rounds=2,
                    ),
                ),
            ),
        ),
        selected_config=result.search_candidate_config,
        fastest_config=result.fastest_config,
    )
    result = replace(
        result,
        budget={
            "budget_s": 30.0,
            "status": "available",
            "remaining_s": 12.0,
            "search_elapsed_s": 10.0,
            "validation_elapsed_s": 8.0,
            "skipped_trials": 1,
            "measured_trials": 5,
        },
        search_trace=trace,
        profitability=result.estimate_profitability(planned_epochs=10),
        cache={"policy": "hint", "hit": True, "reason": "cache entry matched"},
    )

    payload = result.to_dict()
    text = result.summary()
    html = result.to_html(tmp_path / "v05.html")
    rendered = (tmp_path / "v05.html").read_text(encoding="utf-8")

    assert payload["budget"]["status"] == "available"
    assert payload["search_trace"]["strategy"] == "adaptive"
    assert payload["profitability"]["planned_epochs"] == 10
    assert payload["cache"]["hit"] is True
    assert "Search strategy: adaptive" in text
    assert "Budget" in text
    assert "Profitability" in text
    assert "Cache" in text
    assert "Search strategy" in rendered
    assert "Budget" in rendered
    assert "Profitability" in rendered
    assert html is None


def test_html_export_is_standalone_and_escaped(result, tmp_path: Path) -> None:
    target = tmp_path / "reports" / "report.html"

    result.to_html(target)

    html = target.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "PyTorch DataLoader benchmark report" in html
    assert "Recommended configuration" in html
    assert "Recommendation Validation" in html
    assert "confirmed" in html
    assert "num_workers=1" in html
    assert "Limitations" in html
    assert "&lt;script&gt;worker failed&lt;/script&gt;" in html
    assert "<script>worker failed" not in html
    assert "Dataset warning &lt;unsafe&gt;" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()


def test_result_without_measurement_fails_cleanly() -> None:
    missing = LoaderConfig(num_workers=8)
    other = make_measurement(LoaderConfig(num_workers=1), (10.0, 11.0, 10.0))
    result = make_result((other,), missing, missing)

    with pytest.raises(ValueError):
        result.summary()
    with pytest.raises(ValueError):
        result.to_dict()


def test_training_report_exposes_end_to_end_metrics(tmp_path: Path) -> None:
    baseline = LoaderConfig()
    best = LoaderConfig(num_workers=1, prefetch_factor=2)
    measurements = tuple(
        replace(make_measurement(config, values), training=True)
        for config, values in (
            (baseline, (100.0, 101.0, 100.0)),
            (best, (120.0, 121.0, 120.0)),
        )
    )
    settings = replace(
        make_settings(initial_config=baseline),
        benchmark_mode="training",
        epochs=3,
        training_rounds=2,
        training_validation_rounds=2,
    )
    result = make_result(
        measurements,
        best,
        best,
        settings=settings,
    )

    payload = result.to_dict()
    target = tmp_path / "training.html"
    result.to_html(target)
    html = target.read_text(encoding="utf-8")
    text = result.summary()

    assert payload["schema_version"] == 5
    assert payload["search"]["throughput_kind"] == "end_to_end_training"
    assert payload["training"]["epochs"] == 3
    assert payload["training"]["selected"]["summary"]["median_throughput"] == 120.0
    assert "training samples/s" in text
    assert "epochs: 3" in text
    assert "End-to-End Training" in html
    assert "Reproducibility" in html
