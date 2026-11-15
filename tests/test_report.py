from __future__ import annotations

import json
from pathlib import Path

import pytest

from apdo import LoaderConfig
from tests.helpers import (
    make_failed_measurement,
    make_measurement,
    make_result,
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
    assert "Dataset warning <unsafe>" in text
    assert result.best_summary.median_throughput == 100.0


def test_json_export(result, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "report.json"

    result.to_json(target)

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["apdo_version"] == "0.1.0"
    assert payload["recommended"]["config"]["num_workers"] == 1
    assert payload["fastest"]["config"]["num_workers"] == 2
    assert payload["settings"]["batch_size"] == 8
    assert payload["failures"][0]["error_message"] == "<script>worker failed</script>"
    assert payload["warnings"] == ["Dataset warning <unsafe>"]
    assert len(payload["configurations"]) == 4
    recommended = [item for item in payload["configurations"] if item["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["rank"] == 2
    assert "http://" not in target.read_text(encoding="utf-8")


def test_html_export_is_standalone_and_escaped(result, tmp_path: Path) -> None:
    target = tmp_path / "reports" / "report.html"

    result.to_html(target)

    html = target.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "PyTorch DataLoader benchmark report" in html
    assert "Recommended configuration" in html
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
