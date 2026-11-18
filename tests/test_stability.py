from __future__ import annotations

import json
from pathlib import Path

from apdo import LoaderConfig
from apdo.stability import evaluate_stability, summarize_stability
from tests.helpers import make_measurement, make_result


def _experiment(seed: int):
    config = LoaderConfig(num_workers=seed % 3)
    measurement = make_measurement(config, (100.0 + seed, 101.0 + seed, 100.0 + seed))
    return make_result((measurement,), config, config)


def test_evaluate_stability_summarizes_independent_runs(tmp_path: Path) -> None:
    report = evaluate_stability(_experiment, runs=4, base_seed=20)

    assert len(report.runs) == 4
    assert [run.seed for run in report.runs] == [20, 21, 22, 23]
    assert report.summary.runs_total == 4
    assert report.summary.runs_succeeded == 4
    assert report.summary.runs_confirmed == 3
    assert report.summary.coverage_percent == 100.0
    assert report.summary.most_common_recommendation is not None
    assert report.summary.agreement_percent is not None
    assert report.summary.median_improvement_percent is not None
    assert report.summary.median_throughput is not None

    json_path = tmp_path / "stability.json"
    html_path = tmp_path / "stability.html"
    report.to_json(json_path)
    report.to_html(html_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["report_kind"] == "stability"
    assert payload["summary"]["runs_succeeded"] == 4
    assert len(payload["runs"]) == 4
    assert all(item["succeeded"] for item in payload["runs"])
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert "Recommendation stability report" in html_path.read_text(encoding="utf-8")


def test_evaluate_stability_retains_partial_failures() -> None:
    def experiment(seed: int):
        if seed == 31:
            raise RuntimeError("synthetic stability failure")
        return _experiment(seed)

    report = evaluate_stability(experiment, runs=3, base_seed=30)

    assert report.summary.runs_total == 3
    assert report.summary.runs_succeeded == 2
    assert report.summary.runs_failed == 1
    assert report.summary.coverage_percent == 66.66666666666666
    assert report.summary.failure_counts == (("RuntimeError", 1),)
    failed = report.failed_runs[0]
    assert failed.seed == 31
    assert failed.error_type == "RuntimeError"
    assert "synthetic stability failure" in (failed.error_message or "")
    payload = report.to_dict()
    assert payload["runs"][1]["succeeded"] is False


def test_summarize_stability_without_success() -> None:
    def experiment(seed: int):
        raise RuntimeError("always fails")

    report = evaluate_stability(experiment, runs=2, base_seed=0)
    summary = summarize_stability(report.runs)

    assert summary.runs_succeeded == 0
    assert summary.runs_failed == 2
    assert summary.agreement_percent is None
    assert summary.median_improvement_percent is None
    assert summary.failure_counts == (("RuntimeError", 2),)
