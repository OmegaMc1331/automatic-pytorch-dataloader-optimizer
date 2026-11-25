from __future__ import annotations

import html
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apdo._version import __version__
from apdo.config import LoaderConfig
from apdo.metrics import bootstrap_median_ci, median
from apdo.optimizer import OptimizationResult
from apdo.validation import MIN_CONFIRMATION_PAIRS

STABILITY_SCHEMA_VERSION = 1
MIN_STABILITY_RUNS = 5


@dataclass(frozen=True, slots=True)
class StabilityRun:
    run_index: int
    seed: int
    recommended_config: LoaderConfig | None
    search_candidate: LoaderConfig | None
    validation_status: str | None
    improvement_percent: float | None
    ci_low_percent: float | None
    ci_high_percent: float | None
    throughput_median: float | None
    amortized_throughput: float | None
    warning_count: int
    comparability_status: str | None
    sample_fingerprint_verified: bool
    error_type: str | None = None
    error_message: str | None = None
    summary_text: str | None = None
    initial_config: LoaderConfig | None = None
    selection_metric: str | None = None
    validation_enabled: bool = False
    valid_pairs: int | None = None
    invalid_pairs: int | None = None
    planned_pairs: int | None = None
    statistical_p_value: float | None = None
    search_improvement_percent: float | None = None
    duration_s: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None and self.recommended_config is not None

    @property
    def confirmed(self) -> bool:
        return self.validation_status == "confirmed"

    @property
    def fallback(self) -> bool:
        return (
            self.succeeded
            and self.validation_status not in {"confirmed", "not_needed", None}
            and self.recommended_config == self.initial_config
            and self.search_candidate != self.initial_config
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "seed": self.seed,
            "succeeded": self.succeeded,
            "confirmed": self.confirmed,
            "fallback": self.fallback,
            "recommended_config": (
                self.recommended_config.to_dict() if self.recommended_config is not None else None
            ),
            "search_candidate": (
                self.search_candidate.to_dict() if self.search_candidate is not None else None
            ),
            "initial_config": (
                self.initial_config.to_dict() if self.initial_config is not None else None
            ),
            "validation_status": self.validation_status,
            "validation_enabled": self.validation_enabled,
            "valid_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "planned_pairs": self.planned_pairs,
            "statistical_p_value": self.statistical_p_value,
            "selection_metric": self.selection_metric,
            "improvement_percent": self.improvement_percent,
            "search_improvement_percent": self.search_improvement_percent,
            "ci_low_percent": self.ci_low_percent,
            "ci_high_percent": self.ci_high_percent,
            "throughput_median": self.throughput_median,
            "amortized_throughput": self.amortized_throughput,
            "warning_count": self.warning_count,
            "comparability_status": self.comparability_status,
            "sample_fingerprint_verified": self.sample_fingerprint_verified,
            "duration_s": self.duration_s,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "summary": self.summary_text,
        }


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    runs_total: int
    runs_succeeded: int
    runs_failed: int
    runs_confirmed: int
    coverage_percent: float
    recommendation_counts: tuple[tuple[str, int], ...]
    most_common_recommendation: str | None
    agreement_percent: float | None
    median_improvement_percent: float | None
    ci_low_percent: float | None
    ci_high_percent: float | None
    median_throughput: float | None
    throughput_min: float | None
    throughput_max: float | None
    comparability_counts: tuple[tuple[str, int], ...]
    failure_counts: tuple[tuple[str, int], ...]
    candidate_counts: tuple[tuple[str, int], ...] = ()
    most_common_candidate: str | None = None
    candidate_agreement_percent: float | None = None
    fallback_count: int = 0
    runs_with_validation: int = 0
    runs_with_enough_pairs: int = 0
    median_confirmed_gain_percent: float | None = None
    confirmed_ci_low_percent: float | None = None
    confirmed_ci_high_percent: float | None = None
    qualification: str = "partial"

    def to_dict(self) -> dict[str, Any]:
        return {
            "runs_total": self.runs_total,
            "runs_succeeded": self.runs_succeeded,
            "runs_failed": self.runs_failed,
            "runs_confirmed": self.runs_confirmed,
            "coverage_percent": self.coverage_percent,
            "recommendation_counts": dict(self.recommendation_counts),
            "most_common_recommendation": self.most_common_recommendation,
            "agreement_percent": self.agreement_percent,
            "candidate_counts": dict(self.candidate_counts),
            "most_common_candidate": self.most_common_candidate,
            "candidate_agreement_percent": self.candidate_agreement_percent,
            "fallback_count": self.fallback_count,
            "runs_with_validation": self.runs_with_validation,
            "runs_with_enough_pairs": self.runs_with_enough_pairs,
            "median_improvement_percent": self.median_improvement_percent,
            "ci_low_percent": self.ci_low_percent,
            "ci_high_percent": self.ci_high_percent,
            "median_confirmed_gain_percent": self.median_confirmed_gain_percent,
            "confirmed_ci_low_percent": self.confirmed_ci_low_percent,
            "confirmed_ci_high_percent": self.confirmed_ci_high_percent,
            "median_throughput": self.median_throughput,
            "throughput_min": self.throughput_min,
            "throughput_max": self.throughput_max,
            "comparability_counts": dict(self.comparability_counts),
            "failure_counts": dict(self.failure_counts),
            "qualification": self.qualification,
        }


@dataclass(frozen=True, slots=True)
class StabilityReport:
    runs: tuple[StabilityRun, ...]
    summary: StabilitySummary
    base_seed: int
    started_at: datetime
    duration_s: float
    version: str = __version__
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def succeeded_runs(self) -> tuple[StabilityRun, ...]:
        return tuple(run for run in self.runs if run.succeeded)

    @property
    def failed_runs(self) -> tuple[StabilityRun, ...]:
        return tuple(run for run in self.runs if not run.succeeded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STABILITY_SCHEMA_VERSION,
            "report_kind": "stability",
            "apdo_version": self.version,
            "started_at_utc": self.started_at.isoformat(),
            "duration_s": self.duration_s,
            "base_seed": self.base_seed,
            "protocol": {
                "runs_total": len(self.runs),
                "runs_succeeded": self.summary.runs_succeeded,
                "runs_failed": self.summary.runs_failed,
                "minimum_runs_for_qualification": MIN_STABILITY_RUNS,
                "minimum_confirmation_pairs": MIN_CONFIRMATION_PAIRS,
                "qualification": self.summary.qualification,
            },
            "summary": self.summary.to_dict(),
            "runs": [run.to_dict() for run in self.runs],
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def to_html(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._html(), encoding="utf-8")

    def summary_text(self) -> str:
        summary = self.summary
        lines = [
            "Recommendation stability",
            f"  runs: {summary.runs_succeeded}/{summary.runs_total} succeeded, "
            f"{summary.runs_failed} failed, coverage {summary.coverage_percent:.1f}%",
            f"  confirmed gains: {summary.runs_confirmed}/{summary.runs_succeeded}",
            f"  fallbacks to the initial configuration: {summary.fallback_count}",
            f"  qualification: {summary.qualification}",
        ]
        if summary.most_common_recommendation is not None and summary.agreement_percent is not None:
            lines.append(
                f"  recommendation agreement: {summary.agreement_percent:.1f}% for "
                f"{summary.most_common_recommendation}"
            )
        if summary.most_common_candidate is not None and (
            summary.candidate_agreement_percent is not None
        ):
            lines.append(
                f"  candidate agreement: {summary.candidate_agreement_percent:.1f}% for "
                f"{summary.most_common_candidate}"
            )
        if summary.median_improvement_percent is not None:
            interval = ""
            if summary.ci_low_percent is not None and summary.ci_high_percent is not None:
                interval = f" [{summary.ci_low_percent:+.1f}%, {summary.ci_high_percent:+.1f}%]"
            lines.append(
                f"  median observed gain (all validation outcomes): "
                f"{summary.median_improvement_percent:+.1f}%{interval}"
            )
        if summary.median_confirmed_gain_percent is not None:
            lines.append(f"  median confirmed gain: {summary.median_confirmed_gain_percent:+.1f}%")
        if summary.median_throughput is not None:
            lines.append(
                f"  median throughput: {summary.median_throughput:,.1f} "
                f"(range {summary.throughput_min:,.1f} - {summary.throughput_max:,.1f})"
            )
        for label, count in summary.failure_counts:
            lines.append(f"  failures: {label} ({count})")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)

    def _html(self) -> str:
        escape = html.escape
        summary = self.summary
        recommendation_rows = "".join(
            f"<tr><td>{escape(label)}</td><td>{count}</td></tr>"
            for label, count in summary.recommendation_counts
        )
        candidate_rows = "".join(
            f"<tr><td>{escape(label)}</td><td>{count}</td></tr>"
            for label, count in summary.candidate_counts
        )
        comparability_rows = "".join(
            f"<tr><td>{escape(label)}</td><td>{count}</td></tr>"
            for label, count in summary.comparability_counts
        )
        failure_rows = "".join(
            f"<tr><td>{escape(label)}</td><td>{count}</td></tr>"
            for label, count in summary.failure_counts
        )
        run_rows: list[str] = []
        for run in self.runs:
            recommended = (
                escape(run.recommended_config.describe())
                if run.recommended_config is not None
                else "n/a"
            )
            candidate = (
                escape(run.search_candidate.describe())
                if run.search_candidate is not None
                else "n/a"
            )
            gain = _fmt_percent(run.improvement_percent)
            throughput = _fmt_number(run.throughput_median)
            error = escape(run.error_message or "") if run.error_message else ""
            run_rows.append(
                f"<tr><td>{run.run_index}</td><td>{run.seed}</td>"
                f"<td>{'yes' if run.succeeded else 'no'}</td>"
                f"<td>{escape(run.validation_status or 'n/a')}</td>"
                f"<td>{'yes' if run.fallback else 'no'}</td>"
                f"<td>{run.valid_pairs if run.valid_pairs is not None else 'n/a'}"
                f"/{run.planned_pairs if run.planned_pairs is not None else 'n/a'}</td>"
                f"<td>{recommended}</td><td>{candidate}</td>"
                f"<td>{gain}</td><td>{throughput}</td>"
                f"<td>{escape(run.comparability_status or 'n/a')}</td>"
                f"<td>{error}</td></tr>"
            )
        warnings_html = ""
        if self.warnings:
            items = "".join(f"<li>{escape(item)}</li>" for item in self.warnings)
            warnings_html = f"<section><h2>Warnings</h2><ul>{items}</ul></section>"
        agreement = (
            f"{summary.agreement_percent:.1f}%" if summary.agreement_percent is not None else "n/a"
        )
        gain = _fmt_percent(summary.median_improvement_percent)
        confirmed_gain = _fmt_percent(summary.median_confirmed_gain_percent)
        interval = "n/a"
        if summary.ci_low_percent is not None and summary.ci_high_percent is not None:
            interval = f"[{summary.ci_low_percent:+.1f}%, {summary.ci_high_percent:+.1f}%]"
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>apdo recommendation stability report</title>
<style>
:root {{
  color-scheme: light;
  --ink: #17202a;
  --muted: #5b6573;
  --line: #d7dee7;
  --bg: #f4f6f9;
  --card: #fff;
  --accent: #0f766e;
}}
* {{ box-sizing:border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}}
main {{ max-width:1180px; margin:0 auto; padding:36px 24px 56px; }}
section {{ margin:28px 0; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:22px; }}
h1 {{ margin:4px 0 8px; }}
h2 {{ margin-top:0; }}
.eyebrow {{
  margin: 0;
  text-transform: uppercase;
  font-size: 11px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: 0.09em;
}}
.muted {{ color:var(--muted); }}
.metrics {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
  gap: 12px;
  margin: 18px 0;
}}
.metric {{
  background: #f8fafc;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
}}
.metric span {{ display:block; color:var(--muted); font-size:12px; }}
.metric strong {{ font-variant-numeric:tabular-nums; }}
.table-wrap {{
  overflow-x: auto;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
}}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ background:#eef2f6; font-size:12px; text-transform:uppercase; color:var(--muted); }}
code {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }}
footer {{
  margin-top: 32px;
  border-top: 1px solid var(--line);
  padding-top: 16px;
  color: var(--muted);
  font-size: 13px;
}}
</style>
</head>
<body>
<main>
<header class="card">
<p class="eyebrow">apdo {escape(self.version)}</p>
<h1>Recommendation stability report</h1>
<p class="muted">{len(self.runs)} independent optimizations from base seed {self.base_seed},
started {escape(self.started_at.isoformat())}, completed in {self.duration_s:.2f} s.</p>
<div class="metrics">
<div class="metric"><span>Coverage</span><strong>{summary.coverage_percent:.1f}%</strong></div>
<div class="metric"><span>Succeeded</span>
<strong>{summary.runs_succeeded}/{summary.runs_total}</strong></div>
<div class="metric"><span>Confirmed gains</span><strong>{summary.runs_confirmed}</strong></div>
<div class="metric"><span>Fallbacks</span><strong>{summary.fallback_count}</strong></div>
<div class="metric"><span>Agreement</span><strong>{escape(agreement)}</strong></div>
<div class="metric"><span>Observed median gain</span><strong>{escape(gain)}</strong></div>
<div class="metric"><span>Confirmed median gain</span>
<strong>{escape(confirmed_gain)}</strong></div>
<div class="metric"><span>Gain interval</span><strong>{escape(interval)}</strong></div>
<div class="metric"><span>Qualification</span><strong>{escape(summary.qualification)}</strong></div>
</div>
</header>
<section>
<h2>Independent runs</h2>
<div class="table-wrap"><table>
<thead><tr><th>Run</th><th>Seed</th><th>Succeeded</th><th>Validation</th>
<th>Fallback</th><th>Valid/planned pairs</th>
<th>Recommended</th><th>Search candidate</th><th>Gain</th>
<th>Median throughput</th><th>Comparability</th><th>Failure</th></tr></thead>
<tbody>{"".join(run_rows)}</tbody>
</table></div>
</section>
<section>
<h2>Recommendation frequency</h2>
<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Runs</th></tr></thead>
<tbody>{recommendation_rows}</tbody></table></div>
</section>
<section>
<h2>Search candidate frequency</h2>
<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Runs</th></tr></thead>
<tbody>{candidate_rows}</tbody></table></div>
</section>
<section>
<h2>Comparability status</h2>
<div class="table-wrap"><table><thead><tr><th>Status</th><th>Runs</th></tr></thead>
<tbody>{comparability_rows}</tbody></table></div>
</section>
<section>
<h2>Failures</h2>
<div class="table-wrap"><table><thead><tr><th>Failure</th><th>Runs</th></tr></thead>
<tbody>{failure_rows}</tbody></table></div>
</section>
{warnings_html}
<footer>Generated locally by apdo {escape(self.version)}. Raw batch data is not stored.</footer>
</main>
</body>
</html>"""


def evaluate_stability(
    experiment: Callable[[int], OptimizationResult],
    *,
    runs: int = 5,
    base_seed: int = 42,
    progress: Callable[[str], None] | None = None,
) -> StabilityReport:
    if not callable(experiment):
        raise TypeError("experiment must be callable")
    if runs < 1:
        raise ValueError("runs must be at least 1")
    started_at = datetime.now(timezone.utc)
    started = _clock()
    records: list[StabilityRun] = []
    warnings: list[str] = []
    for index in range(runs):
        seed = base_seed + index
        _emit(progress, f"Stability run {index + 1}/{runs} with seed {seed}")
        try:
            result = experiment(seed)
        except Exception as exc:  # noqa: BLE001 - stability must retain partial failures
            _emit(progress, f"  run {index + 1} failed: {type(exc).__name__}: {exc}")
            records.append(
                StabilityRun(
                    run_index=index,
                    seed=seed,
                    recommended_config=None,
                    search_candidate=None,
                    validation_status=None,
                    improvement_percent=None,
                    ci_low_percent=None,
                    ci_high_percent=None,
                    throughput_median=None,
                    amortized_throughput=None,
                    warning_count=0,
                    comparability_status=None,
                    sample_fingerprint_verified=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )
            continue
        comparison = result.validation.comparability
        summary = result.best_summary
        recommendation = result.best_config
        validation = result.validation
        records.append(
            StabilityRun(
                run_index=index,
                seed=seed,
                recommended_config=recommendation,
                search_candidate=result.search_candidate_config,
                validation_status=validation.status,
                improvement_percent=validation.improvement_percent,
                ci_low_percent=validation.ci_low_percent,
                ci_high_percent=validation.ci_high_percent,
                throughput_median=summary.median_throughput,
                amortized_throughput=summary.median_amortized_throughput,
                warning_count=len(result.warnings),
                comparability_status=comparison.status if comparison is not None else None,
                sample_fingerprint_verified=bool(
                    comparison is not None and comparison.sample_identity_verified
                ),
                summary_text=result.summary(),
                initial_config=result.settings.initial_config,
                selection_metric=result.settings.selection_metric,
                validation_enabled=validation.enabled,
                valid_pairs=validation.valid_pairs,
                invalid_pairs=validation.invalid_pairs,
                planned_pairs=validation.planned_pairs,
                statistical_p_value=validation.statistical_p_value,
                search_improvement_percent=result.search_improvement_percent,
                duration_s=result.duration_s,
            )
        )
        _emit(
            progress,
            f"  recommended {recommendation.describe()} "
            f"({result.validation.status}, "
            f"{_fmt_percent(result.validation.improvement_percent)})",
        )
        warnings.extend(f"run {index}: {warning}" for warning in result.warnings)
    summary = summarize_stability(records)
    if summary.runs_succeeded < MIN_STABILITY_RUNS:
        warnings.append(
            f"Only {summary.runs_succeeded} independent optimizations succeeded; "
            f"at least {MIN_STABILITY_RUNS} are required for a complete stability "
            "qualification."
        )
    if summary.runs_succeeded and summary.runs_with_validation < summary.runs_succeeded:
        warnings.append(
            f"{summary.runs_succeeded - summary.runs_with_validation} successful runs "
            "did not enable independent validation."
        )
    if summary.runs_succeeded and summary.runs_with_enough_pairs < summary.runs_succeeded:
        warnings.append(
            f"{summary.runs_succeeded - summary.runs_with_enough_pairs} successful runs "
            f"had fewer than {MIN_CONFIRMATION_PAIRS} valid paired validation observations."
        )
    if summary.fallback_count:
        warnings.append(
            f"{summary.fallback_count} successful runs fell back to the initial "
            "configuration because validation did not confirm the search candidate. "
            "Recommendation agreement that counts fallbacks does not demonstrate a "
            "stable optimization benefit."
        )
    if summary.runs_succeeded and summary.runs_confirmed == 0:
        warnings.append(
            "No independent optimization produced a confirmed gain; agreement reflects "
            "fallback or initial-configuration choices rather than a validated speedup."
        )
    if (
        summary.agreement_percent is not None
        and summary.agreement_percent >= 80.0
        and summary.runs_confirmed == 0
    ):
        warnings.append(
            f"High recommendation agreement ({summary.agreement_percent:.1f}%) is not "
            "evidence of improvement because no run confirmed a gain."
        )
    return StabilityReport(
        runs=tuple(records),
        summary=summary,
        base_seed=base_seed,
        started_at=started_at,
        duration_s=_clock() - started,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def summarize_stability(records: Sequence[StabilityRun]) -> StabilitySummary:
    succeeded = [record for record in records if record.succeeded]
    failed = [record for record in records if not record.succeeded]
    recommendation_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for record in succeeded:
        if record.recommended_config is not None:
            recommendation_counts[record.recommended_config.describe()] += 1
        if record.search_candidate is not None:
            candidate_counts[record.search_candidate.describe()] += 1
    counts = tuple(recommendation_counts.most_common())
    candidate_tuple = tuple(candidate_counts.most_common())
    agreement = None
    most_common = None
    if succeeded and recommendation_counts:
        most_common, most_common_count = counts[0]
        agreement = most_common_count / len(succeeded) * 100.0
    candidate_agreement = None
    most_common_candidate = None
    if succeeded and candidate_counts:
        most_common_candidate, candidate_count = candidate_tuple[0]
        candidate_agreement = candidate_count / len(succeeded) * 100.0
    gains = [
        record.improvement_percent for record in succeeded if record.improvement_percent is not None
    ]
    median_gain = median(gains) if gains else None
    ci_low = ci_high = None
    if len(gains) >= 2:
        ci_low, ci_high = bootstrap_median_ci(gains, confidence=0.95, resamples=4000, seed=0)
    elif len(gains) == 1:
        ci_low = ci_high = gains[0]
    confirmed_gains = [
        record.improvement_percent
        for record in succeeded
        if record.confirmed and record.improvement_percent is not None
    ]
    median_confirmed_gain = median(confirmed_gains) if confirmed_gains else None
    confirmed_ci_low = confirmed_ci_high = None
    if len(confirmed_gains) >= 2:
        confirmed_ci_low, confirmed_ci_high = bootstrap_median_ci(
            confirmed_gains,
            confidence=0.95,
            resamples=4000,
            seed=0,
        )
    elif len(confirmed_gains) == 1:
        confirmed_ci_low = confirmed_ci_high = confirmed_gains[0]
    throughputs = [
        record.throughput_median for record in succeeded if record.throughput_median is not None
    ]
    comparability_counts: Counter[str] = Counter()
    for record in succeeded:
        comparability_counts[record.comparability_status or "unavailable"] += 1
    failure_counts: Counter[str] = Counter()
    for record in failed:
        failure_counts[record.error_type or "unknown"] += 1
    total = len(records)
    runs_with_validation = sum(1 for record in succeeded if record.validation_enabled)
    runs_with_enough_pairs = sum(
        1
        for record in succeeded
        if record.valid_pairs is not None and record.valid_pairs >= MIN_CONFIRMATION_PAIRS
    )
    complete = (
        len(succeeded) >= MIN_STABILITY_RUNS
        and runs_with_validation == len(succeeded)
        and runs_with_enough_pairs == len(succeeded)
    )
    return StabilitySummary(
        runs_total=total,
        runs_succeeded=len(succeeded),
        runs_failed=len(failed),
        runs_confirmed=sum(1 for record in succeeded if record.confirmed),
        coverage_percent=(len(succeeded) / total * 100.0) if total else 0.0,
        recommendation_counts=counts,
        most_common_recommendation=most_common,
        agreement_percent=agreement,
        median_improvement_percent=median_gain,
        ci_low_percent=ci_low,
        ci_high_percent=ci_high,
        median_throughput=median(throughputs) if throughputs else None,
        throughput_min=min(throughputs) if throughputs else None,
        throughput_max=max(throughputs) if throughputs else None,
        comparability_counts=tuple(comparability_counts.most_common()),
        failure_counts=tuple(failure_counts.most_common()),
        candidate_counts=candidate_tuple,
        most_common_candidate=most_common_candidate,
        candidate_agreement_percent=candidate_agreement,
        fallback_count=sum(1 for record in succeeded if record.fallback),
        runs_with_validation=runs_with_validation,
        runs_with_enough_pairs=runs_with_enough_pairs,
        median_confirmed_gain_percent=median_confirmed_gain,
        confirmed_ci_low_percent=confirmed_ci_low,
        confirmed_ci_high_percent=confirmed_ci_high,
        qualification="complete" if complete else "partial",
    )


def _clock() -> float:
    import time

    return time.perf_counter()


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _fmt_percent(value: float | None) -> str:
    return f"{value:+.1f}%" if value is not None else "n/a"


def _fmt_number(value: float | None) -> str:
    return f"{value:,.1f}" if value is not None else "n/a"
