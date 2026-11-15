from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from apdo._version import __version__
from apdo.benchmark import ConfigMeasurement
from apdo.config import BenchmarkSettings, LoaderConfig
from apdo.metrics import MetricsSummary
from apdo.resources import EnvironmentInfo

METHOD = {
    "search": (
        "Two-stage grid search. Stage 1 scans num_workers with default prefetch "
        "settings and both pin_memory values on CUDA. Stage 2 varies prefetch_factor "
        "and persistent_workers for the two best multi-worker counts."
    ),
    "selection_metric": "Median steady-state throughput across measured passes.",
    "uncertainty": (
        "95% percentile bootstrap interval of the median throughput, computed from "
        "2000 resamples of the measured passes."
    ),
    "selection_rule": (
        "The simplest configuration whose median throughput is within 5% of the "
        "fastest configuration and whose bootstrap interval overlaps the fastest "
        "interval is selected. Interval overlap is an equivalence heuristic, not a "
        "formal hypothesis test. Simplicity is ordered by fewer workers, no pinning, "
        "non-persistent workers, and lower prefetch_factor."
    ),
    "throughput_scope": (
        "Data loading plus host-to-device tensor copies when the resolved device is "
        "CUDA. No model forward or backward pass is executed."
    ),
}

LIMITATIONS = (
    "Measurements describe data loading and, on CUDA, host-to-device copies. They "
    "are not training throughput.",
    "max_ram_gb is a sampled system-memory guard, not a hard limit. Allocations can "
    "occur between samples, and unrelated processes affect the observed value.",
    "Configurations are benchmarked sequentially and not in randomized order, so "
    "slow drift in machine load can bias later measurements.",
    "Confidence intervals reflect variation across measured passes only. Interval "
    "overlap is an equivalence heuristic, not a formal hypothesis test, and intervals "
    "do not model dataset randomness or long-term system variability.",
    "Only map-style datasets with __len__ and __getitem__ are supported in v0.1.0.",
    "The same dataset object is reused across configurations. Dataset caching or "
    "state changes can make later configurations appear faster.",
    "Process RSS covers the parent process and is absolute, so it can include memory "
    "retained from earlier configurations. Worker memory is reflected only indirectly "
    "by system-wide memory availability. These memory observations are implemented "
    "for Windows and Linux.",
)


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    best_config: LoaderConfig
    fastest_config: LoaderConfig
    measurements: tuple[ConfigMeasurement, ...]
    environment: EnvironmentInfo
    settings: BenchmarkSettings
    warnings: tuple[str, ...]
    selection_reason: str
    baseline_throughput: float | None
    improvement_percent: float | None
    started_at: datetime
    duration_s: float
    version: str = __version__

    @property
    def best_summary(self) -> MetricsSummary:
        for measurement in self.measurements:
            if (
                measurement.succeeded
                and measurement.config == self.best_config
                and measurement.summary is not None
            ):
                return measurement.summary
        raise ValueError("the recommended configuration has no completed measurement")

    def summary(self) -> str:
        measurement = self._measurement_for(self.best_config)
        summary = self.best_summary
        passes = measurement.passes
        batches = passes[0].batches if passes else 0
        failed = sum(1 for item in self.measurements if not item.succeeded)
        device_note = (
            "cuda (includes host-to-device copies)"
            if self.settings.device.startswith("cuda")
            else self.settings.device
        )
        memory = _memory_text(summary)
        cold_start = _fmt_ms(measurement.cold_start_s)

        lines = [
            "Recommended DataLoader configuration",
            f"  {self.best_config.describe()}",
            "",
            "Measured performance",
            f"  median throughput: {summary.median_throughput:,.1f} samples/s",
            f"  throughput range: {summary.min_throughput:,.1f} - "
            f"{summary.max_throughput:,.1f} samples/s",
            f"  median batch latency: {_fmt_ms(summary.median_latency_s)} "
            f"(p95 {_fmt_ms(summary.p95_latency_s)})",
            f"  startup to first batch: {_fmt_ms(summary.median_startup_s)} "
            f"(cold start {cold_start})",
            f"  observed memory: {memory}",
        ]
        if self.improvement_percent is not None:
            lines.append(f"  improvement vs default: {self.improvement_percent:+.1f}%")
        else:
            lines.append("  improvement vs default: n/a")
        if self.fastest_config != self.best_config:
            lines.append(f"  fastest measured configuration: {self.fastest_config.describe()}")
        lines.extend(
            [
                "",
                "Benchmark",
                f"  {self.settings.repetitions} measured passes, "
                f"{batches} batches per pass, {self.settings.dataset_size} dataset "
                f"samples, batch_size={self.settings.batch_size}",
                f"  device: {device_note}",
                f"  {len(self.measurements)} configurations measured ({failed} failed)",
                "",
                "Selection",
                f"  {self.selection_reason}",
            ]
        )
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        ranked = self._ranked_successes()
        ranks = {item.config: index for index, item in enumerate(ranked, start=1)}
        fastest = self._measurement_for(self.fastest_config)
        fastest_summary = fastest.summary
        if fastest_summary is None:
            raise ValueError("the fastest configuration has no completed measurement")

        configurations = []
        for measurement in ranked:
            entry = measurement.to_dict()
            entry["rank"] = ranks[measurement.config]
            entry["recommended"] = measurement.config == self.best_config
            configurations.append(entry)
        for measurement in self.measurements:
            if measurement.succeeded:
                continue
            entry = measurement.to_dict()
            entry["rank"] = None
            entry["recommended"] = False
            configurations.append(entry)

        failures = [
            {
                "config": item.config.to_dict(),
                "phase": item.phase,
                "error_type": item.error_type,
                "error_message": item.error_message,
            }
            for item in self.measurements
            if not item.succeeded
        ]
        return {
            "schema_version": 1,
            "apdo_version": self.version,
            "started_at_utc": self.started_at.isoformat(),
            "duration_s": self.duration_s,
            "environment": self.environment.to_dict(),
            "settings": self.settings.to_dict(),
            "method": dict(METHOD),
            "limitations": list(LIMITATIONS),
            "recommended": {
                "config": self.best_config.to_dict(),
                "median_throughput": self.best_summary.median_throughput,
                "improvement_percent": self.improvement_percent,
                "selection_reason": self.selection_reason,
            },
            "fastest": {
                "config": self.fastest_config.to_dict(),
                "median_throughput": fastest_summary.median_throughput,
            },
            "configurations": configurations,
            "failures": failures,
            "warnings": list(self.warnings),
        }

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        target.write_text(payload + "\n", encoding="utf-8")

    def to_html(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._html(), encoding="utf-8")

    def _measurement_for(self, config: LoaderConfig) -> ConfigMeasurement:
        for measurement in self.measurements:
            if (
                measurement.succeeded
                and measurement.config == config
                and measurement.summary is not None
            ):
                return measurement
        raise ValueError(f"no completed measurement for {config.describe()}")

    def _ranked_successes(self) -> list[ConfigMeasurement]:
        ranked = [item for item in self.measurements if item.succeeded and item.summary is not None]
        ranked.sort(
            key=lambda item: item.summary.median_throughput if item.summary else 0.0,
            reverse=True,
        )
        return ranked

    def _html(self) -> str:
        escape = html.escape
        best = self.best_summary
        best_measurement = self._measurement_for(self.best_config)
        failed = [item for item in self.measurements if not item.succeeded]
        device_is_cuda = self.settings.device.startswith("cuda")
        fastest_note = ""
        if self.fastest_config != self.best_config:
            fastest_note = (
                '<p class="muted">Fastest measured configuration: '
                f"<code>{escape(self.fastest_config.describe())}</code></p>"
            )

        metric_cards = [
            ("Median throughput", f"{best.median_throughput:,.1f} samples/s"),
            ("Median batch latency", _fmt_ms(best.median_latency_s)),
            ("p95 batch latency", _fmt_ms(best.p95_latency_s)),
            ("Median startup", _fmt_ms(best.median_startup_s)),
            ("Cold start", _fmt_ms(best_measurement.cold_start_s)),
            ("Observed memory", _memory_text(best)),
        ]
        cards_html = "\n".join(
            f'<div class="metric"><span>{escape(label)}</span>'
            f"<strong>{escape(value)}</strong></div>"
            for label, value in metric_cards
        )

        improvement = (
            f"{self.improvement_percent:+.1f}% vs default"
            if self.improvement_percent is not None
            else "n/a"
        )
        warnings_html = ""
        if self.warnings:
            items = "\n".join(f"<li>{escape(item)}</li>" for item in self.warnings)
            warnings_html = f"<section><h2>Warnings</h2><ul>{items}</ul></section>"

        rows = []
        for rank, measurement in enumerate(self._ranked_successes(), start=1):
            summary = measurement.summary
            if summary is None:
                continue
            row_class = ' class="recommended"' if measurement.config == self.best_config else ""
            gpu_cell = ""
            if device_is_cuda:
                gpu_cell = f"<td>{_fmt_mib(summary.peak_gpu_memory_bytes)}</td>"
            rows.append(
                f'<tr{row_class}><td class="rank">{rank}</td>'
                f"<td><code>{escape(measurement.config.describe())}</code></td>"
                f"<td>{summary.median_throughput:,.1f}</td>"
                f"<td>{summary.cv_throughput * 100:.1f}%</td>"
                f"<td>{_fmt_ms(summary.median_latency_s)}</td>"
                f"<td>{_fmt_ms(summary.p95_latency_s)}</td>"
                f"<td>{_fmt_ms(summary.median_startup_s)}</td>"
                f"<td>{_fmt_mib(summary.peak_process_rss_bytes)}</td>"
                f"{gpu_cell}</tr>"
            )
        gpu_header = "<th>GPU peak</th>" if device_is_cuda else ""

        failures_html = ""
        if failed:
            failure_rows = "\n".join(
                f"<tr><td><code>{escape(item.config.describe())}</code></td>"
                f"<td>{escape(item.phase)}</td>"
                f"<td><code>{escape(item.error_type or 'unknown')}</code><br>"
                f'<span class="error">{escape(item.error_message or "")}</span></td></tr>'
                for item in failed
            )
            failures_html = (
                "<section><h2>Failed configurations</h2>"
                '<div class="table-wrap"><table><thead><tr>'
                "<th>Configuration</th><th>Phase</th><th>Error</th>"
                f"</tr></thead><tbody>{failure_rows}</tbody></table></div></section>"
            )

        recent = [
            (
                "Python",
                f"{self.environment.python_version} ({self.environment.python_implementation})",
            ),
            ("PyTorch", self.environment.torch_version),
            ("Platform", self.environment.platform),
            ("Machine", self.environment.machine),
            ("Processor", self.environment.processor or "n/a"),
            ("CPU count", str(self.environment.cpu_count)),
            (
                "CPU affinity",
                str(self.environment.cpu_affinity_count)
                if self.environment.cpu_affinity_count is not None
                else "n/a",
            ),
            ("System memory", _fmt_mib(self.environment.total_memory_bytes)),
            (
                "CUDA",
                f"{self.environment.cuda_version or 'n/a'} "
                f"({'available' if self.environment.cuda_available else 'unavailable'})",
            ),
            (
                "CUDA devices",
                ", ".join(
                    f"{device.name} ({_fmt_mib(device.total_memory_bytes)})"
                    for device in self.environment.cuda_devices
                )
                or "none",
            ),
            ("Torch threads", str(self.environment.torch_num_threads)),
        ]
        environment_rows = "\n".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in recent
        )

        settings_rows = [
            ("Device", self.settings.device),
            ("Requested device", self.settings.requested_device),
            ("Dataset size", f"{self.settings.dataset_size:,} samples"),
            ("Batch size", str(self.settings.batch_size)),
            ("Batches per pass", str(self.best_measurement_batches())),
            ("Measured passes", str(self.settings.repetitions)),
            ("Warmup passes", str(self.settings.warmup_runs)),
            ("Worker search limit", str(self.settings.max_workers)),
            ("max_ram_gb", str(self.settings.max_ram_gb) if self.settings.max_ram_gb else "none"),
            ("Duration", f"{self.duration_s:.2f} s"),
        ]
        settings_html = "\n".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
            for label, value in settings_rows
        )

        method_html = "\n".join(
            f"<li><strong>{escape(key.replace('_', ' ').capitalize())}:</strong> "
            f"{escape(value)}</li>"
            for key, value in METHOD.items()
        )
        limitations_html = "\n".join(f"<li>{escape(item)}</li>" for item in LIMITATIONS)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="apdo {escape(self.version)}">
<title>DataLoader benchmark report</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
<p class="eyebrow">apdo {escape(self.version)}</p>
<h1>PyTorch DataLoader benchmark report</h1>
<p class="muted">Started {escape(self.started_at.isoformat())} &middot;
completed in {self.duration_s:.2f} s</p>
</header>

<section class="card accent">
<p class="eyebrow">Recommended configuration</p>
<h2><code>{escape(self.best_config.describe())}</code></h2>
<div class="metrics">{cards_html}</div>
<p><strong>{escape(improvement)}</strong>. {escape(self.selection_reason)}</p>
{fastest_note}
</section>

{warnings_html}

<section>
<h2>Ranking</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>Rank</th><th>Configuration</th><th>Median samples/s</th><th>CV</th>
<th>Median batch</th><th>p95 batch</th><th>Startup</th><th>Peak RSS</th>{gpu_header}
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
</section>

{failures_html}

<section>
<h2>Environment</h2>
<div class="table-wrap"><table class="kv"><tbody>{environment_rows}</tbody></table></div>
</section>

<section>
<h2>Benchmark settings</h2>
<div class="table-wrap"><table class="kv"><tbody>{settings_html}</tbody></table></div>
</section>

<section>
<h2>Method</h2>
<ul>{method_html}</ul>
</section>

<section>
<h2>Limitations</h2>
<ul>{limitations_html}</ul>
</section>

<footer>
Generated locally by apdo {escape(self.version)}. This report is self-contained and
requires no network access. Throughput covers data loading, not model training.
</footer>
</main>
</body>
</html>
"""

    def best_measurement_batches(self) -> int:
        measurement = self._measurement_for(self.best_config)
        return measurement.passes[0].batches if measurement.passes else 0


_CSS = """
:root {
  color-scheme: light;
  --ink: #17202a;
  --muted: #5b6573;
  --line: #d7dee7;
  --bg: #f4f6f9;
  --card: #ffffff;
  --accent: #0f766e;
  --accent-soft: #e7f4f1;
  --failed: #a61b1b;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 15px;
  line-height: 1.5;
}
main { max-width: 1180px; margin: 0 auto; padding: 40px 24px 56px; }
header { margin-bottom: 28px; }
h1 { font-size: 30px; line-height: 1.15; margin: 4px 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 20px; margin: 0 0 14px; letter-spacing: -0.01em; }
section { margin: 28px 0; }
p { margin: 10px 0; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 0.92em; }
.eyebrow {
  margin: 0;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  font-size: 11px;
  font-weight: 700;
  color: var(--accent);
}
.muted { color: var(--muted); }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 22px;
}
.card.accent { border-left: 4px solid var(--accent); background: var(--accent-soft); }
.card h2 { margin-top: 6px; }
.metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.metric {
  background: rgba(255,255,255,0.72);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
}
.metric span { display: block; font-size: 12px; color: var(--muted); }
.metric strong { font-size: 17px; font-variant-numeric: tabular-nums; }
.table-wrap {
  overflow-x: auto;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 10px;
}
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}
th {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--muted);
  background: #eef2f6;
}
tbody tr:last-child td, tbody tr:last-child th { border-bottom: none; }
tr.recommended td { background: #f0faf7; }
td.rank { font-weight: 700; color: var(--accent); }
table.kv th { width: 210px; text-transform: none; letter-spacing: 0; font-size: 13px; }
.error { color: var(--failed); white-space: pre-wrap; }
ul { margin: 8px 0; padding-left: 22px; }
li { margin: 7px 0; }
footer {
  margin-top: 36px;
  padding-top: 16px;
  border-top: 1px solid var(--line);
  color: var(--muted);
  font-size: 13px;
}
@media (max-width: 700px) {
  main { padding: 24px 14px 40px; }
  h1 { font-size: 24px; }
  th, td { padding: 8px 9px; }
}
"""


def _fmt_ms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:,.3f} ms"


def _fmt_mib(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024**2):,.1f} MiB"


def _memory_text(summary: MetricsSummary) -> str:
    parts = []
    if summary.peak_process_rss_bytes is not None:
        parts.append(f"peak process RSS {_fmt_mib(summary.peak_process_rss_bytes)}")
    if summary.peak_system_used_bytes is not None:
        parts.append(f"peak system memory increase {_fmt_mib(summary.peak_system_used_bytes)}")
    if summary.peak_gpu_memory_bytes is not None:
        parts.append(f"peak GPU allocation {_fmt_mib(summary.peak_gpu_memory_bytes)}")
    return "; ".join(parts) if parts else "n/a"
