from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from apdo._version import __version__
from apdo.benchmark import ConfigMeasurement
from apdo.config import BenchmarkSettings, LoaderConfig
from apdo.loader import LoaderTemplate
from apdo.metrics import MetricsSummary
from apdo.order import RoundOrder
from apdo.resources import EnvironmentInfo
from apdo.validation import ValidationResult

METHOD = {
    "protocol": (
        "Benchmark protocol 2.0. Search configurations are measured in seeded, "
        "balanced rounds. Each configuration receives discarded warmup rounds followed "
        "by one measured pass per round. The executed order is recorded."
    ),
    "search": (
        "Two-stage grid search. Stage 1 scans num_workers with default prefetch "
        "settings and applicable pin_memory values. Stage 2 varies prefetch_factor "
        "and persistent_workers for the two best multi-worker counts."
    ),
    "selection_metric": "Median steady-state throughput across balanced measured rounds.",
    "uncertainty": (
        "95% percentile bootstrap interval of the median throughput from measured "
        "rounds. Validation uses a paired bootstrap interval of per-round relative gains."
    ),
    "selection_rule": (
        "The search favors the simplest configuration whose median throughput is within "
        "5% of the fastest and whose bootstrap interval overlaps the fastest interval. "
        "The search candidate then faces independent paired validation. A candidate is "
        "recommended only if validation shows at least the configured minimum improvement "
        "and its 95% interval is above zero; otherwise the initial configuration is kept. "
        "Interval overlap during search is a heuristic, not a formal test."
    ),
    "validation": (
        "Validation uses fresh measurements, not search results. The initial and candidate "
        "configurations run in alternating balanced order for the configured validation "
        "rounds. The paired relative throughput gains are summarized by their median and a "
        "95% percentile bootstrap interval."
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
    "Balanced rounds and randomized order reduce simple ordering bias but do not remove "
    "operating-system caches, dataset caches, storage warm-up, CPU load drift, thermal "
    "effects, or memory retained between configurations.",
    "Warmup rounds are discarded and measured rounds are reported separately. A new "
    "loader is constructed for each pass, which keeps pass conditions comparable but does "
    "not model every long-lived-worker usage pattern.",
    "APDO does not reset dataset, sampler, transform, or global RNG state. Stateful or "
    "random pipelines can process different samples in different rounds.",
    "Confidence intervals reflect variation across rounds only. Validation intervals are "
    "paired percentile bootstrap intervals, not universal guarantees or formal proof.",
    "Only map-style datasets with __len__ and __getitem__ are supported in v0.2.0.",
    "Custom samplers, batch samplers, generators, collate functions, and worker_init_fn "
    "objects are reused by reference. Their mutable state is shared with the original "
    "loader and can change during benchmarking.",
    "Process RSS covers the parent process and is absolute, so it can include memory "
    "retained from earlier configurations. Worker memory is reflected only indirectly "
    "by system-wide memory availability. These memory observations are implemented "
    "for Windows and Linux.",
    "Variable-size batches are supported through actual sample counting, but the count "
    "must be inferable from tensor leading dimensions or supplied by sample_count_fn.",
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
    validation: ValidationResult
    round_orders: tuple[RoundOrder, ...] = ()
    loader_template: LoaderTemplate | None = None
    search_selected_config: LoaderConfig | None = None
    search_improvement_percent: float | None = None
    search_selection_reason: str | None = None
    budget_exhausted: bool = False
    version: str = __version__

    @property
    def best_summary(self) -> MetricsSummary:
        return self._measurement_for(self.best_config).summary_or_raise()

    @property
    def search_candidate_config(self) -> LoaderConfig:
        if self.search_selected_config is not None:
            return self.search_selected_config
        return self.validation.candidate_config

    def build_dataloader(self) -> DataLoader:
        if self.loader_template is None:
            raise ValueError("this result does not contain a reconstructable loader template")
        return self.loader_template.build(self.best_config)

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
        validation = self.validation
        gain_text = _fmt_percent(validation.improvement_percent)
        interval_text = _fmt_interval(validation)
        if validation.confirmed:
            improvement = f"validated gain: {gain_text} ({interval_text})"
        else:
            improvement = "no confirmed gain; initial configuration retained"

        lines = [
            "Recommended DataLoader configuration",
            f"  {self.best_config.describe()}",
            "",
            "Recommendation validation",
            f"  status: {validation.status}",
            f"  initial configuration: {validation.baseline_config.describe()}",
            f"  search candidate: {validation.candidate_config.describe()}",
            f"  observed median change: {gain_text} ({interval_text})",
            f"  decision: {validation.reason}",
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
            f"  {improvement}",
        ]
        if self.fastest_config != self.best_config:
            lines.append(f"  fastest measured configuration: {self.fastest_config.describe()}")
        if self.search_candidate_config != self.best_config:
            lines.append(
                f"  search candidate not retained: {self.search_candidate_config.describe()}"
            )
        lines.extend(
            [
                "",
                "Benchmark",
                f"  {self.settings.rounds} measured search rounds, "
                f"{self.settings.validation_rounds} validation rounds, "
                f"{batches} batches per pass, {self.settings.dataset_size} dataset "
                f"samples, batch_size={self._batch_size_text()}",
                f"  seed: {self.settings.seed}; input: {self.settings.input_kind}",
                f"  device: {device_note}",
                f"  {len(self.measurements)} search configurations measured ({failed} failed)",
                "",
                "Selection",
                f"  {self.selection_reason}",
            ]
        )
        if self.search_selection_reason is not None:
            lines.append(f"  search: {self.search_selection_reason}")
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"  - {warning}" for warning in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        ranked = self._ranked_successes()
        ranks = {item.config: index for index, item in enumerate(ranked, start=1)}
        fastest = self._search_measurement_for(self.fastest_config)
        fastest_summary = fastest.summary_or_raise()
        final_measurement = self._measurement_for(self.best_config)
        final_summary = final_measurement.summary_or_raise()
        baseline_measurement = self._search_measurement_for(
            self.settings.initial_config,
            required=False,
        )
        baseline_summary = (
            baseline_measurement.summary if baseline_measurement is not None else None
        )

        configurations = []
        for measurement in ranked:
            entry = measurement.to_dict()
            entry["rank"] = ranks[measurement.config]
            entry["recommended"] = measurement.config == self.best_config
            entry["search_candidate"] = measurement.config == self.search_candidate_config
            configurations.append(entry)
        for measurement in self.measurements:
            if measurement.succeeded:
                continue
            entry = measurement.to_dict()
            entry["rank"] = None
            entry["recommended"] = False
            entry["search_candidate"] = measurement.config == self.search_candidate_config
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
        measured_orders = [item.to_dict() for item in self.round_orders if item.measured]
        warmup_orders = [item.to_dict() for item in self.round_orders if not item.measured]
        baseline_payload: dict[str, Any] = {
            "config": self.settings.initial_config.to_dict(),
        }
        if baseline_summary is not None:
            baseline_payload["search_summary"] = baseline_summary.to_dict()
        return {
            "schema_version": 2,
            "apdo_version": self.version,
            "started_at_utc": self.started_at.isoformat(),
            "duration_s": self.duration_s,
            "protocol": {
                "protocol_version": self.settings.protocol_version,
                "seed": self.settings.seed,
                "search_rounds": self.settings.rounds,
                "validation_rounds": self.settings.validation_rounds,
                "min_improvement": self.settings.min_improvement,
                "validation_enabled": self.settings.validation_enabled,
                "time_budget_s": self.settings.time_budget_s,
                "budget_exhausted": self.budget_exhausted,
                "measured_orders": measured_orders,
                "warmup_orders": warmup_orders,
            },
            "environment": self.environment.to_dict(),
            "settings": self.settings.to_dict(),
            "loader": self._loader_details(),
            "method": dict(METHOD),
            "limitations": list(LIMITATIONS),
            "baseline": baseline_payload,
            "search": {
                "recommended_config": self.search_candidate_config.to_dict(),
                "fastest_config": self.fastest_config.to_dict(),
                "improvement_percent": self.search_improvement_percent,
                "selection_reason": self.search_selection_reason,
                "median_throughput": self._search_measurement_for(self.search_candidate_config)
                .summary_or_raise()
                .median_throughput,
            },
            "validation": self.validation.to_dict(),
            "recommended": {
                "config": self.best_config.to_dict(),
                "median_throughput": final_summary.median_throughput,
                "improvement_percent": self.improvement_percent,
                "validation_status": self.validation.status,
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
        measurement = self._search_measurement_for(config, required=False)
        if measurement is None:
            raise ValueError(f"no completed measurement for {config.describe()}")
        return measurement

    def _search_measurement_for(
        self,
        config: LoaderConfig,
        *,
        required: bool = True,
    ) -> ConfigMeasurement | None:
        for measurement in self.measurements:
            if (
                measurement.succeeded
                and measurement.config == config
                and measurement.summary is not None
            ):
                return measurement
        if required:
            raise ValueError(f"no completed search measurement for {config.describe()}")
        return None

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
                '<p class="muted">Fastest measured search configuration: '
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

        improvement_html = self._improvement_html()
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

        loader_details = self._loader_details()
        settings_rows = [
            ("Protocol", self.settings.protocol_version),
            ("Input", self.settings.input_kind),
            ("Loader dataset", loader_details["dataset"] if loader_details else "n/a"),
            ("Loader sampler", loader_details["sampler"] if loader_details else "n/a"),
            (
                "Loader batch sampler",
                (loader_details["batch_sampler"] or "none") if loader_details else "n/a",
            ),
            ("Loader collate", loader_details["collate_fn"] if loader_details else "n/a"),
            ("Loader drop_last", str(loader_details["drop_last"]) if loader_details else "n/a"),
            ("Loader timeout", str(loader_details["timeout"]) if loader_details else "n/a"),
            ("Seed", str(self.settings.seed)),
            ("Device", self.settings.device),
            ("Requested device", self.settings.requested_device),
            ("Dataset size", f"{self.settings.dataset_size:,} samples"),
            ("Batch size", self._batch_size_text()),
            ("Batches per pass", str(self.best_measurement_batches())),
            ("Search rounds", str(self.settings.rounds)),
            ("Validation rounds", str(self.settings.validation_rounds)),
            ("Warmup rounds", str(self.settings.warmup_runs)),
            (
                "Minimum improvement",
                f"{self.settings.min_improvement * 100:.1f}%",
            ),
            ("Worker search limit", str(self.settings.max_workers)),
            (
                "max_ram_gb",
                str(self.settings.max_ram_gb) if self.settings.max_ram_gb else "none",
            ),
            (
                "Time budget",
                f"{self.settings.time_budget_s:.2f} s"
                if self.settings.time_budget_s is not None
                else "none",
            ),
            ("Budget exhausted", "yes" if self.budget_exhausted else "no"),
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
        validation_html = self._validation_html()
        order_html = self._order_html()

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
{improvement_html}
{fastest_note}
</section>

{validation_html}

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

{order_html}

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

    def _improvement_html(self) -> str:
        escape = html.escape
        validation = self.validation
        if validation.confirmed:
            gain = _fmt_percent(validation.improvement_percent)
            return (
                f"<p><strong>Validated gain: {escape(gain)}</strong> "
                f"({escape(_fmt_interval(validation))}). "
                f"{escape(validation.reason)}</p>"
            )
        return (
            "<p><strong>No validated gain.</strong> The initial configuration is retained. "
            f"{escape(validation.reason)}</p>"
        )

    def _validation_html(self) -> str:
        escape = html.escape
        validation = self.validation
        baseline_median = (
            f"{validation.baseline_summary.median_throughput:,.1f} samples/s"
            if validation.baseline_summary is not None
            else "n/a"
        )
        candidate_median = (
            f"{validation.candidate_summary.median_throughput:,.1f} samples/s"
            if validation.candidate_summary is not None
            else "n/a"
        )
        rows = []
        for observation in validation.observations:
            rows.append(
                f"<tr><td>{observation.round_index + 1}</td>"
                f"<td>{observation.baseline.throughput:,.1f}</td>"
                f"<td>{observation.candidate.throughput:,.1f}</td>"
                f"<td>{observation.gain * 100:+.1f}%</td></tr>"
            )
        observations_html = ""
        if rows:
            observations_html = (
                '<div class="table-wrap"><table><thead><tr>'
                "<th>Round</th><th>Baseline samples/s</th><th>Candidate samples/s</th>"
                "<th>Paired gain</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></div>"
            )
        failures_html = ""
        if validation.failures:
            failure_items = "\n".join(f"<li>{escape(item)}</li>" for item in validation.failures)
            failures_html = f"<h3>Validation failures</h3><ul>{failure_items}</ul>"

        search_note = ""
        if self.search_candidate_config != self.best_config:
            search_note = (
                '<p class="muted">The search candidate was '
                f"<code>{escape(self.search_candidate_config.describe())}</code>. "
                "It was not retained because independent validation did not confirm "
                "the required gain.</p>"
            )

        return f"""<section class="card validation">
<p class="eyebrow">Recommendation Validation</p>
<p><span class="status status-{escape(validation.status)}">{escape(validation.status)}</span></p>
<div class="metrics">
<div class="metric"><span>Initial configuration</span>
<strong><code>{escape(validation.baseline_config.describe())}</code></strong></div>
<div class="metric"><span>Search candidate</span>
<strong><code>{escape(validation.candidate_config.describe())}</code></strong></div>
<div class="metric"><span>Initial median</span><strong>{escape(baseline_median)}</strong></div>
<div class="metric"><span>Candidate median</span><strong>{escape(candidate_median)}</strong></div>
<div class="metric"><span>Observed gain</span>
<strong>{escape(_fmt_percent(validation.improvement_percent))}</strong></div>
<div class="metric"><span>95% interval</span>
<strong>{escape(_fmt_interval(validation))}</strong></div>
</div>
<p><strong>Minimum practical gain:</strong>
{validation.min_improvement_percent:.1f}%</p>
<p><strong>Decision:</strong> {escape(validation.reason)}</p>
<p><strong>Final recommendation:</strong>
<code>{escape(self.best_config.describe())}</code></p>
{search_note}
{observations_html}
{failures_html}
</section>"""

    def _order_html(self) -> str:
        escape = html.escape
        measured = [item for item in self.round_orders if item.measured]
        if not measured:
            return ""
        rows = []
        for order in measured:
            configs = "<br>".join(
                f"<code>{escape(config.describe())}</code>" for config in order.configs
            )
            rows.append(
                f"<tr><td>{escape(order.phase)}</td><td>{order.round_index + 1}</td>"
                f"<td>{configs}</td></tr>"
            )
        return (
            "<section><h2>Measured execution order</h2>"
            '<p class="muted">Warmup rounds are discarded and listed in the JSON report.</p>'
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Phase</th><th>Round</th><th>Configurations in execution order</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
        )

    def best_measurement_batches(self) -> int:
        measurement = self._measurement_for(self.best_config)
        return measurement.passes[0].batches if measurement.passes else 0

    def _batch_size_text(self) -> str:
        if self.loader_template is not None and self.loader_template.unbatched:
            return "unbatched"
        if self.settings.batch_size is not None:
            return str(self.settings.batch_size)
        return "custom batch_sampler"

    def _loader_details(self) -> dict[str, Any] | None:
        template = self.loader_template
        if template is None:
            return None
        return {
            "dataset": type(template.dataset).__name__,
            "batch_size": template.batch_size,
            "unbatched": template.unbatched,
            "sampler": type(template.sampler).__name__,
            "batch_sampler": (
                type(template.batch_sampler).__name__
                if template.batch_sampler is not None
                else None
            ),
            "collate_fn": getattr(
                template.collate_fn, "__name__", type(template.collate_fn).__name__
            ),
            "drop_last": template.drop_last,
            "timeout": template.timeout,
            "worker_init_fn": (
                getattr(template.worker_init_fn, "__name__", type(template.worker_init_fn).__name__)
                if template.worker_init_fn is not None
                else None
            ),
            "generator": template.generator is not None,
            "multiprocessing_context": (
                str(template.multiprocessing_context)
                if template.multiprocessing_context is not None
                else None
            ),
            "in_order": template.in_order,
            "pin_memory_device": template.pin_memory_device or None,
            "sample_count_fn": template.sample_count_fn is not None,
        }


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
  --warning: #8a5a00;
  --warning-soft: #fff4d6;
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
h3 { font-size: 16px; margin: 20px 0 8px; }
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
.card.validation { border-left: 4px solid #355070; background: #f5f7fb; }
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
.metric strong { font-size: 15px; font-variant-numeric: tabular-nums; }
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
.status {
  display: inline-block;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.status-confirmed { background: #d7f1e8; color: #0b5d4f; }
.status-inconclusive { background: var(--warning-soft); color: var(--warning); }
.status-regression { background: #f9dddd; color: var(--failed); }
.status-not_needed { background: #e3e8ef; color: #445064; }
.status-not_possible { background: #f1e6f8; color: #64327e; }
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


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.1f}%"


def _fmt_interval(validation: ValidationResult) -> str:
    if validation.ci_low_percent is None or validation.ci_high_percent is None:
        return "n/a"
    return f"[{validation.ci_low_percent:+.1f}%, {validation.ci_high_percent:+.1f}%]"


def _memory_text(summary: MetricsSummary) -> str:
    parts = []
    if summary.peak_process_rss_bytes is not None:
        parts.append(f"peak process RSS {_fmt_mib(summary.peak_process_rss_bytes)}")
    if summary.peak_system_used_bytes is not None:
        parts.append(f"peak system memory increase {_fmt_mib(summary.peak_system_used_bytes)}")
    if summary.peak_gpu_memory_bytes is not None:
        parts.append(f"peak GPU allocation {_fmt_mib(summary.peak_gpu_memory_bytes)}")
    return "; ".join(parts) if parts else "n/a"
