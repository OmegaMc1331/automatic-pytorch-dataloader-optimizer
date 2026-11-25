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
from apdo.loader import LoaderSource, LoaderTemplate
from apdo.metrics import (
    MetricsSummary,
    selection_metric_label,
    selection_value,
)
from apdo.order import RoundOrder
from apdo.resources import EnvironmentInfo
from apdo.validation import MIN_CONFIRMATION_PAIRS, STATISTICAL_ALPHA, ValidationResult

METHOD = {
    "protocol": (
        "Benchmark protocol 4.1. Search configurations run in loaded, seeded, balanced "
        "rounds. Loader trials reuse one DataLoader across the configured epochs; each "
        "trial is wrapped in global RNG state isolation. Training mode adds an explicit "
        "end-to-end finalist stage. Iterator initialization, first-batch acquisition, "
        "worker shutdown, loader cleanup, and total trial wall are recorded separately."
    ),
    "search": (
        "Two-stage loader grid search followed, in training mode, by a progressive "
        "end-to-end stage. Training finalists contain the initial configuration, the "
        "loader-selected candidate, the loader-fastest candidate, and the fastest "
        "remaining configurations up to the configured limit."
    ),
    "selection_metric": (
        "The configured selection metric is sustained throughput by default or amortized "
        "total-trial throughput when requested. The same metric ranks loader candidates, "
        "training finalists, report rankings, and validation gains."
    ),
    "uncertainty": (
        "95% percentile bootstrap interval of the median throughput from measured "
        "rounds. Validation uses a paired bootstrap interval of per-round relative gains "
        "and an exact one-sided paired sign test."
    ),
    "selection_rule": (
        "The search favors the simplest configuration whose selected metric is within "
        "5% of the fastest and whose bootstrap interval overlaps the fastest interval. "
        f"The search candidate then faces at least {MIN_CONFIRMATION_PAIRS} valid fresh "
        "paired validation rounds. A candidate is recommended only if the observed median "
        "reaches the configured minimum, the chosen validation policy is satisfied, and "
        "an exact one-sided paired sign test returns a p-value below "
        f"{STATISTICAL_ALPHA:.2f}. By default the lower 95% bound must also reach the "
        "minimum; the positive policy requires only an interval strictly above zero."
    ),
    "validation": (
        "Validation uses fresh measurements, not search results. The initial and candidate "
        f"configurations run in alternating balanced order for at least "
        f"{MIN_CONFIRMATION_PAIRS} valid paired rounds by default. Invalid observations "
        "are excluded and reported; paired relative throughput gains are summarized by "
        "their median, a 95% percentile bootstrap interval, and an exact one-sided paired "
        "sign test."
    ),
    "throughput_scope": (
        "Loader mode measures data loading plus host-to-device tensor copies when the "
        "resolved device is CUDA. Training mode measures the complete explicit forward, "
        "loss, backward, and optimizer step around the loader. Standard timing uses "
        "whole-cycle CUDA Events; detailed timing adds transfer and per-step device "
        "events without per-batch synchronization."
    ),
}

LIMITATIONS = (
    "Loader throughput is not training throughput. Training throughput depends on the "
    "user-supplied training_step_factory and cannot be generalized beyond that workload.",
    "The training factory must create a fresh model, optimizer, and training environment "
    "for each invocation. Reusing mutated model or optimizer state invalidates comparisons.",
    "max_ram_gb is a sampled system-memory guard, not a hard limit. Allocations can "
    "occur between samples, and unrelated processes affect the observed value.",
    "Balanced rounds and randomized order reduce simple ordering bias but do not remove "
    "operating-system caches, dataset caches, storage warm-up, CPU load drift, thermal "
    "effects, or memory retained between configurations.",
    "Warmup rounds are discarded and measured rounds are reported separately. A single "
    "loader is reused within a trial across epochs; each trial still pays its own "
    "construction and worker startup costs.",
    "Supported generator, sampler, batch-sampler, and global RNG state are isolated per "
    "trial. Undetected mutable state inside arbitrary datasets, transforms, collate "
    "functions, worker_init_fn, or user factories can still reduce comparability.",
    "Confidence intervals reflect variation across rounds only. A sign test assumes "
    "independent paired rounds from the measured process; it is not a universal guarantee "
    "or formal proof of future performance.",
    "Validation requires at least eight valid paired rounds for confirmation. Shorter "
    "experiments are mechanism checks; they cannot confirm a candidate.",
    "High recommendation agreement across independent stability runs does not by itself "
    "demonstrate that the workload is stable. Runs that fall back to the initial "
    "configuration are reported separately.",
    "APDO supports standard map-style DataLoaders with finite samplers and an explicit "
    "dataloader_factory contract for pipelines that need independent construction.",
    "Process RSS covers the parent process and is absolute, so it can include memory "
    "retained from earlier configurations. Worker memory is reflected only indirectly "
    "by system-wide memory availability. These memory observations are implemented "
    "for Windows and Linux.",
    "Variable-size batches are supported through actual sample counting, but the count "
    "must be inferable from tensor leading dimensions or supplied by sample_count_fn.",
    "Comparability diagnostics hash batch structure and optional sample fingerprints. "
    "They can detect supported differences in sampled content and order, but they are "
    "not a proof that arbitrary data pipelines are equivalent.",
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
    baseline_amortized_throughput: float | None
    improvement_percent: float | None
    started_at: datetime
    duration_s: float
    validation: ValidationResult
    round_orders: tuple[RoundOrder, ...] = ()
    loader_template: LoaderTemplate | None = None
    loader_source: LoaderSource | None = None
    search_selected_config: LoaderConfig | None = None
    search_improvement_percent: float | None = None
    search_selection_reason: str | None = None
    budget_exhausted: bool = False
    version: str = __version__

    @property
    def comparability(self) -> Any:
        return self.validation.comparability

    @property
    def best_summary(self) -> MetricsSummary:
        return self._measurement_for(self.best_config).summary_or_raise()

    @property
    def search_candidate_config(self) -> LoaderConfig:
        if self.search_selected_config is not None:
            return self.search_selected_config
        return self.validation.candidate_config

    def build_dataloader(self) -> DataLoader:
        if self.loader_source is not None and self.loader_source.factory is not None:
            return self.loader_source.build(self.best_config, isolate=False)
        if self.loader_template is None:
            raise ValueError("this result does not contain a reconstructable loader template")
        return self.loader_template.build(self.best_config)

    def summary(self) -> str:
        measurement = self._measurement_for(self.best_config)
        summary = self.best_summary
        batches = self.epoch_batches()
        failed = sum(1 for item in self.measurements if not item.succeeded)
        training = self.settings.benchmark_mode == "training"
        device_note = (
            "cuda (includes host-to-device copies)"
            if self.settings.device.startswith("cuda")
            else self.settings.device
        )
        throughput_label = "training samples/s" if training else "samples/s"
        memory = _memory_text(summary)
        cold_start = _fmt_ms(measurement.cold_start_s)
        validation = self.validation
        gain_text = _fmt_percent(validation.improvement_percent)
        interval_text = _fmt_interval(validation)
        selection_metric = self.settings.selection_metric
        selection_label = selection_metric_label(selection_metric)
        selected_value = selection_value(summary, selection_metric)
        selected_value_text = (
            f"{selected_value:,.1f} {throughput_label}" if selected_value is not None else "n/a"
        )
        statistical_text = (
            f"p={validation.statistical_p_value:.4f} "
            f"(alpha {validation.statistical_alpha:.2f}, "
            f"{validation.sign_test_successes}/{validation.valid_pairs} rounds)"
            if validation.statistical_p_value is not None
            else "n/a"
        )
        if validation.confirmed:
            improvement = f"validated gain: {gain_text} ({interval_text})"
        else:
            improvement = "no confirmed gain; initial configuration retained"

        lines = [
            "Recommended DataLoader configuration",
            f"  {self.best_config.describe()}",
            f"  benchmark mode: {self.settings.benchmark_mode}; epochs: {self.settings.epochs}",
            f"  selection metric: {selection_label} ({selected_value_text})",
            "",
            "Recommendation validation",
            f"  status: {validation.status}",
            f"  initial configuration: {validation.baseline_config.describe()}",
            f"  search candidate: {validation.candidate_config.describe()}",
            f"  observed median change: {gain_text} ({interval_text})",
            f"  evidence: {validation.valid_pairs} valid of {validation.completed_pairs} "
            f"completed pairs ({validation.invalid_pairs} invalid); minimum "
            f"{validation.minimum_required_pairs}; planned {validation.planned_pairs}",
            f"  statistical test: {validation.statistical_method}; {statistical_text}",
            f"  decision: {validation.reason}",
            "",
            "Measured performance",
            f"  median throughput: {summary.median_throughput:,.1f} {throughput_label}",
            f"  throughput range: {summary.min_throughput:,.1f} - "
            f"{summary.max_throughput:,.1f} {throughput_label}",
            f"  median batch latency: {_fmt_ms(summary.median_latency_s)} "
            f"(p95 {_fmt_ms(summary.p95_latency_s)})",
            f"  startup to first batch: {_fmt_ms(summary.median_startup_s)} "
            f"(cold start {cold_start})",
            f"  amortized trial throughput: "
            f"{_fmt_optional_throughput(summary.median_amortized_throughput)}",
            f"  median total trial wall: {_fmt_ms(summary.median_total_elapsed_s)}",
            f"  lifecycle: iterator {_fmt_ms(summary.median_iterator_init_s)}, "
            f"first batch {_fmt_ms(summary.median_first_batch_s)}, "
            f"worker shutdown {_fmt_ms(summary.median_worker_shutdown_s)}, "
            f"loader cleanup {_fmt_ms(summary.median_loader_cleanup_s)}",
            f"  observed memory: {memory}",
            f"  {improvement}",
        ]
        if summary.median_sustained_throughput is not None and self.settings.epochs > 1:
            lines.append(
                f"  sustained throughput: {summary.median_sustained_throughput:,.1f} "
                f"{throughput_label}"
            )
        if training:
            lines.append(
                "  training step time: "
                f"{_fmt_ms(summary.median_step_latency_s)} median; "
                f"data wait {_fmt_ms(summary.median_data_wait_s)}; "
                f"transfer {_fmt_ms(summary.median_transfer_s)}"
            )
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
                + (f"{self.settings.training_rounds} training rounds, " if training else "")
                + f"{self.settings.validation_rounds} validation rounds, "
                f"{batches} batches per epoch, {self.settings.dataset_size} dataset "
                f"samples, batch_size={self._batch_size_text()}",
                f"  seed: {self.settings.seed}; input: {self.settings.input_kind}; "
                f"reproducibility: {self.settings.reproducibility_status}",
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
        loader_baseline = self._search_measurement_for(
            self.settings.initial_config,
            required=False,
            training=False,
        )
        training_baseline = self._search_measurement_for(
            self.settings.initial_config,
            required=False,
            training=True,
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
        if loader_baseline is not None and loader_baseline.summary is not None:
            baseline_payload["search_summary"] = loader_baseline.summary.to_dict()
        if training_baseline is not None and training_baseline.summary is not None:
            baseline_payload["training_summary"] = training_baseline.summary.to_dict()
            baseline_payload["training_config"] = training_baseline.config.to_dict()
        return {
            "schema_version": 4,
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
                "validation_policy": self.settings.validation_policy,
                "timing_profile": self.settings.timing_profile,
                "selection_metric": self.settings.selection_metric,
                "comparability_policy": self.settings.comparability_policy,
                "sample_fingerprint": self.settings.sample_fingerprint,
                "training_rounds": self.settings.training_rounds,
                "training_validation_rounds": self.settings.training_validation_rounds,
                "max_training_candidates": self.settings.max_training_candidates,
                "minimum_confirmation_pairs": MIN_CONFIRMATION_PAIRS,
                "statistical_method": self.validation.statistical_method,
                "statistical_alpha": self.validation.statistical_alpha,
                "time_budget_s": self.settings.time_budget_s,
                "budget_exhausted": self.budget_exhausted,
                "measured_orders": measured_orders,
                "warmup_orders": warmup_orders,
            },
            "benchmark_mode": self.settings.benchmark_mode,
            "epochs": self.settings.epochs,
            "environment": self.environment.to_dict(),
            "reproducibility": self._reproducibility_details(),
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
                "median_amortized_throughput": self._search_measurement_for(
                    self.search_candidate_config
                )
                .summary_or_raise()
                .median_amortized_throughput,
                "selection_metric": self.settings.selection_metric,
                "selection_value": self._selection_value_for(
                    self._search_measurement_for(self.search_candidate_config)
                ),
                "throughput_kind": (
                    "end_to_end_training"
                    if self.settings.benchmark_mode == "training"
                    else "loader"
                ),
                "epochs": self.settings.epochs,
            },
            "training": self._training_details(),
            "validation": self.validation.to_dict(),
            "timing": self._timing_details(final_measurement),
            "comparability": (
                self.validation.comparability.to_dict()
                if self.validation.comparability is not None
                else None
            ),
            "recommended": {
                "config": self.best_config.to_dict(),
                "median_throughput": final_summary.median_throughput,
                "median_sustained_throughput": final_summary.median_sustained_throughput,
                "median_amortized_throughput": final_summary.median_amortized_throughput,
                "median_total_elapsed_s": final_summary.median_total_elapsed_s,
                "selection_metric": self.settings.selection_metric,
                "selection_value": self._selection_value_for(final_measurement),
                "throughput_kind": (
                    "end_to_end_training"
                    if self.settings.benchmark_mode == "training"
                    else "loader"
                ),
                "improvement_percent": self.improvement_percent,
                "validation_status": self.validation.status,
                "selection_reason": self.selection_reason,
            },
            "fastest": {
                "config": self.fastest_config.to_dict(),
                "median_throughput": fastest_summary.median_throughput,
                "throughput_kind": (
                    "end_to_end_training"
                    if self.settings.benchmark_mode == "training"
                    else "loader"
                ),
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
        training: bool | None = None,
    ) -> ConfigMeasurement | None:
        expected_training = (
            self.settings.benchmark_mode == "training" if training is None else training
        )
        matches = [
            measurement
            for measurement in self.measurements
            if (
                measurement.succeeded
                and measurement.config == config
                and measurement.summary is not None
                and measurement.training == expected_training
            )
        ]
        if matches:
            return max(matches, key=self._selection_value_for)
        if required:
            phase = "training " if expected_training else ""
            raise ValueError(f"no completed {phase}measurement for {config.describe()}")
        return None

    def _selection_value_for(self, measurement: ConfigMeasurement) -> float:
        if measurement.summary is None:
            return 0.0
        value = selection_value(measurement.summary, self.settings.selection_metric)
        return value if value is not None else 0.0

    def _ranked_successes(self) -> list[ConfigMeasurement]:
        training = self.settings.benchmark_mode == "training"
        ranked = [
            item
            for item in self.measurements
            if item.succeeded and item.summary is not None and item.training == training
        ]
        ranked.sort(key=self._selection_value_for, reverse=True)
        return ranked

    def _html(self) -> str:
        escape = html.escape
        best = self.best_summary
        best_measurement = self._measurement_for(self.best_config)
        failed = [item for item in self.measurements if not item.succeeded]
        device_is_cuda = self.settings.device.startswith("cuda")
        training = self.settings.benchmark_mode == "training"
        throughput_label = "training samples/s" if training else "samples/s"
        fastest_note = ""
        if self.fastest_config != self.best_config:
            fastest_note = (
                '<p class="muted">Fastest measured search configuration: '
                f"<code>{escape(self.fastest_config.describe())}</code></p>"
            )

        metric_cards = [
            (
                "Median training throughput" if training else "Median throughput",
                f"{best.median_throughput:,.1f} {throughput_label}",
            ),
            (
                f"Selected {selection_metric_label(self.settings.selection_metric)}",
                _fmt_optional_throughput(selection_value(best, self.settings.selection_metric)),
            ),
            ("Median batch latency", _fmt_ms(best.median_latency_s)),
            ("p95 batch latency", _fmt_ms(best.p95_latency_s)),
            ("Median startup", _fmt_ms(best.median_startup_s)),
            ("Cold start", _fmt_ms(best_measurement.cold_start_s)),
            ("Observed memory", _memory_text(best)),
        ]
        if training:
            metric_cards.extend(
                [
                    ("Median step time", _fmt_ms(best.median_step_latency_s)),
                    ("Median data wait", _fmt_ms(best.median_data_wait_s)),
                    ("Median transfer", _fmt_ms(best.median_transfer_s)),
                ]
            )
        if self.settings.epochs > 1:
            metric_cards.append(
                (
                    "Sustained throughput",
                    f"{_fmt_optional_throughput(best.median_sustained_throughput)}",
                )
            )
        metric_cards.extend(
            [
                (
                    "Amortized trial throughput",
                    _fmt_optional_throughput(best.median_amortized_throughput),
                ),
                ("Median total trial wall", _fmt_ms(best.median_total_elapsed_s)),
                ("First batch", _fmt_ms(best.median_first_batch_s)),
                ("Worker shutdown", _fmt_ms(best.median_worker_shutdown_s)),
                ("Loader cleanup", _fmt_ms(best.median_loader_cleanup_s)),
            ]
        )
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

        cycles_html = self._cycles_html()
        training_html = self._training_html()
        reproducibility_html = self._reproducibility_html()
        rows = []
        for rank, measurement in enumerate(self._ranked_successes(), start=1):
            summary = measurement.summary
            if summary is None:
                continue
            row_class = ' class="recommended"' if measurement.config == self.best_config else ""
            gpu_cell = ""
            if device_is_cuda:
                gpu_cell = f"<td>{_fmt_mib(summary.peak_gpu_memory_bytes)}</td>"
            selected = selection_value(summary, self.settings.selection_metric)
            rows.append(
                f'<tr{row_class}><td class="rank">{rank}</td>'
                f"<td><code>{escape(measurement.config.describe())}</code></td>"
                f"<td>{summary.median_throughput:,.1f}</td>"
                f"<td>{_fmt_optional_throughput(selected)}</td>"
                f"<td>{_fmt_optional_throughput(summary.median_amortized_throughput)}</td>"
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
            ("Mode", self.settings.benchmark_mode),
            ("Epochs", str(self.settings.epochs)),
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
            ("Training rounds", str(self.settings.training_rounds)),
            ("Training validation rounds", str(self.settings.training_validation_rounds)),
            ("Validation rounds", str(self.settings.validation_rounds)),
            ("Warmup rounds", str(self.settings.warmup_runs)),
            ("Validation policy", self.settings.validation_policy),
            ("Timing profile", self.settings.timing_profile),
            ("Selection metric", self.settings.selection_metric),
            ("Comparability policy", self.settings.comparability_policy),
            ("Sample fingerprint", "enabled" if self.settings.sample_fingerprint else "disabled"),
            ("Reproducibility", self.settings.reproducibility_status),
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
        timing_html = self._timing_html()
        comparability_html = self._comparability_html()
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
completed in {self.duration_s:.2f} s &middot;
mode {escape(self.settings.benchmark_mode)} &middot;
{self.settings.epochs} epoch(s)</p>
</header>

<section class="card accent">
<p class="eyebrow">Recommended configuration</p>
<h2><code>{escape(self.best_config.describe())}</code></h2>
<div class="metrics">{cards_html}</div>
{improvement_html}
{fastest_note}
</section>

{validation_html}

{comparability_html}

{timing_html}

{training_html}

{reproducibility_html}

{warnings_html}

<section>
<h2>Ranking</h2>
<div class="table-wrap">
<table>
<thead><tr>
<th>Rank</th><th>Configuration</th><th>Median sustained</th>
<th>Selected metric</th><th>Median amortized</th><th>CV</th>
<th>Median batch</th><th>p95 batch</th><th>Startup</th><th>Peak RSS</th>{gpu_header}
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
</section>

{order_html}

{cycles_html}

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
requires no network access. Throughput is {escape(throughput_label)} in
{escape(self.settings.benchmark_mode)} mode.
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
        metric_label = selection_metric_label(validation.selection_metric)
        baseline_selected = (
            selection_value(validation.baseline_summary, validation.selection_metric)
            if validation.baseline_summary is not None
            else None
        )
        candidate_selected = (
            selection_value(validation.candidate_summary, validation.selection_metric)
            if validation.candidate_summary is not None
            else None
        )
        baseline_median = _fmt_optional_throughput(baseline_selected)
        candidate_median = _fmt_optional_throughput(candidate_selected)
        rows = []
        for observation in validation.observations:
            data = observation.to_dict(validation.selection_metric)
            baseline_value = data["baseline_selection_throughput"]
            candidate_value = data["candidate_selection_throughput"]
            gain_value = data["gain_selection_percent"]
            rows.append(
                f"<tr><td>{observation.round_index + 1}</td>"
                f"<td>{_fmt_optional_throughput(baseline_value)}</td>"
                f"<td>{_fmt_optional_throughput(candidate_value)}</td>"
                f"<td>{_fmt_percent(gain_value)}</td>"
                f"<td>{'yes' if data['valid'] else 'no'}</td>"
                f"<td>{escape(data['validity_reason'] or '')}</td></tr>"
            )
        observations_html = ""
        if rows:
            observations_html = (
                '<div class="table-wrap"><table><thead><tr>'
                f"<th>Round</th><th>Baseline {escape(metric_label)}</th>"
                f"<th>Candidate {escape(metric_label)}</th>"
                "<th>Paired gain</th><th>Valid</th><th>Rejection reason</th>"
                "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            )
        rejections_html = ""
        if validation.invalid_reasons:
            rejection_items = "\n".join(
                f"<li>{escape(item)}</li>" for item in validation.invalid_reasons
            )
            rejections_html = f"<h3>Rejected observations</h3><ul>{rejection_items}</ul>"
        failures_html = ""
        if validation.failures:
            failure_items = "\n".join(f"<li>{escape(item)}</li>" for item in validation.failures)
            failures_html = f"<h3>Validation failures</h3><ul>{failure_items}</ul>"
        statistical_text = (
            f"{validation.statistical_method}, alpha {validation.statistical_alpha:.2f}; "
            f"{validation.sign_test_successes}/{validation.valid_pairs} successful pairs; "
            f"p={validation.statistical_p_value:.4f}"
            if validation.statistical_p_value is not None
            else f"{validation.statistical_method}; no valid paired sample"
        )

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
<div class="metric"><span>Initial {escape(metric_label)}</span>
<strong>{escape(baseline_median)}</strong></div>
<div class="metric"><span>Candidate {escape(metric_label)}</span>
<strong>{escape(candidate_median)}</strong></div>
<div class="metric"><span>Observed gain</span>
<strong>{escape(_fmt_percent(validation.improvement_percent))}</strong></div>
<div class="metric"><span>95% interval</span>
<strong>{escape(_fmt_interval(validation))}</strong></div>
<div class="metric"><span>Valid pairs</span>
<strong>{validation.valid_pairs} / {validation.completed_pairs} completed
(minimum {validation.minimum_required_pairs}; planned {validation.planned_pairs})</strong></div>
<div class="metric"><span>Rejected pairs</span>
<strong>{validation.invalid_pairs}</strong></div>
</div>
<p><strong>Selection metric:</strong> {escape(metric_label)}</p>
<p><strong>Validation policy:</strong> {escape(validation.policy)}</p>
<p><strong>Minimum practical gain:</strong>
{validation.min_improvement_percent:.1f}%</p>
<p><strong>Statistical test:</strong> {escape(statistical_text)}</p>
<p><strong>95% lower bound meets minimum:</strong>
{"yes" if validation.threshold_guaranteed else "no"}</p>
<p><strong>Validation complete:</strong>
{"no" if validation.validation_incomplete else "yes"}</p>
<p><strong>Decision:</strong> {escape(validation.reason)}</p>
<p><strong>Final recommendation:</strong>
<code>{escape(self.best_config.describe())}</code></p>
{search_note}
{observations_html}
{rejections_html}
{failures_html}
</section>"""

    def _training_html(self) -> str:
        details = self._training_details()
        if details is None:
            return ""
        escape = html.escape
        baseline_summary = details["baseline"]["summary"]
        selected_summary = details["selected"]["summary"]
        return f"""<section class="card validation">
<p class="eyebrow">End-to-End Training</p>
<div class="metrics">
<div class="metric"><span>Initial training config</span>
<strong><code>{escape(self.settings.initial_config.describe())}</code></strong></div>
<div class="metric"><span>Training-selected config</span>
<strong><code>{escape(self.search_candidate_config.describe())}</code></strong></div>
<div class="metric"><span>Initial training throughput</span>
<strong>{baseline_summary["median_throughput"]:,.1f} training samples/s</strong></div>
<div class="metric"><span>Selected training throughput</span>
<strong>{selected_summary["median_throughput"]:,.1f} training samples/s</strong></div>
<div class="metric"><span>Epochs per trial</span>
<strong>{details["epochs"]}</strong></div>
<div class="metric"><span>Training rounds</span>
<strong>{details["rounds"]}</strong></div>
<div class="metric"><span>Median step time</span>
<strong>{_fmt_ms(details["median_step_latency_s"])}</strong></div>
<div class="metric"><span>Median data wait</span>
<strong>{_fmt_ms(details["median_data_wait_s"])}</strong></div>
<div class="metric"><span>Median transfer</span>
<strong>{_fmt_ms(details["median_transfer_s"])}</strong></div>
</div>
<p class="muted">Training throughput includes the explicit user-supplied forward,
loss, backward, and optimizer step around the DataLoader.</p>
</section>"""

    def _reproducibility_html(self) -> str:
        details = self._reproducibility_details()
        escape = html.escape
        risks = details.get("risks") or []
        risks_html = ""
        if risks:
            items = "".join(f"<li>{escape(item)}</li>" for item in risks)
            risks_html = f"<h3>Detected risks</h3><ul>{items}</ul>"
        managed = details.get("managed") or []
        managed_text = ", ".join(managed) if managed else "none"
        return f"""<section>
<h2>Reproducibility</h2>
<div class="table-wrap"><table class="kv"><tbody>
<tr><th>Status</th><td><code>{escape(str(details.get("status", "unknown")))}</code></td></tr>
<tr><th>Managed state</th><td>{escape(managed_text)}</td></tr>
<tr><th>Training allowed</th><td>{details.get("training_allowed")}</td></tr>
</tbody></table></div>
<p>{escape(str(details.get("reason", "")))}</p>
{risks_html}
</section>"""

    def _cycles_html(self) -> str:
        rows: list[str] = []
        measurement = self._measurement_for(self.best_config)
        for pass_index, metrics in enumerate(measurement.passes, start=1):
            for cycle in metrics.cycles:
                rows.append(
                    f"<tr><td>{pass_index}</td><td>{cycle.cycle_index + 1}</td>"
                    f"<td>{cycle.batches}</td><td>{cycle.samples}</td>"
                    f"<td>{cycle.throughput:,.1f}</td>"
                    f"<td>{_fmt_ms(cycle.startup_s)}</td>"
                    f"<td>{_fmt_ms(cycle.data_wait_s)}</td>"
                    f"<td>{_fmt_ms(cycle.transfer_s)}</td></tr>"
                )
        if not rows:
            return ""
        return (
            "<section><h2>Epoch measurements</h2>"
            '<div class="table-wrap"><table><thead><tr>'
            "<th>Round</th><th>Epoch</th><th>Batches</th><th>Samples</th>"
            "<th>Samples/s</th><th>Startup</th><th>Data wait</th><th>Transfer</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></section>"
        )

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
        return self.epoch_batches()

    def epoch_batches(self) -> int:
        measurement = self._measurement_for(self.best_config)
        passes = measurement.passes
        if not passes:
            return 0
        if passes[0].cycles:
            return passes[0].cycles[0].batches
        return passes[0].batches

    def _reproducibility_details(self) -> dict[str, Any]:
        if self.loader_source is None:
            return {
                "status": self.settings.reproducibility_status,
                "managed": [],
                "risks": [],
                "reason": "No loader source metadata is attached to this result.",
                "training_allowed": self.settings.reproducibility_status
                in {"verified", "factory_managed"},
            }
        return self.loader_source.isolation.to_dict()

    def _training_details(self) -> dict[str, Any] | None:
        if self.settings.benchmark_mode != "training":
            return None
        baseline = self._search_measurement_for(
            self.settings.initial_config,
            required=False,
            training=True,
        )
        selected = self._search_measurement_for(
            self.search_candidate_config,
            required=False,
            training=True,
        )
        if baseline is None or selected is None:
            return None
        return {
            "epochs": self.settings.epochs,
            "rounds": self.settings.training_rounds,
            "validation_rounds": self.settings.training_validation_rounds,
            "baseline": {
                "config": baseline.config.to_dict(),
                "summary": baseline.summary_or_raise().to_dict(),
            },
            "selected": {
                "config": selected.config.to_dict(),
                "summary": selected.summary_or_raise().to_dict(),
            },
            "median_data_wait_s": selected.summary_or_raise().median_data_wait_s,
            "median_step_latency_s": selected.summary_or_raise().median_step_latency_s,
            "median_transfer_s": selected.summary_or_raise().median_transfer_s,
            "total_transfer_bytes": selected.summary_or_raise().total_transfer_bytes,
        }

    def _timing_details(self, measurement: ConfigMeasurement) -> dict[str, Any] | None:
        summary = measurement.summary
        if summary is None:
            return None
        return {
            "timing_profile": self.settings.timing_profile,
            "selection_metric": self.settings.selection_metric,
            "median_throughput": summary.median_throughput,
            "selection_value": selection_value(summary, self.settings.selection_metric),
            "median_total_elapsed_s": summary.median_total_elapsed_s,
            "median_amortized_throughput": summary.median_amortized_throughput,
            "ci_low_amortized_throughput": summary.ci_low_amortized_throughput,
            "ci_high_amortized_throughput": summary.ci_high_amortized_throughput,
            "median_construction_s": summary.median_construction_s,
            "median_training_setup_s": summary.median_training_setup_s,
            "median_iterator_init_s": summary.median_iterator_init_s,
            "median_first_batch_s": summary.median_first_batch_s,
            "median_startup_s": summary.median_startup_s,
            "median_worker_shutdown_s": summary.median_worker_shutdown_s,
            "median_loader_cleanup_s": summary.median_loader_cleanup_s,
            "median_sustained_throughput": summary.median_sustained_throughput,
            "median_cuda_execution_s": summary.median_cuda_execution_s,
            "median_transfer_device_s": summary.median_transfer_device_s,
            "median_step_device_latency_s": summary.median_step_device_latency_s,
        }

    def _timing_html(self) -> str:
        measurement = self._measurement_for(self.best_config)
        details = self._timing_details(measurement)
        if details is None:
            return ""
        escape = html.escape
        rows = [
            ("Timing profile", self.settings.timing_profile),
            ("Selection metric", self.settings.selection_metric),
            ("Median sustained throughput", f"{details['median_throughput']:,.1f} samples/s"),
            (
                f"Median {selection_metric_label(self.settings.selection_metric)}",
                _fmt_optional_throughput(details["selection_value"]),
            ),
            ("Median total trial wall", _fmt_ms(details["median_total_elapsed_s"])),
            (
                "Median amortized throughput",
                _fmt_optional_throughput(details["median_amortized_throughput"]),
            ),
            (
                "Amortized 95% interval",
                (
                    f"[{details['ci_low_amortized_throughput']:,.1f}, "
                    f"{details['ci_high_amortized_throughput']:,.1f}] samples/s"
                    if details["ci_low_amortized_throughput"] is not None
                    and details["ci_high_amortized_throughput"] is not None
                    else "n/a"
                ),
            ),
            ("Loader construction", _fmt_ms(details["median_construction_s"])),
            ("Training setup", _fmt_ms(details["median_training_setup_s"])),
            ("Iterator initialization", _fmt_ms(details["median_iterator_init_s"])),
            ("First batch acquisition", _fmt_ms(details["median_first_batch_s"])),
            ("Startup to first batch", _fmt_ms(details["median_startup_s"])),
            ("Worker shutdown", _fmt_ms(details["median_worker_shutdown_s"])),
            ("Loader cleanup", _fmt_ms(details["median_loader_cleanup_s"])),
            ("CUDA execution", _fmt_ms(details["median_cuda_execution_s"])),
            ("Device transfer", _fmt_ms(details["median_transfer_device_s"])),
            ("Step device time", _fmt_ms(details["median_step_device_latency_s"])),
        ]
        body = "".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows
        )
        return (
            "<section><h2>Timing Breakdown</h2>"
            '<p class="muted">The total trial wall includes construction, every epoch '
            "startup and shutdown, and final loader cleanup. Sustained throughput excludes "
            "the first loader batch; amortized throughput uses the total wall.</p>"
            f'<div class="table-wrap"><table class="kv"><tbody>{body}</tbody></table></div>'
            "</section>"
        )

    def _comparability_html(self) -> str:
        comparison = self.validation.comparability
        escape = html.escape
        if comparison is None:
            return (
                "<section><h2>Batch Comparability</h2>"
                f"<p>No comparison was produced under the "
                f"<code>{escape(self.settings.comparability_policy)}</code> policy.</p>"
                "</section>"
            )
        rows = [
            ("Status", comparison.status),
            ("Confidence", comparison.confidence),
            ("Compared rounds", str(comparison.rounds_compared)),
            ("Rounds with fingerprints", str(comparison.rounds_with_fingerprints)),
            ("Structural mismatch rounds", str(comparison.structural_mismatch_rounds)),
            ("Different sample rounds", str(comparison.different_sample_rounds)),
            (
                "Structural identity",
                "verified" if comparison.structural_verified else "not verified",
            ),
            (
                "Sample identity",
                "verified" if comparison.sample_identity_verified else "not verified",
            ),
            ("Sufficient for strict policy", "yes" if comparison.sufficient_for_strict else "no"),
        ]
        body = "".join(
            f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>" for label, value in rows
        )
        notes = ""
        if comparison.notes:
            notes = (
                "<ul>" + "".join(f"<li>{escape(note)}</li>" for note in comparison.notes) + "</ul>"
            )
        return (
            "<section><h2>Batch Comparability</h2>"
            f"<p>{escape(comparison.summary)}</p>"
            f'<div class="table-wrap"><table class="kv"><tbody>{body}</tbody></table></div>'
            f"{notes}</section>"
        )

    def _batch_size_text(self) -> str:
        if self.loader_template is not None and self.loader_template.unbatched:
            return "unbatched"
        if self.settings.batch_size is not None:
            return str(self.settings.batch_size)
        return "custom batch_sampler"

    def _loader_details(self) -> dict[str, Any] | None:
        template = self.loader_template
        if template is None and self.loader_source is not None:
            template = self.loader_source.template
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


def _fmt_optional_throughput(value: float | None) -> str:
    return f"{value:,.1f} samples/s" if value is not None else "n/a"


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
