from __future__ import annotations

import argparse
import json
from pathlib import Path

from apdo import optimize

try:
    from benchmarks.workloads import build_workload_spec
except ModuleNotFoundError:
    from workloads import build_workload_spec

WORKLOADS = (
    "in-memory",
    "disk-backed",
    "cpu-heavy",
    "startup-bound",
    "compute-bound",
    "training",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the six APDO v0.5.0 workloads and write JSON/HTML reports."
    )
    parser.add_argument(
        "--workload",
        choices=(*WORKLOADS, "all"),
        default="all",
        help="workload to run (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="benchmark device (default: cpu)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="measured search rounds (default: 3)",
    )
    parser.add_argument(
        "--validation-rounds",
        type=int,
        default=10,
        help="independent validation rounds (default: 10; at least 8 valid pairs needed)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="maximum workers to search (default: 8)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="override the workload batch cap",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="epochs per training trial (default: 3)",
    )
    parser.add_argument(
        "--loader-epochs",
        type=int,
        default=None,
        help="epochs per loader trial (default: 1; set this for multi-epoch qualification)",
    )
    parser.add_argument(
        "--training-rounds",
        type=int,
        default=2,
        help="measured training finalist rounds (default: 2)",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("sustained", "amortized"),
        default="sustained",
        help="metric used for search, ranking, and validation (default: sustained)",
    )
    parser.add_argument(
        "--timing-profile",
        choices=("standard", "detailed"),
        default="standard",
        help="timing profile (default: standard)",
    )
    parser.add_argument(
        "--comparability-policy",
        choices=("off", "warn", "strict"),
        default="warn",
        help="batch comparability policy (default: warn)",
    )
    parser.add_argument(
        "--no-validation",
        dest="validation",
        action="store_false",
        help="skip independent validation (smoke tests only; gains cannot be confirmed)",
    )
    parser.add_argument(
        "--search-strategy",
        choices=("grid", "adaptive"),
        default="grid",
        help="search strategy (default: grid)",
    )
    parser.add_argument(
        "--time-budget-s",
        type=float,
        default=None,
        help="cooperative optimization budget per workload",
    )
    parser.add_argument(
        "--planned-epochs",
        type=int,
        default=None,
        help="future epoch horizon for profitability estimates",
    )
    parser.add_argument(
        "--planned-sessions",
        type=int,
        default=1,
        help="future session count for profitability estimates (default: 1)",
    )
    parser.add_argument(
        "--cache-policy",
        choices=("off", "hint", "revalidate"),
        default="off",
        help="recommendation cache policy (default: off)",
    )
    parser.add_argument(
        "--cache-key",
        default=None,
        help="explicit recommendation cache key",
    )
    parser.add_argument(
        "--recommendation-cache-directory",
        type=Path,
        default=None,
        help="recommendation cache directory (default: user cache directory)",
    )
    parser.add_argument(
        "--cache-directory",
        type=Path,
        default=Path("benchmarks/.cache"),
        help="local directory for generated disk-backed data",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("benchmarks/results"),
        help="directory for JSON and HTML reports",
    )
    parser.add_argument("--seed", type=int, default=42, help="execution seed (default: 42)")
    parser.set_defaults(validation=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    names = WORKLOADS if args.workload == "all" else (args.workload,)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name in names:
        spec = build_workload_spec(name, args.cache_directory, device=args.device)
        max_batches = args.max_batches or spec.max_batches
        training = spec.mode == "training"
        epochs = args.epochs if training else (args.loader_epochs or 1)
        print(
            f"\n=== {name}: {len(spec.dataset)} samples, batch_size={spec.batch_size}, "
            f"mode={spec.mode}, epochs={epochs} ===",
            flush=True,
        )
        result = optimize(
            spec.dataset,
            batch_size=spec.batch_size,
            device=args.device,
            mode=spec.mode,
            epochs=epochs,
            training_step_factory=spec.training_step_factory,
            training_rounds=args.training_rounds,
            training_validation_rounds=args.validation_rounds,
            max_batches=max_batches,
            rounds=args.rounds,
            validation=args.validation,
            validation_rounds=args.validation_rounds,
            max_workers=args.max_workers,
            seed=args.seed,
            timing_profile=args.timing_profile,
            selection_metric=args.selection_metric,
            comparability_policy=args.comparability_policy,
            search_strategy=args.search_strategy,
            time_budget_s=args.time_budget_s,
            planned_epochs=args.planned_epochs,
            planned_sessions=args.planned_sessions,
            cache_policy=args.cache_policy,
            cache_key=args.cache_key,
            cache_directory=args.recommendation_cache_directory,
            progress=lambda message: print(message, flush=True),
        )
        json_path = args.output_directory / f"{name}.json"
        html_path = args.output_directory / f"{name}.html"
        result.to_json(json_path)
        result.to_html(html_path)
        print(result.summary())
        print(f"JSON: {json_path}")
        print(f"HTML: {html_path}")
        summaries.append(
            {
                "workload": name,
                "mode": result.settings.benchmark_mode,
                "epochs": result.settings.epochs,
                "samples": len(spec.dataset),
                "batch_size": spec.batch_size,
                "batches_per_epoch": max_batches,
                "recommended": result.best_config.describe(),
                "selection_metric": result.settings.selection_metric,
                "validation_status": result.validation.status,
                "validation_improvement_percent": result.validation.improvement_percent,
                "validation_ci_low_percent": result.validation.ci_low_percent,
                "validation_ci_high_percent": result.validation.ci_high_percent,
                "validation_valid_pairs": result.validation.valid_pairs,
                "validation_invalid_pairs": result.validation.invalid_pairs,
                "validation_completed_pairs": result.validation.completed_pairs,
                "validation_planned_pairs": result.validation.planned_pairs,
                "validation_statistical_p_value": result.validation.statistical_p_value,
                "validation_incomplete": result.validation.validation_incomplete,
                "search_improvement_percent": result.search_improvement_percent,
                "median_throughput": result.best_summary.median_throughput,
                "selection_value": (
                    result.best_summary.median_amortized_throughput
                    if result.settings.selection_metric == "amortized"
                    else result.best_summary.median_throughput
                ),
                "throughput_kind": ("end_to_end_training" if training else "loader"),
                "baseline_throughput": result.baseline_throughput,
                "median_sustained_throughput": result.best_summary.median_sustained_throughput,
                "median_amortized_throughput": result.best_summary.median_amortized_throughput,
                "median_total_elapsed_s": result.best_summary.median_total_elapsed_s,
                "comparability": (
                    result.validation.comparability.to_dict()
                    if result.validation.comparability is not None
                    else None
                ),
                "duration_s": result.duration_s,
                "search_strategy": result.settings.search_strategy,
                "search_trace": (
                    result.search_trace.to_dict() if result.search_trace is not None else None
                ),
                "budget": result.budget,
                "profitability": (
                    result.profitability.to_dict() if result.profitability is not None else None
                ),
                "cache": result.cache,
            }
        )

    summary_path = args.output_directory / "summary.json"
    existing = []
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
    by_workload = {item["workload"]: item for item in existing}
    by_workload.update({item["workload"]: item for item in summaries})
    ordered = [by_workload[name] for name in WORKLOADS if name in by_workload]
    summary_path.write_text(
        json.dumps(ordered, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
