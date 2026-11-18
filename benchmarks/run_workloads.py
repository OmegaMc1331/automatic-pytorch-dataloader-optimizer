from __future__ import annotations

import argparse
import json
from pathlib import Path

from apdo import optimize

try:
    from benchmarks.workloads import build_training_step_factory, build_workload
except ModuleNotFoundError:
    from workloads import build_training_step_factory, build_workload

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
        description="Run the six APDO v0.4.0 workloads and write JSON/HTML reports."
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
        default=5,
        help="independent validation rounds (default: 5)",
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
        "--training-rounds",
        type=int,
        default=2,
        help="measured training finalist rounds (default: 2)",
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    names = WORKLOADS if args.workload == "all" else (args.workload,)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    summaries = []

    for name in names:
        dataset, batch_size, workload_batches = build_workload(name, args.cache_directory)
        max_batches = args.max_batches or workload_batches
        training = name == "training"
        print(
            f"\n=== {name}: {len(dataset)} samples, batch_size={batch_size}, "
            f"mode={'training' if training else 'loader'} ===",
            flush=True,
        )
        result = optimize(
            dataset,
            batch_size=batch_size,
            device=args.device,
            mode="training" if training else "loader",
            epochs=args.epochs if training else 1,
            training_step_factory=(build_training_step_factory(args.device) if training else None),
            training_rounds=args.training_rounds,
            training_validation_rounds=args.validation_rounds,
            max_batches=max_batches,
            rounds=args.rounds,
            validation=True,
            validation_rounds=args.validation_rounds,
            max_workers=args.max_workers,
            seed=args.seed,
            timing_profile="standard",
            selection_metric="sustained",
            comparability_policy="warn",
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
                "samples": len(dataset),
                "batch_size": batch_size,
                "batches_per_epoch": max_batches,
                "recommended": result.best_config.describe(),
                "validation_status": result.validation.status,
                "validation_improvement_percent": result.validation.improvement_percent,
                "search_improvement_percent": result.search_improvement_percent,
                "median_throughput": result.best_summary.median_throughput,
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
