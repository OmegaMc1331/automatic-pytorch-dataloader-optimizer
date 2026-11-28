from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from apdo._version import __version__
from apdo.cache import ConfigCache
from apdo.errors import ApdoError
from apdo.optimizer import optimize
from apdo.stability import evaluate_stability


class _CliError(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apdo",
        description=(
            "Benchmark PyTorch DataLoader configurations and recommend settings "
            "for the current machine."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="benchmark candidate DataLoader configurations",
        description=(
            "Load a map-style dataset, an existing DataLoader, or an explicit "
            "DataLoader factory, benchmark loader or end-to-end training settings, "
            "validate the recommendation, and print the result."
        ),
        epilog=(
            "Examples:\n"
            "  apdo benchmark --dataset my_project.data:build_loader "
            "--device auto --json report.json --html report.html\n"
            "  apdo benchmark --factory my_project.data:make_loader --mode training "
            "--training-step-factory my_project.train:make_step --epochs 3\n\n"
            "Dataset factories are zero-argument callables returning a map-style "
            "dataset or DataLoader. Dataloader factories accept one LoaderConfig "
            "argument and return an independent DataLoader. Training step factories "
            "are zero-argument callables returning a step(batch) callable that performs "
            "forward, loss, backward, and optimizer work. On Windows, classes must be "
            "defined at module level so worker processes can import them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = benchmark.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset",
        metavar="MODULE:FACTORY",
        help="import path of a zero-argument dataset or DataLoader factory",
    )
    input_group.add_argument(
        "--factory",
        metavar="MODULE:FACTORY",
        help=(
            "import path of a callable accepting LoaderConfig and returning an "
            "independent DataLoader"
        ),
    )
    benchmark.add_argument(
        "--mode",
        choices=("loader", "training"),
        default="loader",
        help="benchmark DataLoader only or end-to-end training (default: loader)",
    )
    benchmark.add_argument(
        "--epochs",
        type=_positive_int,
        default=None,
        help="epochs per trial; default 1 in loader mode and 3 in training mode",
    )
    benchmark.add_argument(
        "--training-step-factory",
        metavar="MODULE:FACTORY",
        default=None,
        help="import path of a zero-argument training step factory",
    )
    benchmark.add_argument(
        "--training-rounds",
        type=_at_least_two,
        default=2,
        help="measured training finalist rounds (default: 2)",
    )
    benchmark.add_argument(
        "--training-validation-rounds",
        type=_at_least_two,
        default=None,
        help="paired validation rounds in training mode (default: validation rounds)",
    )
    benchmark.add_argument(
        "--max-training-candidates",
        type=_positive_int,
        default=3,
        help="maximum end-to-end training finalists (default: 3)",
    )
    benchmark.add_argument(
        "--validation-policy",
        choices=("minimum", "positive"),
        default="minimum",
        help=(
            "minimum: lower 95%% bound must reach min-improvement; "
            "positive: lower bound must exceed zero (default: minimum)"
        ),
    )
    benchmark.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help="batch size for dataset factories (default: 32); DataLoader batch size is preserved",
    )
    benchmark.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or 'cuda:N' (default: auto)",
    )
    benchmark.add_argument(
        "--max-ram-gb",
        type=_positive_float,
        default=None,
        help="soft limit on observed benchmark memory growth",
    )
    benchmark.add_argument(
        "--warmup-runs",
        type=_positive_int,
        default=1,
        help="discarded warmup rounds per configuration (default: 1)",
    )
    benchmark.add_argument(
        "--repetitions",
        type=_at_least_two,
        default=3,
        help="legacy alias for measured search rounds (default: 3)",
    )
    benchmark.add_argument(
        "--rounds",
        type=_at_least_two,
        default=None,
        help="measured balanced search rounds per configuration (default: repetitions)",
    )
    benchmark.add_argument(
        "--seed",
        type=_nonnegative_int,
        default=42,
        help="seed for reproducible execution order (default: 42)",
    )
    benchmark.add_argument(
        "--validation-rounds",
        type=_at_least_two,
        default=10,
        help="independent paired validation rounds (default: 10; at least 8 valid pairs needed)",
    )
    benchmark.add_argument(
        "--min-improvement",
        type=_fraction,
        default=0.05,
        help="minimum validated relative gain, as a fraction (default: 0.05)",
    )
    benchmark.add_argument(
        "--no-validation",
        dest="validation",
        action="store_false",
        help=(
            "skip independent validation; the initial configuration is retained "
            "for an unconfirmed search result"
        ),
    )
    benchmark.add_argument(
        "--time-budget-s",
        type=_positive_float,
        default=None,
        help="cooperative optimization time budget in seconds",
    )
    benchmark.add_argument(
        "--search-strategy",
        choices=("grid", "adaptive"),
        default="grid",
        help="grid search or deterministic adaptive screening/refinement (default: grid)",
    )
    benchmark.add_argument(
        "--planned-epochs",
        type=_positive_int,
        default=None,
        help="future epoch horizon for the profitability estimate",
    )
    benchmark.add_argument(
        "--planned-sessions",
        type=_positive_int,
        default=1,
        help="future training sessions for the profitability estimate (default: 1)",
    )
    benchmark.add_argument(
        "--cache-policy",
        choices=("off", "hint", "revalidate"),
        default="off",
        help="local recommendation cache behavior (default: off)",
    )
    benchmark.add_argument(
        "--cache-key",
        default=None,
        help="explicit cache key; otherwise a deterministic identity is used",
    )
    benchmark.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="local cache directory (default: ~/.apdo/cache)",
    )
    benchmark.add_argument(
        "--sample-count-fn",
        metavar="MODULE:FACTORY",
        default=None,
        help="optional factory returning a callable that counts samples in a batch",
    )
    batch_group = benchmark.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--max-batches",
        type=_at_least_two,
        default=100,
        help="maximum batches per pass, including the first (default: 100)",
    )
    batch_group.add_argument(
        "--full-dataset",
        action="store_true",
        help="iterate every batch in each pass instead of limiting max_batches",
    )
    benchmark.add_argument(
        "--max-workers",
        type=_positive_int,
        default=None,
        help="maximum num_workers value to test (default: min(8, available CPUs))",
    )
    benchmark.add_argument(
        "--timing-profile",
        choices=("standard", "detailed"),
        default="standard",
        help=(
            "standard: whole-cycle CUDA events; detailed: add transfer and per-step "
            "device events without per-batch synchronization (default: standard)"
        ),
    )
    benchmark.add_argument(
        "--selection-metric",
        choices=("sustained", "amortized"),
        default="sustained",
        help=(
            "sustained: execution throughput excluding first loader batch; "
            "amortized: all samples divided by complete trial wall (default: sustained)"
        ),
    )
    benchmark.add_argument(
        "--comparability-policy",
        choices=("off", "warn", "strict"),
        default="warn",
        help=(
            "off: no comparison diagnostics; warn: report observed mismatches; "
            "strict: require verified sample identity (default: warn)"
        ),
    )
    benchmark.add_argument(
        "--sample-fingerprint",
        metavar="MODULE:FACTORY",
        default=None,
        help=(
            "import path of a factory returning a callable that hashes stable sample "
            "identifiers for batch comparability"
        ),
    )
    benchmark.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the structured JSON report to PATH",
    )
    benchmark.add_argument(
        "--html",
        dest="html_path",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the standalone HTML report to PATH",
    )
    benchmark.set_defaults(handler=_run_benchmark, validation=True)
    stability = subparsers.add_parser(
        "stability",
        help="repeat independent optimizations and summarize recommendation stability",
        description=(
            "Run the same optimization experiment multiple times with independent "
            "seeds, retain partial failures, and report recommendation agreement, "
            "gain uncertainty, throughput spread, and comparability outcomes."
        ),
        epilog=(
            "Examples:\n"
            "  apdo stability --dataset my_project.data:build_loader --runs 5 "
            "--json stability.json --html stability.html\n"
            "  apdo stability --factory my_project.data:make_loader --mode training "
            "--training-step-factory my_project.train:make_step --runs 3"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stability_input = stability.add_mutually_exclusive_group(required=True)
    stability_input.add_argument(
        "--dataset",
        metavar="MODULE:FACTORY",
        help="import path of a zero-argument dataset or DataLoader factory",
    )
    stability_input.add_argument(
        "--factory",
        metavar="MODULE:FACTORY",
        help=(
            "import path of a callable accepting LoaderConfig and returning "
            "an independent DataLoader"
        ),
    )
    stability.add_argument(
        "--mode",
        choices=("loader", "training"),
        default="loader",
        help="benchmark DataLoader only or end-to-end training (default: loader)",
    )
    stability.add_argument("--epochs", type=_positive_int, default=None)
    stability.add_argument("--training-step-factory", metavar="MODULE:FACTORY", default=None)
    stability.add_argument("--training-rounds", type=_at_least_two, default=2)
    stability.add_argument("--training-validation-rounds", type=_at_least_two, default=None)
    stability.add_argument("--max-training-candidates", type=_positive_int, default=3)
    stability.add_argument(
        "--validation-policy", choices=("minimum", "positive"), default="minimum"
    )
    stability.add_argument("--batch-size", type=_positive_int, default=None)
    stability.add_argument("--device", default="auto")
    stability.add_argument("--max-ram-gb", type=_positive_float, default=None)
    stability.add_argument("--warmup-runs", type=_positive_int, default=1)
    stability.add_argument("--rounds", type=_at_least_two, default=3)
    stability.add_argument("--validation-rounds", type=_at_least_two, default=10)
    stability.add_argument("--min-improvement", type=_fraction, default=0.05)
    stability.add_argument("--max-workers", type=_positive_int, default=None)
    stability.add_argument("--max-batches", type=_at_least_two, default=100)
    stability.add_argument("--sample-count-fn", metavar="MODULE:FACTORY", default=None)
    stability.add_argument("--sample-fingerprint", metavar="MODULE:FACTORY", default=None)
    stability.add_argument("--timing-profile", choices=("standard", "detailed"), default="standard")
    stability.add_argument(
        "--selection-metric", choices=("sustained", "amortized"), default="sustained"
    )
    stability.add_argument(
        "--comparability-policy", choices=("off", "warn", "strict"), default="warn"
    )
    stability.add_argument("--time-budget-s", type=_positive_float, default=None)
    stability.add_argument("--search-strategy", choices=("grid", "adaptive"), default="grid")
    stability.add_argument("--planned-epochs", type=_positive_int, default=None)
    stability.add_argument("--planned-sessions", type=_positive_int, default=1)
    stability.add_argument("--cache-policy", choices=("off", "hint", "revalidate"), default="off")
    stability.add_argument("--cache-key", default=None)
    stability.add_argument("--cache-dir", type=Path, default=None, metavar="PATH")
    stability.add_argument(
        "--runs",
        type=_positive_int,
        default=5,
        help="independent optimization runs (default: 5)",
    )
    stability.add_argument(
        "--base-seed",
        type=_nonnegative_int,
        default=42,
        help="first seed; later runs use base_seed + run_index (default: 42)",
    )
    stability.add_argument("--no-validation", dest="validation", action="store_false")
    stability.add_argument("--json", dest="json_path", type=Path, default=None, metavar="PATH")
    stability.add_argument("--html", dest="html_path", type=Path, default=None, metavar="PATH")
    stability.set_defaults(handler=_run_stability, validation=True)
    cache = subparsers.add_parser(
        "cache",
        help="inspect or clear the local recommendation cache",
        description="Inspect or clear the opt-in local recommendation cache.",
    )
    cache_commands = cache.add_subparsers(dest="cache_command", metavar="command", required=True)
    cache_list = cache_commands.add_parser("list", help="list cached recommendations")
    cache_list.add_argument("--directory", type=Path, default=None, metavar="PATH")
    cache_list.set_defaults(handler=_run_cache)
    cache_clear = cache_commands.add_parser("clear", help="delete cached recommendations")
    cache_clear.add_argument("--directory", type=Path, default=None, metavar="PATH")
    cache_clear.add_argument(
        "--yes",
        action="store_true",
        help="confirm deletion of all cached recommendations",
    )
    cache_clear.set_defaults(handler=_run_cache)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("apdo: benchmark interrupted", file=sys.stderr)
        return 130


def _run_benchmark(args: argparse.Namespace) -> int:
    try:
        data = _load_dataset(args.dataset) if args.dataset is not None else None
        dataloader_factory = (
            _load_attribute(args.factory, "loader factory") if args.factory is not None else None
        )
        training_step_factory = (
            _load_attribute(args.training_step_factory, "training step factory")
            if args.training_step_factory is not None
            else None
        )
        sample_count_fn = (
            _load_factory(args.sample_count_fn, "sample-count")
            if args.sample_count_fn is not None
            else None
        )
        sample_fingerprint = (
            _load_factory(args.sample_fingerprint, "sample-fingerprint")
            if args.sample_fingerprint is not None
            else None
        )
    except _CliError as exc:
        print(f"apdo: {exc}", file=sys.stderr)
        return 1

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    try:
        result = optimize(
            data,
            batch_size=args.batch_size,
            device=args.device,
            max_ram_gb=args.max_ram_gb,
            dataloader_factory=dataloader_factory,
            mode=args.mode,
            epochs=args.epochs,
            training_step_factory=training_step_factory,
            training_rounds=args.training_rounds,
            training_validation_rounds=args.training_validation_rounds,
            max_training_candidates=args.max_training_candidates,
            validation_policy=args.validation_policy,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
            rounds=args.rounds,
            max_batches=None if args.full_dataset else args.max_batches,
            max_workers=args.max_workers,
            seed=args.seed,
            validation=args.validation,
            validation_rounds=args.validation_rounds,
            min_improvement=args.min_improvement,
            time_budget_s=args.time_budget_s,
            search_strategy=args.search_strategy,
            planned_epochs=args.planned_epochs,
            planned_sessions=args.planned_sessions,
            cache_policy=args.cache_policy,
            cache_key=args.cache_key,
            cache_directory=args.cache_dir,
            sample_count_fn=sample_count_fn,
            timing_profile=args.timing_profile,
            selection_metric=args.selection_metric,
            comparability_policy=args.comparability_policy,
            sample_fingerprint_fn=sample_fingerprint,
            progress=progress,
        )
    except (ApdoError, OSError) as exc:
        print(f"apdo: {exc}", file=sys.stderr)
        return 1

    print(result.summary())
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    try:
        if args.json_path is not None:
            result.to_json(args.json_path)
            print(f"JSON report written to {args.json_path}")
        if args.html_path is not None:
            result.to_html(args.html_path)
            print(f"HTML report written to {args.html_path}")
    except OSError as exc:
        print(f"apdo: could not write report: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_stability(args: argparse.Namespace) -> int:
    try:
        data = _load_dataset(args.dataset) if args.dataset is not None else None
        dataloader_factory = (
            _load_attribute(args.factory, "loader factory") if args.factory is not None else None
        )
        training_step_factory = (
            _load_attribute(args.training_step_factory, "training step factory")
            if args.training_step_factory is not None
            else None
        )
        sample_count_fn = (
            _load_factory(args.sample_count_fn, "sample-count")
            if args.sample_count_fn is not None
            else None
        )
        sample_fingerprint = (
            _load_factory(args.sample_fingerprint, "sample-fingerprint")
            if args.sample_fingerprint is not None
            else None
        )
    except _CliError as exc:
        print(f"apdo: {exc}", file=sys.stderr)
        return 1

    def progress(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def experiment(seed: int) -> Any:
        return optimize(
            data,
            batch_size=args.batch_size,
            device=args.device,
            max_ram_gb=args.max_ram_gb,
            dataloader_factory=dataloader_factory,
            mode=args.mode,
            epochs=args.epochs,
            training_step_factory=training_step_factory,
            training_rounds=args.training_rounds,
            training_validation_rounds=args.training_validation_rounds,
            max_training_candidates=args.max_training_candidates,
            validation_policy=args.validation_policy,
            warmup_runs=args.warmup_runs,
            repetitions=args.rounds,
            rounds=None,
            max_batches=args.max_batches,
            max_workers=args.max_workers,
            seed=seed,
            validation=args.validation,
            validation_rounds=args.validation_rounds,
            min_improvement=args.min_improvement,
            time_budget_s=args.time_budget_s,
            search_strategy=args.search_strategy,
            planned_epochs=args.planned_epochs,
            planned_sessions=args.planned_sessions,
            cache_policy=args.cache_policy,
            cache_key=args.cache_key,
            cache_directory=args.cache_dir,
            sample_count_fn=sample_count_fn,
            timing_profile=args.timing_profile,
            selection_metric=args.selection_metric,
            comparability_policy=args.comparability_policy,
            sample_fingerprint_fn=sample_fingerprint,
            progress=progress,
        )

    report = evaluate_stability(
        experiment,
        runs=args.runs,
        base_seed=args.base_seed,
        progress=progress,
    )
    print(report.summary_text())
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    try:
        if args.json_path is not None:
            report.to_json(args.json_path)
            print(f"JSON stability report written to {args.json_path}")
        if args.html_path is not None:
            report.to_html(args.html_path)
            print(f"HTML stability report written to {args.html_path}")
    except OSError as exc:
        print(f"apdo: could not write stability report: {exc}", file=sys.stderr)
        return 1
    return 0 if report.summary.runs_succeeded else 1


def _run_cache(args: argparse.Namespace) -> int:
    cache = ConfigCache(args.directory)
    if args.cache_command == "list":
        entries = cache.list_entries()
        if not entries:
            print(f"No cached recommendations in {cache.directory}")
            return 0
        print(f"Cached recommendations in {cache.directory}: {len(entries)}")
        for entry in entries:
            gain = (
                f"{entry.improvement_percent:+.1f}%"
                if entry.improvement_percent is not None
                else "n/a"
            )
            print(
                f"  {entry.created_at}  {entry.key[:12]}  "
                f"{entry.validation_status}  gain={gain}  "
                f"{entry.recommended_config.describe()}"
            )
        return 0
    if not args.yes:
        print(
            "apdo: cache clear requires --yes to delete cached recommendations",
            file=sys.stderr,
        )
        return 2
    try:
        removed = cache.clear()
    except OSError as exc:
        print(f"apdo: could not clear cache: {exc}", file=sys.stderr)
        return 1
    print(f"Removed {removed} cached recommendation(s) from {cache.directory}")
    return 0


def _load_dataset(specification: str) -> Any:
    return _load_factory(specification, "dataset")


def _load_attribute(specification: str, kind: str) -> Any:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise _CliError(f"invalid {kind} specification {specification!r}; expected MODULE:FACTORY")

    current_directory = str(Path.cwd())
    if current_directory not in sys.path:
        sys.path.insert(0, current_directory)

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise _CliError(
            f"could not import {kind} module {module_name!r}: {type(exc).__name__}: {exc}"
        ) from exc

    value = getattr(module, attribute, None)
    if value is None:
        raise _CliError(f"{kind} module {module_name!r} has no attribute {attribute!r}")
    if not callable(value):
        raise _CliError(f"{kind} attribute {module_name}:{attribute} is not callable")
    return value


def _load_factory(specification: str, kind: str) -> Any:
    factory = _load_attribute(specification, kind)
    try:
        return factory()
    except Exception as exc:
        raise _CliError(
            f"{kind} factory {specification!r} failed: {type(exc).__name__}: {exc}"
        ) from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be at least 0")
    return parsed


def _at_least_two(value: str) -> int:
    parsed = _positive_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("value must be at least 2")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if parsed <= 0 or parsed != parsed or parsed == float("inf"):
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _fraction(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number") from exc
    if parsed < 0 or parsed >= 1 or parsed != parsed:
        raise argparse.ArgumentTypeError("value must be a finite fraction from 0 up to 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
