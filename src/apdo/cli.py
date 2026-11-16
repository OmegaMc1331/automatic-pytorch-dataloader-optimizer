from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from apdo._version import __version__
from apdo.errors import ApdoError
from apdo.optimizer import optimize


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
            "Load a map-style dataset or an existing DataLoader from an importable "
            "factory, benchmark DataLoader settings, validate the recommendation, "
            "and print the result."
        ),
        epilog=(
            "Example:\n"
            "  apdo benchmark --dataset my_project.data:build_loader "
            "--device auto --json report.json --html report.html\n\n"
            "The factory must be importable from the current working directory and "
            "must be a zero-argument callable returning a map-style PyTorch dataset "
            "or a torch.utils.data.DataLoader. On Windows, classes must be defined at "
            "module level so worker processes can import them."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark.add_argument(
        "--dataset",
        required=True,
        metavar="MODULE:FACTORY",
        help="import path of a zero-argument dataset or DataLoader factory",
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
        default=5,
        help="independent paired validation rounds (default: 5)",
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
        help="approximate search time budget in seconds, checked at round boundaries",
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
        data = _load_dataset(args.dataset)
        sample_count_fn = (
            _load_factory(args.sample_count_fn, "sample-count")
            if args.sample_count_fn is not None
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
            sample_count_fn=sample_count_fn,
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


def _load_dataset(specification: str) -> Any:
    return _load_factory(specification, "dataset")


def _load_factory(specification: str, kind: str) -> Any:
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

    factory = getattr(module, attribute, None)
    if factory is None:
        raise _CliError(f"{kind} module {module_name!r} has no attribute {attribute!r}")
    if not callable(factory):
        raise _CliError(f"{kind} attribute {module_name}:{attribute} is not callable")

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
