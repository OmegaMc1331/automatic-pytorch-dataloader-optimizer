from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str, timeout: int = 240) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_path if not existing else source_path + os.pathsep + existing
    return subprocess.run(
        [sys.executable, "-m", "apdo", *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_help() -> None:
    completed = run_cli("--help")

    assert completed.returncode == 0
    assert "benchmark" in completed.stdout
    assert "DataLoader" in completed.stdout


def test_benchmark_help() -> None:
    completed = run_cli("benchmark", "--help")

    assert completed.returncode == 0
    assert "MODULE:FACTORY" in completed.stdout
    assert "--max-batches" in completed.stdout
    assert "--seed" in completed.stdout
    assert "--validation-rounds" in completed.stdout
    assert "--min-improvement" in completed.stdout
    assert "--time-budget-s" in completed.stdout
    assert "--no-validation" in completed.stdout
    assert "--factory" in completed.stdout
    assert "--mode" in completed.stdout
    assert "--epochs" in completed.stdout
    assert "--training-step-factory" in completed.stdout
    assert "--validation-policy" in completed.stdout


def test_version() -> None:
    completed = run_cli("--version")

    assert completed.returncode == 0
    assert "apdo 0.3.0" in completed.stdout


def test_no_command_prints_help() -> None:
    completed = run_cli()

    assert completed.returncode == 0
    assert "usage: apdo" in completed.stdout


def test_benchmark_end_to_end(tmp_path: Path) -> None:
    json_path = tmp_path / "reports" / "benchmark.json"
    html_path = tmp_path / "reports" / "benchmark.html"

    completed = run_cli(
        "benchmark",
        "--dataset",
        "examples.synthetic_dataset:create_dataset",
        "--batch-size",
        "16",
        "--device",
        "cpu",
        "--max-batches",
        "3",
        "--repetitions",
        "2",
        "--max-workers",
        "1",
        "--sample-count-fn",
        "examples.synthetic_dataset:create_sample_counter",
        "--json",
        str(json_path),
        "--html",
        str(html_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert "Recommended DataLoader configuration" in completed.stdout
    assert json_path.is_file()
    assert html_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["recommended"]["config"]["num_workers"] >= 0
    assert payload["settings"]["device"] == "cpu"
    assert payload["loader"]["sample_count_fn"] is True
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_benchmark_accepts_a_dataloader_factory(tmp_path: Path) -> None:
    json_path = tmp_path / "loader.json"

    completed = run_cli(
        "benchmark",
        "--dataset",
        "examples.synthetic_dataset:create_loader",
        "--device",
        "cpu",
        "--max-batches",
        "3",
        "--repetitions",
        "2",
        "--validation-rounds",
        "2",
        "--no-validation",
        "--max-workers",
        "1",
        "--json",
        str(json_path),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["settings"]["input_kind"] == "dataloader"
    assert payload["settings"]["batch_size"] == 16
    assert payload["validation"]["status"] in {"not_needed", "not_possible"}
    assert payload["recommended"]["config"]["num_workers"] == 0


def test_invalid_numeric_options_are_rejected() -> None:
    completed = run_cli(
        "benchmark",
        "--dataset",
        "examples.synthetic_dataset:create_dataset",
        "--validation-rounds",
        "1",
    )

    assert completed.returncode == 2
    assert "at least 2" in completed.stderr


def test_missing_dataset_module_is_reported() -> None:
    completed = run_cli(
        "benchmark",
        "--dataset",
        "module_that_does_not_exist:factory",
        "--device",
        "cpu",
    )

    assert completed.returncode == 1
    assert "could not import dataset module" in completed.stderr


def test_invalid_dataset_specification_is_reported() -> None:
    completed = run_cli("benchmark", "--dataset", "not-a-factory", "--device", "cpu")

    assert completed.returncode == 1
    assert "expected MODULE:FACTORY" in completed.stderr


def test_training_benchmark_with_factory(tmp_path: Path) -> None:
    json_path = tmp_path / "training.json"

    completed = run_cli(
        "benchmark",
        "--factory",
        "examples.synthetic_dataset:create_training_loader",
        "--mode",
        "training",
        "--epochs",
        "1",
        "--training-step-factory",
        "examples.synthetic_dataset:training_step_factory",
        "--training-rounds",
        "2",
        "--device",
        "cpu",
        "--max-batches",
        "2",
        "--repetitions",
        "2",
        "--no-validation",
        "--max-workers",
        "1",
        "--json",
        str(json_path),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["settings"]["input_kind"] == "dataloader_factory"
    assert payload["settings"]["benchmark_mode"] == "training"
    assert payload["settings"]["epochs"] == 1
    assert payload["search"]["throughput_kind"] == "end_to_end_training"
    assert payload["training"]["epochs"] == 1
    assert payload["reproducibility"]["status"] == "factory_managed"
