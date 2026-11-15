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


def test_version() -> None:
    completed = run_cli("--version")

    assert completed.returncode == 0
    assert "apdo 0.1.0" in completed.stdout


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
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


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
