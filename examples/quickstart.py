from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from apdo import optimize


def main() -> None:
    features = torch.randn(512, 32)
    labels = torch.randint(0, 10, (512,))
    dataset = TensorDataset(features, labels)

    result = optimize(
        dataset,
        batch_size=32,
        device="cpu",
        max_ram_gb=2,
        max_batches=50,
        rounds=3,
        validation_rounds=5,
        max_workers=4,
        seed=42,
    )
    print(result.summary())
    print(f"Validation status: {result.validation.status}")

    optimized_loader = result.build_dataloader()
    features_batch, _ = next(iter(optimized_loader))
    print(f"Optimized batch shape: {tuple(features_batch.shape)}")

    reports = Path("reports")
    result.to_json(reports / "benchmark.json")
    result.to_html(reports / "benchmark.html")
    print(f"JSON report: {reports / 'benchmark.json'}")
    print(f"HTML report: {reports / 'benchmark.html'}")


if __name__ == "__main__":
    main()
