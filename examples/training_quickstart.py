from __future__ import annotations

from pathlib import Path

from apdo import optimize

try:
    from examples.synthetic_dataset import create_training_loader, training_step_factory
except ModuleNotFoundError:
    from synthetic_dataset import create_training_loader, training_step_factory


def main() -> None:
    result = optimize(
        dataloader_factory=create_training_loader,
        mode="training",
        device="cpu",
        epochs=3,
        training_step_factory=training_step_factory,
        max_batches=8,
        rounds=2,
        training_rounds=2,
        validation_rounds=10,
        max_workers=2,
        seed=42,
    )
    print(result.summary())
    print(f"Training validation: {result.validation.status}")
    reports = Path("reports")
    result.to_json(reports / "training_benchmark.json")
    result.to_html(reports / "training_benchmark.html")


if __name__ == "__main__":
    main()
