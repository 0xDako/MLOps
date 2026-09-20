"""Quality gate: проверяет метрики моделей перед релизом."""

import json
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

THRESHOLDS = {
    "cnn_small": 0.985,
    "cnn_robust": 0.95,
}


def check_metrics(metrics: dict) -> list[str]:
    failed = []
    for name, threshold in THRESHOLDS.items():
        accuracy = metrics[name]["accuracy"]
        print(f"{name}: accuracy={accuracy:.4f} threshold={threshold}")
        if accuracy < threshold:
            failed.append(name)
    return failed


def main() -> None:
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    failed = check_metrics(metrics)
    if failed:
        print(f"Quality gate failed: {failed}")
        sys.exit(1)
    print("Quality gate passed")


if __name__ == "__main__":
    main()
