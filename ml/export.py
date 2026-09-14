"""Экспорт обученных моделей в ONNX и генерация metrics.json."""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from dataset import get_loader
from models import MODEL_NAMES, get_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def export_to_onnx(model: nn.Module, onnx_path: Path) -> None:
    model.eval()
    dummy_input = torch.rand(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )


def check_parity(model: nn.Module, onnx_path: Path, atol: float = 1e-5) -> None:
    sample = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        torch_out = model(sample).numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_out = session.run(None, {"input": sample.numpy()})[0]

    if not np.allclose(torch_out, onnx_out, atol=atol):
        raise ValueError(f"PyTorch/ONNX parity mismatch for {onnx_path.name}")


def _load_trained(name: str) -> nn.Module:
    model = get_model(name)
    state = torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _accuracy(model: nn.Module, name: str) -> float:
    loader = get_loader(train=False, augment=name == "cnn_robust")
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def _measure_inference_ms(onnx_path: Path, runs: int = 50) -> float:
    session = ort.InferenceSession(str(onnx_path))
    sample = np.random.rand(1, 1, 28, 28).astype(np.float32)
    for _ in range(5):
        session.run(None, {"input": sample})
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"input": sample})
    return (time.perf_counter() - start) / runs * 1000


def export_one(name: str) -> dict:
    model = _load_trained(name)
    onnx_path = MODELS_DIR / f"{name}.onnx"

    export_to_onnx(model, onnx_path)
    check_parity(model, onnx_path)

    return {
        "accuracy": _accuracy(model, name),
        "size_bytes": onnx_path.stat().st_size,
        "inference_ms": _measure_inference_ms(onnx_path),
    }


def main() -> None:
    metrics = {name: export_one(name) for name in MODEL_NAMES}
    for name, values in metrics.items():
        print(f"{name}: {values}")
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
