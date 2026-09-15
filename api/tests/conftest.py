"""Общие фикстуры: маленькие ONNX-заглушки вместо настоящих обученных моделей."""

from pathlib import Path

import pytest
import torch
from api.predict import MODEL_NAMES, ModelRegistry
from fastapi.testclient import TestClient
from torch import nn


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.linear(x.flatten(1)), dim=1)


def _export_tiny_onnx(path: Path) -> None:
    model = _TinyModel()
    model.eval()
    dummy = torch.rand(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )


@pytest.fixture
def models_dir(tmp_path: Path) -> Path:
    for name in MODEL_NAMES:
        _export_tiny_onnx(tmp_path / f"{name}.onnx")
    return tmp_path


@pytest.fixture
def registry(models_dir: Path) -> ModelRegistry:
    return ModelRegistry(models_dir=models_dir)


@pytest.fixture
def client(
    registry: ModelRegistry, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> TestClient:
    import api.main as main_module

    monkeypatch.setattr(main_module, "registry", registry)
    monkeypatch.setattr(main_module, "MODELS_DIR", models_dir)
    return TestClient(main_module.app)
