"""Загрузка ONNX-моделей и инференс."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_NAMES = ["logreg", "cnn_small", "cnn_robust"]


class ModelRegistry:
    """Загружает все доступные ONNX-модели при старте и кэширует сессии инференса."""

    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.sessions: dict[str, ort.InferenceSession] = {}
        for name in MODEL_NAMES:
            onnx_path = models_dir / f"{name}.onnx"
            if onnx_path.exists():
                self.sessions[name] = ort.InferenceSession(str(onnx_path))

    @property
    def available_models(self) -> list[str]:
        return list(self.sessions)

    def predict_all(self, image_array: np.ndarray) -> dict[str, np.ndarray]:
        return {
            name: predict_one(session, image_array)
            for name, session in self.sessions.items()
        }


def predict_one(session: ort.InferenceSession, image_array: np.ndarray) -> np.ndarray:
    """Прогоняет одно изображение (1, 28, 28) или батч (N, 1, 28, 28) через сессию."""
    batch = image_array.astype(np.float32)
    if batch.ndim == 3:
        batch = batch[np.newaxis, ...]
    output = session.run(None, {"input": batch})[0]
    return output[0]
