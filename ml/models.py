"""Три архитектуры моделей для распознавания цифр MNIST."""

import torch
from torch import nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

MODEL_NAMES = ["logreg", "cnn_small", "cnn_robust"]


class LogReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - MNIST_MEAN) / MNIST_STD
        return torch.softmax(self.net(x), dim=1)


class CnnSmall(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - MNIST_MEAN) / MNIST_STD
        x = self.features(x)
        return torch.softmax(self.classifier(x), dim=1)


class CnnRobust(CnnSmall):
    """Та же архитектура, что CnnSmall — отличие только в данных при обучении."""


_MODELS = {
    "logreg": LogReg,
    "cnn_small": CnnSmall,
    "cnn_robust": CnnRobust,
}


def get_model(name: str) -> nn.Module:
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name!r}. Available: {MODEL_NAMES}")
    return _MODELS[name]()
