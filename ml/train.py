"""Обучение моделей: python ml/train.py --model all|logreg|cnn_small|cnn_robust."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from dataset import get_loader
from models import MODEL_NAMES, get_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_EPOCHS = 5
LEARNING_RATE = 1e-3


def train_one(name: str, epochs: int, subset: bool) -> None:
    augment = name == "cnn_robust"
    train_loader = get_loader(train=True, augment=augment, subset=subset)

    model = get_model(name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for _ in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            probs = model(images)
            loss = F.nll_loss(torch.log(probs.clamp_min(1e-12)), labels)
            loss.backward()
            optimizer.step()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"{name}: saved to {weights_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[*MODEL_NAMES, "all"], default="all")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--subset", action="store_true", help="10% данных для быстрого smoke-теста"
    )
    args = parser.parse_args()

    names = MODEL_NAMES if args.model == "all" else [args.model]
    for name in names:
        train_one(name, epochs=args.epochs, subset=args.subset)


if __name__ == "__main__":
    main()
