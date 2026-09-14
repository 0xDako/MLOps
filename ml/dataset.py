"""Загрузка MNIST и аугментации для обучения."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _add_noise(tensor: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    return (tensor + torch.randn_like(tensor) * std).clamp(0.0, 1.0)


BASE_TRANSFORM = transforms.ToTensor()

ROBUST_TRANSFORM = transforms.Compose(
    [
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.2)),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Lambda(_add_noise),
    ]
)


def get_dataset(train: bool, augment: bool = False, subset: bool = False) -> Dataset:
    transform = ROBUST_TRANSFORM if augment else BASE_TRANSFORM
    dataset = datasets.MNIST(
        root=DATA_DIR, train=train, download=True, transform=transform
    )
    if subset:
        size = max(1, len(dataset) // 10)
        dataset = Subset(dataset, range(size))
    return dataset


def get_loader(
    train: bool, augment: bool = False, subset: bool = False, batch_size: int = 128
) -> DataLoader:
    dataset = get_dataset(train=train, augment=augment, subset=subset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
