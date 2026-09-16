import cv2
import numpy as np
import pytest
from api.vision import format_phone, preprocess, segment


@pytest.mark.parametrize(
    "shape",
    [
        (40, 40),  # grayscale (L)
        (40, 40, 3),  # RGB
        (40, 40, 4),  # RGBA
        (100, 60, 3),  # другой размер, RGB
    ],
)
def test_preprocess_shapes(shape):
    image = np.random.randint(0, 256, size=shape, dtype=np.uint8)
    result = preprocess(image)
    assert result.shape == (1, 28, 28)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


@pytest.mark.parametrize(
    ("digits", "expected"),
    [
        ("89161234567", "+7 (916) 123-45-67"),
        ("79161234567", "+7 (916) 123-45-67"),
        ("12345", "12345"),
        ("", ""),
        ("891612345678", "891612345678"),
    ],
)
def test_format_phone(digits, expected):
    assert format_phone(digits) == expected


def test_preprocess_centers_by_mass():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[10:30, 10:30] = 255  # блок сильно смещён в угол

    result = preprocess(image)[0]
    ys, xs = np.where(result > 0.1)
    assert 12 <= ys.mean() <= 16
    assert 12 <= xs.mean() <= 16


def test_preprocess_dilate_thickens_strokes():
    image = np.zeros((40, 40), dtype=np.uint8)
    image[:, 20] = 255  # тонкая линия в один пиксель

    thin = preprocess(image, dilate=False)
    thick = preprocess(image, dilate=True)
    assert thick.sum() > thin.sum()


def test_segment_finds_three_digits_left_to_right():
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    positions = [20, 120, 220]
    for x in positions:
        cv2.rectangle(image, (x, 30), (x + 40, 70), (0, 0, 0), thickness=-1)

    boxes_and_crops = segment(image)
    assert len(boxes_and_crops) == 3

    xs = [box[0] for box, _ in boxes_and_crops]
    assert xs == sorted(xs)
