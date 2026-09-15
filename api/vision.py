"""Предобработка изображений, сегментация цифр на фото и форматирование номера телефона."""

import re

import cv2
import numpy as np

MODEL_INPUT_SIZE = 28
MAX_DIMENSION = 1000
MIN_CONTOUR_AREA = 40

_PHONE_PATTERN = re.compile(r"^[78](\d{10})$")


def preprocess(image: np.ndarray) -> np.ndarray:
    """Приводит изображение (RGB/RGBA/grayscale, любой размер) к float32 (1, 28, 28) в [0, 1].

    Цифра приводится к белой на чёрном фоне независимо от исходной полярности.
    """
    gray = _to_grayscale(image)
    resized = cv2.resize(
        gray, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA
    )
    normalized = resized.astype(np.float32) / 255.0

    if normalized.mean() > 0.5:
        normalized = 1.0 - normalized

    return normalized[np.newaxis, :, :]


def segment(image: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Находит цифры на фото: список (bbox, crop) по строкам сверху вниз, слева направо."""
    resized = _resize_to_max(image, MAX_DIMENSION)
    gray = _to_grayscale(resized)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < MIN_CONTOUR_AREA:
            continue
        boxes.append((x, y, w, h))

    boxes = _sort_by_rows(boxes)
    return [
        (box, gray[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]) for box in boxes
    ]


def format_phone(digits: str) -> str:
    """`89161234567` → `+7 (916) 123-45-67`, иначе строка возвращается без изменений."""
    match = _PHONE_PATTERN.match(digits)
    if not match:
        return digits
    d = match.group(1)
    return f"+7 ({d[0:3]}) {d[3:6]}-{d[6:8]}-{d[8:10]}"


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _resize_to_max(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_dimension / max(height, width)
    if scale >= 1:
        return image
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _sort_by_rows(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    avg_height = sum(h for _, _, _, h in boxes) / len(boxes)

    def row_index(box: tuple[int, int, int, int]) -> int:
        _, y, _, h = box
        return round((y + h / 2) / avg_height)

    return sorted(boxes, key=lambda box: (row_index(box), box[0]))
