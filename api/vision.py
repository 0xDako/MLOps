"""Предобработка изображений, сегментация цифр на фото и форматирование номера телефона."""

import re

import cv2
import numpy as np

MODEL_INPUT_SIZE = 28
MAX_DIMENSION = 1000
MIN_CONTOUR_AREA = 40

_PHONE_PATTERN = re.compile(r"^[78](\d{10})$")


def preprocess(image: np.ndarray, dilate: bool = False) -> np.ndarray:
    """Приводит изображение (RGB/RGBA/grayscale, любой размер) к float32 (1, 28, 28) в [0, 1].

    Цифра приводится к белой на чёрном фоне, обрезается по содержимому и центрируется
    по центру масс — как в оригинальном MNIST, где модель училась на отцентрованных цифрах.
    `dilate=True` утолщает штрихи: тонкие линии на фото не похожи на толстые мазки MNIST.
    """
    gray = _to_grayscale(image)
    normalized = gray.astype(np.float32) / 255.0

    if normalized.mean() > 0.5:
        normalized = 1.0 - normalized

    if dilate:
        normalized = cv2.dilate(normalized, np.ones((3, 3), np.uint8))

    cropped = _crop_and_pad(normalized)
    resized = cv2.resize(
        cropped, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_AREA
    )
    centered = _recenter_by_mass(resized)

    return centered[np.newaxis, :, :].astype(np.float32)


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


def _crop_and_pad(image: np.ndarray, margin_ratio: float = 0.2) -> np.ndarray:
    """Обрезает изображение по содержимому и паддит в квадрат с отступом по краям."""
    ys, xs = np.where(image > 0.05)
    if len(xs) == 0 or len(ys) == 0:
        return image

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    cropped = image[y0 : y1 + 1, x0 : x1 + 1]

    height, width = cropped.shape
    side = max(int(max(height, width) * (1 + margin_ratio)), 1)
    padded = np.zeros((side, side), dtype=image.dtype)
    y_offset = (side - height) // 2
    x_offset = (side - width) // 2
    padded[y_offset : y_offset + height, x_offset : x_offset + width] = cropped
    return padded


def _recenter_by_mass(image: np.ndarray) -> np.ndarray:
    """Сдвигает изображение так, чтобы центр масс совпадал с геометрическим центром."""
    total = image.sum()
    if total == 0:
        return image

    rows, cols = image.shape
    ys, xs = np.indices(image.shape)
    center_y = (ys * image).sum() / total
    center_x = (xs * image).sum() / total

    shift_x = cols / 2.0 - center_x
    shift_y = rows / 2.0 - center_y
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, matrix, (cols, rows))


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
