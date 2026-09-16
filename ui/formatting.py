"""Чистые функции форматирования данных для UI (без обращения к Streamlit)."""

import cv2
import numpy as np

LOW_CONFIDENCE_THRESHOLD = 0.5


def build_metrics_rows(metrics: dict) -> list[dict]:
    rows = []
    for name, values in metrics.items():
        rows.append(
            {
                "модель": name,
                "accuracy": values.get("accuracy"),
                "размер, КБ": round(values.get("size_bytes", 0) / 1024, 1),
                "инференс, мс": round(values.get("inference_ms", 0), 3),
            }
        )
    return rows


def build_photo_comparison_rows(photo_result: dict) -> list[dict]:
    rows = []
    for name, values in photo_result.get("models", {}).items():
        confidences = values.get("confidences", [])
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        rows.append(
            {
                "модель": name,
                "строка": values.get("string", ""),
                "телефон": values.get("phone", ""),
                "увер-ть, %": round(avg_confidence * 100, 1),
            }
        )
    return rows


def draw_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    digits: list[int],
    confidences: list[float],
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> np.ndarray:
    """Рисует рамки поверх фото; неуверенные предсказания (< threshold) — красным."""
    annotated = image.copy()
    for (x, y, w, h), digit, confidence in zip(boxes, digits, confidences, strict=True):
        color = (0, 200, 0) if confidence >= threshold else (0, 0, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            annotated,
            str(digit),
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return annotated
