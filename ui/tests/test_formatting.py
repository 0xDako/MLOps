import numpy as np
from ui.formatting import build_metrics_rows, build_photo_comparison_rows, draw_boxes


def test_build_metrics_rows():
    metrics = {"logreg": {"accuracy": 0.91, "size_bytes": 32768, "inference_ms": 0.05}}
    rows = build_metrics_rows(metrics)
    assert rows == [
        {"модель": "logreg", "accuracy": 0.91, "размер, КБ": 32.0, "инференс, мс": 0.05}
    ]


def test_build_photo_comparison_rows():
    photo_result = {
        "models": {
            "logreg": {"string": "123", "phone": "123", "confidences": [0.9, 0.8, 0.7]},
        }
    }
    rows = build_photo_comparison_rows(photo_result)
    assert rows[0]["модель"] == "logreg"
    assert rows[0]["строка"] == "123"
    assert rows[0]["увер-ть, %"] == 80.0


def test_draw_boxes_changes_pixels():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    annotated = draw_boxes(image, [(5, 5, 20, 20)], [3], [0.9])
    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)


def test_draw_boxes_low_confidence_is_red():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    annotated = draw_boxes(image, [(5, 5, 20, 20)], [3], [0.1], threshold=0.5)
    assert annotated[5, 5:25, 2].max() > 0
