import cv2
import numpy as np


def _synthetic_photo_bytes(n_boxes: int = 3) -> bytes:
    image = np.full((100, 300, 3), 255, dtype=np.uint8)
    for i in range(n_boxes):
        x = 20 + i * 90
        cv2.rectangle(image, (x, 30), (x + 40, 70), (0, 0, 0), thickness=-1)
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_predict_photo_returns_boxes_for_all_models(client):
    files = {"file": ("photo.png", _synthetic_photo_bytes(), "image/png")}
    response = client.post("/predict/photo", files=files)
    assert response.status_code == 200

    body = response.json()
    assert len(body["boxes"]) == 3
    assert set(body["models"]) == {"logreg", "cnn_small", "cnn_robust"}
    for result in body["models"].values():
        assert len(result["digits"]) == 3
        assert len(result["confidences"]) == 3
        assert len(result["string"]) == 3
