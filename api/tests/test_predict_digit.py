import cv2
import numpy as np


def _sample_image_bytes() -> bytes:
    image = np.zeros((28, 28), dtype=np.uint8)
    image[10:18, 10:18] = 255
    success, buffer = cv2.imencode(".png", image)
    assert success
    return buffer.tobytes()


def test_predict_digit_all_models(client):
    files = {"file": ("digit.png", _sample_image_bytes(), "image/png")}
    response = client.post("/predict/digit", files=files)
    assert response.status_code == 200

    body = response.json()
    assert set(body["predictions"]) == {"logreg", "cnn_small", "cnn_robust"}
    for prediction in body["predictions"].values():
        assert 0 <= prediction["digit"] <= 9
        assert len(prediction["probabilities"]) == 10
        assert abs(sum(prediction["probabilities"]) - 1.0) < 1e-4


def test_predict_digit_single_model(client):
    files = {"file": ("digit.png", _sample_image_bytes(), "image/png")}
    response = client.post("/predict/digit", files=files, params={"model": "cnn_small"})
    assert response.status_code == 200
    assert set(response.json()["predictions"]) == {"cnn_small"}


def test_predict_digit_unknown_model(client):
    files = {"file": ("digit.png", _sample_image_bytes(), "image/png")}
    response = client.post("/predict/digit", files=files, params={"model": "unknown"})
    assert response.status_code == 422
