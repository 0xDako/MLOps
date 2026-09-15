def test_predict_digit_invalid_file(client):
    files = {"file": ("not-an-image.txt", b"hello world", "text/plain")}
    response = client.post("/predict/digit", files=files)
    assert response.status_code == 400


def test_predict_photo_invalid_file(client):
    files = {"file": ("not-an-image.txt", b"hello world", "text/plain")}
    response = client.post("/predict/photo", files=files)
    assert response.status_code == 400
