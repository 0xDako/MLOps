import httpx
import pytest
from ui import client


class _FakeResponse:
    def __init__(self, json_data: dict) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_data


def test_health_success(monkeypatch):
    def fake_request(method, url, **kwargs):
        assert method == "GET"
        assert url.endswith("/health")
        return _FakeResponse({"status": "ok", "models": ["logreg"]})

    monkeypatch.setattr(client.httpx, "request", fake_request)
    assert client.health() == {"status": "ok", "models": ["logreg"]}


def test_health_timeout(monkeypatch):
    def fake_request(method, url, **kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr(client.httpx, "request", fake_request)
    with pytest.raises(client.ApiError):
        client.health()


def test_predict_digit_success(monkeypatch):
    def fake_request(method, url, **kwargs):
        assert method == "POST"
        assert url.endswith("/predict/digit")
        assert kwargs["params"] == {"model": "all"}
        return _FakeResponse({"predictions": {}})

    monkeypatch.setattr(client.httpx, "request", fake_request)
    assert client.predict_digit(b"fake-bytes") == {"predictions": {}}
