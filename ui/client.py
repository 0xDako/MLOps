"""HTTP-клиент для обращения к DigitPad API."""

import os

import httpx

API_URL = os.environ.get("API_URL", "http://localhost:8000/api")
TIMEOUT = 10.0


class ApiError(Exception):
    """API недоступен или вернул ошибку."""


def _request(method: str, path: str, **kwargs) -> dict:
    try:
        response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        raise ApiError(str(exc)) from exc


def health() -> dict:
    return _request("GET", "/health")


def get_models() -> dict:
    return _request("GET", "/models")


def predict_digit(image_bytes: bytes, model: str = "all") -> dict:
    files = {"file": ("digit.png", image_bytes, "image/png")}
    return _request("POST", "/predict/digit", files=files, params={"model": model})


def predict_photo(image_bytes: bytes) -> dict:
    files = {"file": ("photo.png", image_bytes, "image/png")}
    return _request("POST", "/predict/photo", files=files)
