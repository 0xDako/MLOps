"""FastAPI-сервис: эндпоинты распознавания цифр."""

import json
from pathlib import Path

import cv2
import numpy as np
from api.predict import ModelRegistry, predict_one
from api.vision import format_phone, preprocess, segment
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MAX_FILE_SIZE = 10 * 1024 * 1024


class HealthResponse(BaseModel):
    status: str
    models: list[str]


class ModelMetrics(BaseModel):
    accuracy: float
    size_bytes: int
    inference_ms: float


class DigitPrediction(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]


class PredictDigitResponse(BaseModel):
    predictions: dict[str, DigitPrediction]


class PhotoModelResult(BaseModel):
    digits: list[int]
    confidences: list[float]
    string: str
    phone: str


class PredictPhotoResponse(BaseModel):
    boxes: list[tuple[int, int, int, int]]
    models: dict[str, PhotoModelResult]


app = FastAPI(title="DigitPad API", root_path="/api")
registry = ModelRegistry()


def _load_metrics() -> dict:
    metrics_path = MODELS_DIR / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return {}


def _decode_image(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")
    return data


def _to_prediction(probs: np.ndarray) -> DigitPrediction:
    digit = int(np.argmax(probs))
    return DigitPrediction(
        digit=digit, confidence=float(probs[digit]), probabilities=probs.tolist()
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models=registry.available_models)


@app.get("/models", response_model=dict[str, ModelMetrics])
def models() -> dict:
    return _load_metrics()


@app.post("/predict/digit", response_model=PredictDigitResponse)
async def predict_digit(
    file: UploadFile = File(...), model: str = Query("all")
) -> PredictDigitResponse:
    data = await _read_upload(file)
    image = _decode_image(data)
    array = preprocess(image)

    if model == "all":
        probs_by_model = registry.predict_all(array)
    elif model in registry.sessions:
        probs_by_model = {model: predict_one(registry.sessions[model], array)}
    else:
        raise HTTPException(status_code=422, detail=f"Unknown model: {model}")

    predictions = {
        name: _to_prediction(probs) for name, probs in probs_by_model.items()
    }
    return PredictDigitResponse(predictions=predictions)


@app.post("/predict/photo", response_model=PredictPhotoResponse)
async def predict_photo(file: UploadFile = File(...)) -> PredictPhotoResponse:
    data = await _read_upload(file)
    image = _decode_image(data)

    crops = segment(image)
    boxes = [box for box, _ in crops]
    per_model: dict[str, dict[str, list]] = {
        name: {"digits": [], "confidences": []} for name in registry.available_models
    }

    for _, crop in crops:
        array = preprocess(crop)
        for name, probs in registry.predict_all(array).items():
            digit = int(np.argmax(probs))
            per_model[name]["digits"].append(digit)
            per_model[name]["confidences"].append(float(probs[digit]))

    models_result = {}
    for name, values in per_model.items():
        string_result = "".join(str(d) for d in values["digits"])
        models_result[name] = PhotoModelResult(
            digits=values["digits"],
            confidences=values["confidences"],
            string=string_result,
            phone=format_phone(string_result),
        )

    return PredictPhotoResponse(boxes=boxes, models=models_result)
