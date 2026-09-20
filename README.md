# DigitPad

[![CI](https://github.com/0xDako/MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/0xDako/MLOps/actions/workflows/ci.yml)

Сервис распознавания рукописных цифр по трём моделям одновременно.

## ML

Три модели, обучаемые на MNIST, с единым ONNX-интерфейсом (вход `(N, 1, 28, 28)` float32,
выход `(N, 10)` — вероятности, нормализация встроена в модель):

| Модель | Архитектура | Данные |
|---|---|---|
| `logreg` | Flatten → Linear(784, 10) | чистый MNIST |
| `cnn_small` | Conv→ReLU→Pool → Conv→ReLU→Pool → FC → FC | чистый MNIST |
| `cnn_robust` | та же архитектура, что `cnn_small` | MNIST + аугментации (поворот ±15°, масштаб, размытие, шум) — устойчивее к «грязным» фото |

Код: `ml/models.py` (архитектуры), `ml/dataset.py` (загрузка MNIST и аугментации),
`ml/train.py` (обучение), `ml/export.py` (экспорт в ONNX + метрики).

```bash
make train    # python ml/train.py --model all
              # обучает все три модели (--model logreg|cnn_small|cnn_robust — только одну,
              # --epochs N — число эпох, --subset — 10% данных для быстрой проверки),
              # сохраняет веса *.pt в models/

make export   # python ml/export.py
              # конвертирует каждую *.pt в ONNX (softmax и нормализация в графе,
              # динамическая ось батча), проверяет паритет PyTorch/ONNX (atol=1e-5)
              # и пишет models/metrics.json (accuracy, размер файла, время инференса)
```

## API

FastAPI-сервис поверх ONNX-моделей из `models/` (PyTorch на сервере не нужен). Перед первым
запуском нужны обученные и экспортированные модели (`make train && make export`).

```bash
make api   # uvicorn api.main:app --reload
           # поднимает сервис на http://localhost:8000
           # Swagger: http://localhost:8000/api/docs
```

Эндпоинты (все под `/api`, лимит файла — 10 МБ):

| Метод и путь | Что делает |
|---|---|
| `GET /api/health` | статус сервиса и список загруженных моделей |
| `GET /api/models` | метрики моделей из `metrics.json` (accuracy, размер, время инференса) |
| `POST /api/predict/digit` | одна цифра (файл в теле, `?model=all\|logreg\|cnn_small\|cnn_robust`) → вероятности по классам |
| `POST /api/predict/photo` | фото с цифрами → рамки, распознанная строка и номер телефона по каждой из трёх моделей |

Пример запроса:

```bash
curl -F "file=@digit.png" "http://localhost:8000/api/predict/digit?model=all"
```

## UI

Streamlit-интерфейс, ходит только в API по `API_URL` (см. `.env.example`), ничего не знает
о моделях напрямую.

```bash
make api   # сначала поднять API — UI без него бесполезен
make ui    # streamlit run ui/app.py → http://localhost:8501
```

Три вкладки:

| Вкладка | Что делает |
|---|---|
| **Нарисовать** | холст (`streamlit-drawable-canvas`) → кнопка «Распознать» → карточки по каждой модели: топ-1 цифра, уверенность, столбики вероятностей |
| **Фото** | загрузка файла или снимок с камеры → фото с рамками найденных цифр, таблица сравнения трёх моделей (строка, номер телефона, средняя уверенность); неуверенные цифры на фото подсвечены красным |
| **Метрики** | таблица моделей из `GET /models` (accuracy, размер, время инференса) |

Если API недоступен — соответствующая вкладка покажет ошибку вместо падения.

## Docker

```bash
make up     # docker compose up --build → api:8000, ui:8501, общая сеть
make down   # docker compose down
```

`api/Dockerfile` и `ui/Dockerfile` ставят только нужные для каждого сервиса зависимости
(PyTorch в образы не попадает — обучение только локально/в CI). Перед первым запуском
нужны обученные модели в `models/*.onnx` (`make train && make export`), они копируются
в API-образ на этапе сборки.

Прод (`docker-compose.prod.yml`) берёт готовые образы из GHCR вместо сборки из исходников.

## CI

На каждый push и PR запускаются три параллельные джобы (`.github/workflows/ci.yml`):

| Джоба | Что делает |
|---|---|
| `lint` | `ruff check` + `ruff format --check` |
| `test` | `pytest` с покрытием (`pytest-cov`), HTML-отчёт публикуется как артефакт сборки |
| `smoke` | `train --subset --epochs 1 → export` (паритет PyTorch/ONNX проверяется внутри `export.py`) → `docker build` обоих образов |

MNIST и pip-кэш переиспользуются между запусками.
