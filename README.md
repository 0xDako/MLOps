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
