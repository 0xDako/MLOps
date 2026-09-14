import pytest

from export import check_parity, export_to_onnx
from models import MODEL_NAMES, get_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_onnx_parity(tmp_path, name):
    model = get_model(name)
    onnx_path = tmp_path / f"{name}.onnx"
    export_to_onnx(model, onnx_path)
    check_parity(model, onnx_path)
