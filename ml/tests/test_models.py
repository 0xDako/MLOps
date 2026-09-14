import pytest
import torch

from models import MODEL_NAMES, get_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_output_shape_and_probabilities(name):
    model = get_model(name)
    model.eval()
    x = torch.rand(4, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 10)
    assert torch.allclose(out.sum(dim=1), torch.ones(4), atol=1e-5)


def test_get_model_unknown_name():
    with pytest.raises(ValueError):
        get_model("unknown")
