import pytest
import torch

from pl_bolts.models.self_supervised.resnets import (
    resnet18,
)

@pytest.mark.parametrize("model_class", [
    resnet18
])
def test_resnets(tmpdir, model_class):
    x = torch.rand(1, 3, 32, 32)
    model = model_class()
    model(x)
