import pytest
import torch
from pl_bolts.models.self_supervised.amdim import AMDIMEncoder
from pl_bolts.models.self_supervised.cpc import cpc_resnet50
from pl_bolts.models.self_supervised.resnets import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    resnext50_32x4d,
    resnext101_32x8d,
    wide_resnet50_2,
    wide_resnet101_2,
)
from pl_bolts.utils import _IS_WINDOWS


@pytest.mark.skipif(_IS_WINDOWS, reason="strange MemoryError")  # todo
@torch.no_grad()
def test_cpc_resnet():
    x = torch.rand(3, 3, 64, 64)
    model = cpc_resnet50(x)
    model(x)


@pytest.mark.parametrize(
    "model_class",
    [
        resnet18,
        resnet34,
        resnet50,
        resnet101,
        resnet152,
        resnext50_32x4d,
        pytest.param(resnext101_32x8d, marks=pytest.mark.skipif(_IS_WINDOWS, reason="strange MemoryError")),  # todo
        wide_resnet50_2,
        pytest.param(wide_resnet101_2, marks=pytest.mark.skipif(_IS_WINDOWS, reason="strange MemoryError")),  # todo
    ],
)
@torch.no_grad()
def test_torchvision_resnets(model_class):
    x = torch.rand(3, 3, 64, 64)
    model = model_class(pretrained=False)
    model(x)


@pytest.mark.parametrize(
    "size",
    [
        32,
        pytest.param(64, marks=pytest.mark.skipif(_IS_WINDOWS, reason="failing...")),
        pytest.param(128, marks=pytest.mark.skipif(_IS_WINDOWS, reason="failing...")),
    ],
)
@torch.no_grad()
def test_amdim_encoder(size):
    dummy_batch = torch.zeros((2, 3, size, size))
    model = AMDIMEncoder(dummy_batch, encoder_size=size)
    model.init_weights()
    model(dummy_batch)
