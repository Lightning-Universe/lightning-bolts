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


@torch.no_grad()
def test_cpc_resnet():
    x = torch.rand(3, 3, 64, 64)
    model = cpc_resnet50(x)
    model(x)


@pytest.mark.parametrize(
    "model_class", [
        resnet18(pretrained=False),
        resnet34(pretrained=False),
        resnet50(pretrained=False),
        resnet101(pretrained=False),
        resnet152(pretrained=False),
        resnext50_32x4d(pretrained=False),
        resnext101_32x8d(pretrained=False),
        wide_resnet50_2(pretrained=False),
        wide_resnet101_2(pretrained=False),
    ]
)
@torch.no_grad()
def test_torchvision_resnets(model_class):
    x = torch.rand(3, 3, 64, 64)
    model = model_class()
    model(x)


@torch.no_grad()
@pytest.mark.parametrize("size", [32, 64, 128])
def test_amdim_encoder(size):
    dummy_batch = torch.zeros((2, 3, size, size))
    model = AMDIMEncoder(dummy_batch, encoder_size=size)
    model.init_weights()
    model(dummy_batch)
