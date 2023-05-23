from warnings import warn

import pytest
import torch
from pytorch_lightning import seed_everything

try:
    from torchvision import transforms
except ImportError:
    warn(  # pragma: no-cover
        "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
    )

from pl_bolts.transforms.self_supervised.amdim_transforms import (
    AMDIMEvalTransformsCIFAR10,
    AMDIMEvalTransformsImageNet128,
    AMDIMEvalTransformsSTL10,
    AMDIMTrainTransformsCIFAR10,
    AMDIMTrainTransformsImageNet128,
    AMDIMTrainTransformsSTL10,
)
from pl_bolts.transforms.self_supervised.cpc_transforms import (
    CPCEvalTransformsCIFAR10,
    CPCEvalTransformsImageNet128,
    CPCEvalTransformsSTL10,
    CPCTrainTransformsCIFAR10,
    CPCTrainTransformsImageNet128,
    CPCTrainTransformsSTL10,
)
from pl_bolts.transforms.self_supervised.moco_transforms import (
    MoCo2EvalCIFAR10Transforms,
    MoCo2EvalImagenetTransforms,
    MoCo2EvalSTL10Transforms,
    MoCo2TrainCIFAR10Transforms,
    MoCo2TrainImagenetTransforms,
    MoCo2TrainSTL10Transforms,
)
from pl_bolts.transforms.self_supervised.simclr_transforms import SimCLREvalDataTransform, SimCLRTrainDataTransform


@pytest.mark.parametrize(
    "img_size",
    [
        (3, 32, 32),
        (3, 96, 96),
        (3, 160, 160),
    ],
)
def test_simclr_transforms(img_size):
    seed_everything(0)

    (c, h, w) = img_size
    x = torch.rand(c, h, w)
    x = transforms.ToPILImage(mode="RGB")(x)

    transform = SimCLREvalDataTransform(input_height=h)
    transform(x)

    transform = SimCLRTrainDataTransform(input_height=h)
    transform(x)


@pytest.mark.parametrize(
    "transform",
    [
        CPCTrainTransformsCIFAR10,
        CPCEvalTransformsCIFAR10,
        AMDIMEvalTransformsCIFAR10,
        AMDIMTrainTransformsCIFAR10,
        MoCo2TrainCIFAR10Transforms,
        MoCo2EvalCIFAR10Transforms,
    ],
)
def test_cifar10_transforms(transform):
    x = torch.rand(3, 32, 32)
    x = transforms.ToPILImage(mode="RGB")(x)

    transform = transform()
    transform(x)


@pytest.mark.parametrize(
    "transform",
    [
        CPCTrainTransformsSTL10,
        CPCEvalTransformsSTL10,
        AMDIMTrainTransformsSTL10,
        AMDIMEvalTransformsSTL10,
        MoCo2TrainSTL10Transforms,
        MoCo2EvalSTL10Transforms,
    ],
)
def test_stl10_transforms(transform):
    x = torch.rand(3, 64, 64)
    x = transforms.ToPILImage(mode="RGB")(x)

    transform = transform()
    transform(x)


@pytest.mark.parametrize(
    "transform",
    [
        CPCTrainTransformsImageNet128,
        CPCEvalTransformsImageNet128,
        AMDIMTrainTransformsImageNet128,
        AMDIMEvalTransformsImageNet128,
        MoCo2TrainImagenetTransforms,
        MoCo2EvalImagenetTransforms,
    ],
)
def test_imagenet_transforms(transform):
    x = torch.rand(3, 128, 128)
    x = transforms.ToPILImage(mode="RGB")(x)

    transform = transform()
    transform(x)
