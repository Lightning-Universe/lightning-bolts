import torch
from torchvision import transforms
import pytorch_lightning as pl
import pytest
from pl_bolts.models.self_supervised.cpc.transforms import (
    CPCTrainTransformsCIFAR10,
    CPCEvalTransformsCIFAR10,
    CPCTrainTransformsSTL10,
    CPCEvalTransformsSTL10,
    CPCTrainTransformsImageNet128,
    CPCEvalTransformsImageNet128
)
from pl_bolts.models.self_supervised.amdim.transforms import (
    AMDIMEvalTransformsCIFAR10,
    AMDIMTrainTransformsCIFAR10,
    AMDIMTrainTransformsSTL10,
    AMDIMEvalTransformsSTL10,
    AMDIMTrainTransformsImageNet128,
    AMDIMEvalTransformsImageNet128
)
from pl_bolts.models.self_supervised.moco.transforms import (
    Moco2TrainCIFAR10Transforms,
    Moco2EvalCIFAR10Transforms,
    Moco2TrainSTL10Transforms,
    Moco2EvalSTL10Transforms,
    Moco2TrainImagenetTransforms,
    Moco2EvalImagenetTransforms
)
from pl_bolts.models.self_supervised.simclr.simclr_transforms import (
    SimCLREvalDataTransform,
    SimCLRTrainDataTransform
)


@pytest.mark.parametrize("img_size", [
    (3, 32, 32),
    (3, 96, 96),
    (3, 160, 160),
])
def test_simclr_transforms(tmpdir, img_size):
    pl.seed_everything(0)

    (c, h, w) = img_size
    x = torch.rand(c, h, w)
    x = transforms.ToPILImage(mode='RGB')(x)

    transform = SimCLREvalDataTransform(input_height=h)
    transform(x)

    transform = SimCLRTrainDataTransform(input_height=h)
    transform(x)


@pytest.mark.parametrize("transform", [
    CPCTrainTransformsCIFAR10,
    CPCEvalTransformsCIFAR10,
    AMDIMEvalTransformsCIFAR10,
    AMDIMTrainTransformsCIFAR10,
    Moco2TrainCIFAR10Transforms,
    Moco2EvalCIFAR10Transforms,
])
def test_cifar10_transforms(tmpdir, transform):
    x = torch.rand(3, 32, 32)
    x = transforms.ToPILImage(mode='RGB')(x)

    transform = transform()
    transform(x)


@pytest.mark.parametrize("transform", [
    CPCTrainTransformsSTL10,
    CPCEvalTransformsSTL10,
    AMDIMTrainTransformsSTL10,
    AMDIMEvalTransformsSTL10,
    Moco2TrainSTL10Transforms,
    Moco2EvalSTL10Transforms,
])
def test_stl10_transforms(tmpdir, transform):
    x = torch.rand(3, 64, 64)
    x = transforms.ToPILImage(mode='RGB')(x)

    transform = transform()
    transform(x)


@pytest.mark.parametrize("transform", [
    CPCTrainTransformsImageNet128,
    CPCEvalTransformsImageNet128,
    AMDIMTrainTransformsImageNet128,
    AMDIMEvalTransformsImageNet128,
    Moco2TrainImagenetTransforms,
    Moco2EvalImagenetTransforms
])
def test_imagenet_transforms(tmpdir, transform):
    x = torch.rand(3, 128, 128)
    x = transforms.ToPILImage(mode='RGB')(x)

    transform = transform()
    transform(x)
