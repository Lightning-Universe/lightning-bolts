import importlib
from unittest import mock

import pytest


@pytest.mark.parametrize(
    "dm_cls,deps", [
        ("AsynchronousLoader", []),
        ("BinaryMNISTDataModule", ["torchvision"]),
        ("CIFAR10DataModule", ["torchvision"]),
        ("TinyCIFAR10DataModule", ["torchvision"]),
        ("DiscountedExperienceSource", ["gym"]),
        ("ExperienceSource", ["gym"]),
        ("ExperienceSourceDataset", ["gym"]),
        ("FashionMNISTDataModule", ["torchvision"]),
        ("ImagenetDataModule", ["torchvision"]),
        ("MNISTDataModule", ["torchvision"]),
        ("SklearnDataModule", ["sklearn"]),
        ("SklearnDataset", []),
        ("TensorDataset", []),
        ("SSLImagenetDataModule", ["torchvision"]),
        ("STL10DataModule", ["torchvision"]),
        ("VOCDetectionDataModule", ["torchvision"]),
        ("CityscapesDataModule", ["torchvision"]),
        ("KittiDataset", ["PIL"]),
        ("KittiDataModule", ["torchvision"]),
    ]
)
def test_import(dm_cls, deps):
    """Tests importing when dependencies are not met.

    Set the followings in @pytest.mark.parametrize:
        dm_cls: class to test importing
        deps: packages required for dm_cls
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in deps}):
        dms_module = importlib.import_module("pl_bolts.datamodules")
        assert hasattr(dms_module, dm_cls), f"`from pl_bolts.datamodules import {dm_cls}` failed."
