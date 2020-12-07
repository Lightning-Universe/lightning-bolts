import importlib
from unittest import mock

import pytest

from tests import optional_pkg_names


@pytest.mark.parametrize("name", [
    "AsynchronousLoader",
    "BinaryMNISTDataModule",
    "CIFAR10DataModule",
    "TinyCIFAR10DataModule",
    "DiscountedExperienceSource",
    "ExperienceSource",
    "ExperienceSourceDataset",
    "FashionMNISTDataModule",
    "ImagenetDataModule",
    "MNISTDataModule",
    "SklearnDataModule",
    "SklearnDataset",
    "TensorDataset",
    "SSLImagenetDataModule",
    "STL10DataModule",
    "VOCDetectionDataModule",
    "CityscapesDataModule",
    "KittiDataset",
    "KittiDataModule",
])
def test_import(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module = importlib.import_module("pl_bolts.datamodules")
        assert hasattr(module, name), f"`from pl_bolts.datamodules import {name}` failed."
