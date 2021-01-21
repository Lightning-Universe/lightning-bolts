import importlib
from unittest import mock

import pytest

from tests import optional_pkg_names


@pytest.mark.parametrize(
    "name", [
        "AsynchronousLoader",
        "BinaryMNISTDataModule",
        "CIFAR10DataModule",
        "TinyCIFAR10DataModule",
        "CityscapesDataModule",
        "DiscountedExperienceSource",
        "ExperienceSource",
        "ExperienceSourceDataset",
        "FashionMNISTDataModule",
        "ImagenetDataModule",
        "KittiDataModule",
        "MNISTDataModule",
        "SklearnDataModule",
        "SklearnDataset",
        "TensorDataset",
        "SSLImagenetDataModule",
        "STL10DataModule",
        "VOCDetectionDataModule",
        "KittiDataset",
    ]
)
def test_import(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.datamodules"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."
