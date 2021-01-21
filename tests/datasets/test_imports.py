import importlib
from unittest import mock

import pytest

from tests import optional_pkg_names


@pytest.mark.parametrize(
    "name", [
        "LightDataset",
        "CIFAR10",
        "TrialCIFAR10",
        "ConcatDataset",
        "DummyDataset",
        "DummyDetectionDataset",
        "RandomDataset",
        "RandomDictDataset",
        "RandomDictStringDataset",
        "extract_archive",
        "parse_devkit_archive",
        "UnlabeledImagenet",
        "KittiDataset",
        "BinaryMNIST",
        "CIFAR10Mixed",
        "SSLDatasetMixin",
    ]
)
def test_import(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.datasets"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."
