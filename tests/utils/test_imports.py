import importlib
from unittest import mock

import pytest

from tests import optional_pkg_names


@pytest.mark.parametrize("name", [
    "LitArg",
    "LightningArgumentParser",
    "gather_lit_args",
])
def test_import_arguments(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.utils.arguments"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


@pytest.mark.parametrize("name", [
    "urls",
    "vae_imagenet2012",
    "cpcv2_resnet18",
    "load_pretrained",
])
def test_import_pretrained_weights(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.utils.pretrained_weights"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


@pytest.mark.parametrize("name", [
    "torchvision_ssl_encoder",
])
def test_import_self_supervised(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.utils.self_supervised"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


@pytest.mark.parametrize("name", [
    "Identity",
    "balance_classes",
    "generate_half_labeled_batches",
])
def test_import_semi_supervised(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.utils.semi_supervised"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


@pytest.mark.parametrize("name", [
    "tile",
])
def test_import_shaping(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.utils.shaping"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


@pytest.mark.parametrize("name", [
    "warn_missing_pkg",
])
def test_import_warnings(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.utils.warnings"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


