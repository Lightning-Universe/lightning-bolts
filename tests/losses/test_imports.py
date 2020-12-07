import importlib
from unittest import mock

import pytest

from tests import optional_pkg_names


@pytest.mark.parametrize("name", [
    "dqn_loss",
    "per_dqn_loss",
])
def test_import_rl(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.losses.rl"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."


@pytest.mark.parametrize("name", [
    "nt_xent_loss",
    "CPCTask",
    "AmdimNCELoss",
    "FeatureMapContrastiveTask",
    "tanh_clip",
])
def test_import(name):
    """Tests importing when dependencies are not met.

    Set the following in @pytest.mark.parametrize:
        name: class, function or variable name to test importing
    """
    with mock.patch.dict("sys.modules", {pkg: None for pkg in optional_pkg_names}):
        module_name = "pl_bolts.losses.self_supervised_learning"
        module = importlib.import_module(module_name)
        assert hasattr(module, name), f"`from {module_name} import {name}` failed."
