import importlib
from unittest import mock

import pytest


@pytest.mark.parametrize("dm_cls,deps", [
    ("SklearnDataModule", ["sklearn"]),
])
def test_import(dm_cls, deps):
    with mock.patch.dict("sys.modules", {pkg: None for pkg in deps}):
        dms_module = importlib.import_module("pl_bolts.datamodules")
        assert hasattr(dms_module, dm_cls)
