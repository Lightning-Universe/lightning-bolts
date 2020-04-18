import os

from tests import PACKAGE_ROOT


def test_paths():
    assert os.path.isdir(PACKAGE_ROOT)
