import os

from tests import PROJECT_ROOT


def test_paths():
    assert os.path.isdir(PROJECT_ROOT)
