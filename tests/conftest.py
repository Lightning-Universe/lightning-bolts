import pytest


# GitHub Actions use this path to cache datasets.
# Use `data_dir` fixture where possible and use `DATA_DIR` in
# `pytest.mark.parametrize()` where you cannot use `data_dir`.
# https://github.com/pytest-dev/pytest/issues/349
from tests import DATASETS_PATH


@pytest.fixture(scope="session")
def data_dir():
    return DATASETS_PATH
