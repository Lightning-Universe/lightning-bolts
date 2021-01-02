from pathlib import Path

import pytest

# GitHub Actions use this path to cache datasets.
# Use `datadir` fixture where possible and use `DATASETS_PATH` in
# `pytest.mark.parametrize()` where you cannot use `datadir`.
# https://github.com/pytest-dev/pytest/issues/349
from tests import DATASETS_PATH


@pytest.fixture(scope="session")
def datadir():
    return Path(DATASETS_PATH)
