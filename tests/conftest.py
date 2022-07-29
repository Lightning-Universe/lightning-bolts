import warnings
from pathlib import Path

import pytest

from pl_bolts.utils.stability import UnderReviewWarning

# GitHub Actions use this path to cache datasets.
# Use `datadir` fixture where possible and use `DATASETS_PATH` in
# `pytest.mark.parametrize()` where you cannot use `datadir`.
# https://github.com/pytest-dev/pytest/issues/349
from tests import DATASETS_PATH


@pytest.fixture(scope="session")
def datadir():
    return Path(DATASETS_PATH)


@pytest.fixture
def catch_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", UnderReviewWarning)
        yield
