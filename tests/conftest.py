import pytest


@pytest.fixture(scope="session")
def data_dir():
    # GitHub Actions use this path to cache datasets.
    return "./datasets"
