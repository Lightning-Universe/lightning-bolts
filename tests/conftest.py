import pytest


@pytest.fixture(scope="session")
def tmpdir():
    # GitHub Actions use this path to cache datasets.
    return "./datasets"
