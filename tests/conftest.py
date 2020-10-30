import pytest


# GitHub Actions use this path to cache datasets.
DATA_DIR = "./datasets"


@pytest.fixture(scope="session")
def data_dir():
    return DATA_DIR
