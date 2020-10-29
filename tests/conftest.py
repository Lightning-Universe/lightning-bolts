import pytest


# GitHub Actions use this path to cache datasets.
@pytest.fixture(scope="session")
def data_dir():
    # TODO(akihironitta): Use something like pytest-lazy-fixture
    # to use this fixture in `pytest.mark.parametrize`.
    # See https://github.com/pytest-dev/pytest/issues/349.
    return "./datasets"
