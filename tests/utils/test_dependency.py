import pytest
from pl_bolts.utils._dependency import requires


@requires("torch")
def using_torch():
    return True


@requires("torch.anything.wrong")
def using_torch_wrong_path():
    return True


@requires("torch>99.0")
def using_torch_bad_version():
    return True


def test_requires_pass():
    assert using_torch() is True


def test_requires_fail():
    with pytest.raises(ModuleNotFoundError, match="Required dependencies not available"):
        assert using_torch_wrong_path()
    with pytest.raises(ModuleNotFoundError, match="Required dependencies not available"):
        assert using_torch_bad_version()
