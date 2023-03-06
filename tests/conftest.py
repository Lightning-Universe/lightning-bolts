import os
import signal
import warnings
from pathlib import Path

import pytest
import torch
from pytorch_lightning.trainer.connectors.signal_connector import SignalConnector
from lightning_fabric.utilities.imports import _IS_WINDOWS

from pl_bolts.utils import _TORCHVISION_AVAILABLE, _TORCHVISION_LESS_THAN_0_13
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
        if _TORCHVISION_AVAILABLE and _TORCHVISION_LESS_THAN_0_13:
            warnings.filterwarnings("ignore", "FLIP_LEFT_RIGHT is deprecated", DeprecationWarning)
        yield


@pytest.fixture(scope="function", autouse=True)
def restore_env_variables():
    """Ensures that environment variables set during the test do not leak out."""
    env_backup = os.environ.copy()
    yield
    leaked_vars = os.environ.keys() - env_backup.keys()
    # restore environment as it was before running the test
    os.environ.clear()
    os.environ.update(env_backup)
    # these are currently known leakers - ideally these would not be allowed
    allowlist = {
        "CUBLAS_WORKSPACE_CONFIG",  # enabled with deterministic flag
        "CUDA_DEVICE_ORDER",
        "LOCAL_RANK",
        "NODE_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "PL_GLOBAL_SEED",
        "PL_SEED_WORKERS",
        "WANDB_MODE",
        "WANDB_REQUIRE_SERVICE",
        "WANDB_SERVICE",
        "HOROVOD_FUSION_THRESHOLD",
        "RANK",  # set by DeepSpeed
        "POPLAR_ENGINE_OPTIONS",  # set by IPUStrategy
        # set by XLA
        "TF2_BEHAVIOR",
        "XRT_MESH_SERVICE_ADDRESS",
        "XRT_TORCH_DIST_ROOT",
        "XRT_MULTI_PROCESSING_DEVICE",
        "XRT_SHARD_WORLD_SIZE",
        "XRT_LOCAL_WORKER",
        "XRT_HOST_WORLD_SIZE",
        "XRT_SHARD_ORDINAL",
        "XRT_SHARD_LOCAL_ORDINAL",
        "TF_CPP_MIN_LOG_LEVEL",
    }
    leaked_vars.difference_update(allowlist)
    assert not leaked_vars, f"test is leaking environment variable(s): {set(leaked_vars)}"


@pytest.fixture(scope="function", autouse=True)
def restore_signal_handlers():
    """Ensures that signal handlers get restored before the next test runs.

    This is a safety net for tests that don't run Trainer's teardown.
    """
    valid_signals = SignalConnector._valid_signals()
    if not _IS_WINDOWS:
        # SIGKILL and SIGSTOP are not allowed to be modified by the user
        valid_signals -= {signal.SIGKILL, signal.SIGSTOP}
    handlers = {signum: signal.getsignal(signum) for signum in valid_signals}
    yield
    for signum, handler in handlers.items():
        if handler is not None:
            signal.signal(signum, handler)


@pytest.fixture(scope="function", autouse=True)
def teardown_process_group():
    """Ensures that the distributed process group gets closed before the next test runs."""
    yield
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="function", autouse=True)
def reset_deterministic_algorithm():
    """Ensures that torch determinism settings are reset before the next test runs."""
    yield
    torch.use_deterministic_algorithms(False)
