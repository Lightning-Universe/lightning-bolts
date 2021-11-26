import importlib
import operator
from typing import Callable

import torch
from packaging.version import Version
from pkg_resources import DistributionNotFound
from pytorch_lightning.utilities import _module_available

from pl_bolts.callbacks.verification.batch_gradient import BatchGradientVerification  # type: ignore


# Ported from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/imports.py
def _compare_version(package: str, op: Callable, version: str) -> bool:
    """Compare package version with some requirements.

    >>> _compare_version("torch", operator.ge, "0.1")
    True
    """
    try:
        pkg = importlib.import_module(package)
    except (ModuleNotFoundError, DistributionNotFound):
        return False
    try:
        pkg_version = Version(pkg.__version__)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


_NATIVE_AMP_AVAILABLE: bool = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")

_TORCHVISION_AVAILABLE: bool = _module_available("torchvision")
_GYM_AVAILABLE: bool = _module_available("gym")
_SKLEARN_AVAILABLE: bool = _module_available("sklearn")
_PIL_AVAILABLE: bool = _module_available("PIL")
_OPENCV_AVAILABLE: bool = _module_available("cv2")
_WANDB_AVAILABLE: bool = _module_available("wandb")
_MATPLOTLIB_AVAILABLE: bool = _module_available("matplotlib")
_TORCHVISION_LESS_THAN_0_9_1: bool = _compare_version("torchvision", operator.lt, "0.9.1")
_PL_GREATER_EQUAL_1_4 = _compare_version("pytorch_lightning", operator.ge, "1.4.0")
_PL_GREATER_EQUAL_1_4_5 = _compare_version("pytorch_lightning", operator.ge, "1.4.5")
_TORCH_ORT_AVAILABLE = _module_available("torch_ort")
_TORCH_MAX_VERSION_SPARSEML = _compare_version("torch", operator.lt, "1.10.0")
_SPARSEML_AVAILABLE = _module_available("sparseml") and _PL_GREATER_EQUAL_1_4_5 and _TORCH_MAX_VERSION_SPARSEML

__all__ = ["BatchGradientVerification"]
