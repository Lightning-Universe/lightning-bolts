import operator
import platform

import torch
from lightning_utilities.core.imports import compare_version, module_available

from pl_bolts.callbacks.verification.batch_gradient import BatchGradientVerification  # type: ignore

_NATIVE_AMP_AVAILABLE: bool = module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")
_IS_WINDOWS = platform.system() == "Windows"
_TORCH_ORT_AVAILABLE = module_available("torch_ort")
_TORCH_MESHGRID_REQUIRES_INDEXING = compare_version("torch", operator.ge, "1.10.0")
_TORCHVISION_AVAILABLE: bool = module_available("torchvision")
_TORCHVISION_LESS_THAN_0_9_1: bool = compare_version("torchvision", operator.lt, "0.9.1")
_TORCHVISION_LESS_THAN_0_13: bool = compare_version("torchvision", operator.le, "0.13.0")
_TORCHMETRICS_DETECTION_AVAILABLE: bool = module_available("torchmetrics.detection")
_PL_GREATER_EQUAL_1_4 = compare_version("pytorch_lightning", operator.ge, "1.4.0")
_PL_GREATER_EQUAL_1_4_5 = compare_version("pytorch_lightning", operator.ge, "1.4.5")
_GYM_AVAILABLE: bool = module_available("gym")
_GYM_GREATER_EQUAL_0_20 = compare_version("gym", operator.ge, "0.20.0")
_SKLEARN_AVAILABLE: bool = module_available("sklearn")
_PIL_AVAILABLE: bool = module_available("PIL")
_SPARSEML_AVAILABLE = module_available("sparseml")
_OPENCV_AVAILABLE: bool = module_available("cv2")
_WANDB_AVAILABLE: bool = module_available("wandb")
_MATPLOTLIB_AVAILABLE: bool = module_available("matplotlib")
_JSONARGPARSE_GREATER_THAN_4_16_0 = compare_version("jsonargparse", operator.gt, "4.16.0")

_SPARSEML_TORCH_SATISFIED = _SPARSEML_AVAILABLE
_SPARSEML_TORCH_SATISFIED_ERROR = None
if _SPARSEML_AVAILABLE:
    try:
        from sparseml.pytorch.base import _TORCH_MAX_VERSION  # noqa: F401

    except (ImportError, ValueError) as err:
        _SPARSEML_TORCH_SATISFIED = False
        _SPARSEML_TORCH_SATISFIED_ERROR = str(err)

__all__ = ["BatchGradientVerification"]
