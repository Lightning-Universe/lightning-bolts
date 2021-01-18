import torch
from pytorch_lightning.utilities import _module_available

from pl_bolts.callbacks.verification.batch_gradient import BatchGradientVerification  # type: ignore

_NATIVE_AMP_AVAILABLE: bool = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")

_TORCHVISION_AVAILABLE: bool = _module_available("torchvision")
_GYM_AVAILABLE: bool = _module_available("gym")
_SKLEARN_AVAILABLE: bool = _module_available("sklearn")
_PIL_AVAILABLE: bool = _module_available("PIL")
_OPENCV_AVAILABLE: bool = _module_available("cv2")
_WANDB_AVAILABLE: bool = _module_available("wandb")
_MATPLOTLIB_AVAILABLE: bool = _module_available("matplotlib")

__all__ = ["BatchGradientVerification"]
