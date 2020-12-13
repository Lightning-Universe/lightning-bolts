import torch
from pytorch_lightning.utilities import _module_available

_NATIVE_AMP_AVAILABLE = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")

_TORCHVISION_AVAILABLE = _module_available("torchvision")
_GYM_AVAILABLE = _module_available("gym")
_SKLEARN_AVAILABLE = _module_available("sklearn")
_PIL_AVAILABLE = _module_available("PIL")
_OPENCV_AVAILABLE = _module_available("cv2")