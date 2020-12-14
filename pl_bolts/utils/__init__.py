from importlib.util import find_spec

import torch
from pytorch_lightning.utilities import _module_available

_NATIVE_AMP_AVAILABLE = _module_available("torch.cuda.amp") and hasattr(torch.cuda.amp, "autocast")

_TORCHVISION_AVAILABLE = find_spec("torchvision") is not None
_GYM_AVAILABLE = find_spec("gym") is not None
_SKLEARN_AVAILABLE = find_spec("sklearn") is not None
_PIL_AVAILABLE = find_spec("PIL") is not None
_OPENCV_AVAILABLE = find_spec("cv2") is not None
