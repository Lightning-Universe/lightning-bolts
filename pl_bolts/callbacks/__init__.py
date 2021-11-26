"""Collection of PyTorchLightning callbacks."""
from pl_bolts.callbacks.byol_updates import BYOLMAWeightUpdate
from pl_bolts.callbacks.data_monitor import ModuleDataMonitor, TrainingDataMonitor
from pl_bolts.callbacks.printing import PrintTableMetricsCallback
from pl_bolts.callbacks.sparseml import SparseMLCallback
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pl_bolts.callbacks.torch_ort import ORTCallback
from pl_bolts.callbacks.variational import LatentDimInterpolator
from pl_bolts.callbacks.verification.batch_gradient import BatchGradientVerificationCallback  # type: ignore
from pl_bolts.callbacks.vision.confused_logit import ConfusedLogitCallback
from pl_bolts.callbacks.vision.image_generation import TensorboardGenerativeModelImageSampler
from pl_bolts.callbacks.vision.sr_image_logger import SRImageLoggerCallback

__all__ = [
    "BatchGradientVerificationCallback",
    "BYOLMAWeightUpdate",
    "ModuleDataMonitor",
    "TrainingDataMonitor",
    "PrintTableMetricsCallback",
    "SSLOnlineEvaluator",
    "LatentDimInterpolator",
    "ConfusedLogitCallback",
    "TensorboardGenerativeModelImageSampler",
    "SRImageLoggerCallback",
    "ORTCallback",
    "SparseMLCallback",
]
