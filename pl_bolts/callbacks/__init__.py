"""
Collection of PyTorchLightning callbacks
"""
from pl_bolts.callbacks.data_monitor import ModuleDataMonitor, TrainingDataMonitor  # noqa: F401
from pl_bolts.callbacks.printing import PrintTableMetricsCallback  # noqa: F401
from pl_bolts.callbacks.variational import LatentDimInterpolator  # noqa: F401
from pl_bolts.callbacks.vision.confused_logit import ConfusedLogitCallback  # noqa: F401
from pl_bolts.callbacks.vision.image_generation import TensorboardGenerativeModelImageSampler  # noqa: F401
