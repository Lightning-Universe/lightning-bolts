"""
Collection of PyTorchLightning loggers
"""

__all__ = []

try:
    from pl_bolts.loggers.azureml import AzureMlLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('AzureMlLogger')
