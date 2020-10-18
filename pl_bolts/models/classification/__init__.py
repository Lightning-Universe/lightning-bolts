__all__ = []

try:
    from pl_bolts.models.classification.cnn import CNN
except ModuleNotFoundError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append("CNN")
