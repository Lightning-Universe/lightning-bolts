__all__ = []

try:
    from pl_bolts.models.audio.resnet34 import ResNet34
except ModuleNotFoundError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('ResNet34')
