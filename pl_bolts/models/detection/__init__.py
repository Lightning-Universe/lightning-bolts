__all__ = []

try:
    from pl_bolts.models.detection.faster_rcnn import FasterRCNN
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('FasterRCNN')
