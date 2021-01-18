try:
    from pl_bolts.models.detection import components  # noqa: F401
    from pl_bolts.models.detection.faster_rcnn import FasterRCNN  # noqa: F401
except ModuleNotFoundError:  # pragma: no-cover
    pass  # pragma: no-cover
