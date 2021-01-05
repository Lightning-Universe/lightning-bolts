try:
    from pl_bolts.models.detection import components  # noqa
    from pl_bolts.models.detection.faster_rcnn import FasterRCNN  # noqa
except ModuleNotFoundError:  # pragma: no-cover
    pass  # pragma: no-cover
