try:
    from pl_bolts.models.detection.faster_rcnn import FasterRCNN
    from pl_bolts.models.detection import components
except ModuleNotFoundError:  # pragma: no-cover
    pass  # pragma: no-cover
