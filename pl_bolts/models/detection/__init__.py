try:
    from pl_bolts.models.detection import components
    from pl_bolts.models.detection.faster_rcnn import FasterRCNN
except ModuleNotFoundError:  # pragma: no-cover
    pass  # pragma: no-cover
