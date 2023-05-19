from torch import Tensor
from torchvision.ops import box_iou, generalized_box_iou


def iou(preds: Tensor, target: Tensor) -> Tensor:
    """Calculates the intersection over union.

    Args:
        preds: an Nx4 batch of prediction bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
        target: an Mx4 batch of target bounding boxes with representation ``[x_min, y_min, x_max, y_max]``

    Example:

        >>> import torch
        >>> from pl_bolts.metrics.object_detection import iou
        >>> preds = torch.tensor([[100, 100, 200, 200]])
        >>> target = torch.tensor([[150, 150, 250, 250]])
        >>> iou(preds, target)
        tensor([[0.1429]])

    Returns:
        IoU tensor: an NxM tensor containing the pairwise IoU values for every element in preds and target,
                    where N is the number of prediction bounding boxes and M is the number of target bounding boxes
    """
    return box_iou(preds, target)


def giou(preds: Tensor, target: Tensor) -> Tensor:
    """Calculates the generalized intersection over union.

    It has been proposed in `Generalized Intersection over Union: A Metric and A
    Loss for Bounding Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        preds: an Nx4 batch of prediction bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
        target: an Mx4 batch of target bounding boxes with representation ``[x_min, y_min, x_max, y_max]``

    Example:

        >>> import torch
        >>> from pl_bolts.metrics.object_detection import giou
        >>> preds = torch.tensor([[100, 100, 200, 200]])
        >>> target = torch.tensor([[150, 150, 250, 250]])
        >>> giou(preds, target)
        tensor([[-0.0794]])

    Returns:
        GIoU in an NxM tensor containing the pairwise GIoU values for every element in preds and target,
        where N is the number of prediction bounding boxes and M is the number of target bounding boxes
    """
    return generalized_box_iou(preds, target)
