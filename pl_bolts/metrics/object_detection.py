import torch
from torch import Tensor


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
    x_min = torch.max(preds[:, None, 0], target[:, 0])
    y_min = torch.max(preds[:, None, 1], target[:, 1])
    x_max = torch.min(preds[:, None, 2], target[:, 2])
    y_max = torch.min(preds[:, None, 3], target[:, 3])
    intersection = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
    pred_area = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area[:, None] + target_area - intersection
    iou = torch.true_divide(intersection, union)
    return iou


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
    x_min = torch.max(preds[:, None, 0], target[:, 0])
    y_min = torch.max(preds[:, None, 1], target[:, 1])
    x_max = torch.min(preds[:, None, 2], target[:, 2])
    y_max = torch.min(preds[:, None, 3], target[:, 3])
    intersection = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
    pred_area = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area[:, None] + target_area - intersection
    C_x_min = torch.min(preds[:, None, 0], target[:, 0])
    C_y_min = torch.min(preds[:, None, 1], target[:, 1])
    C_x_max = torch.max(preds[:, None, 2], target[:, 2])
    C_y_max = torch.max(preds[:, None, 3], target[:, 3])
    C_area = (C_x_max - C_x_min).clamp(min=0) * (C_y_max - C_y_min).clamp(min=0)
    iou = torch.true_divide(intersection, union)
    giou = iou - torch.true_divide((C_area - union), C_area)
    return giou
