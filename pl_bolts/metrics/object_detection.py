import torch


def giou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the generalized intersection over union.

    It has been proposed in `Generalized Intersection over Union: A Metric and A
    Loss for Bounding Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred: batch of prediction bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
        target: batch of target bounding boxes with representation ``[x_min, y_min, x_max, y_max]``

    Returns:
        GIoU value
    """
    x_min = torch.max(pred[:, None, 0], target[:, 0])
    y_min = torch.max(pred[:, None, 1], target[:, 1])
    x_max = torch.min(pred[:, None, 2], target[:, 2])
    y_max = torch.min(pred[:, None, 3], target[:, 3])
    intersection = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area[:, None] + target_area - intersection
    C_x_min = torch.min(pred[:, None, 0], target[:, 0])
    C_y_min = torch.min(pred[:, None, 1], target[:, 1])
    C_x_max = torch.max(pred[:, None, 2], target[:, 2])
    C_y_max = torch.max(pred[:, None, 3], target[:, 3])
    C_area = (C_x_max - C_x_min).clamp(min=0) * (C_y_max - C_y_min).clamp(min=0)
    iou = torch.true_divide(intersection, union)
    return iou - torch.true_divide((C_area - union), C_area)
