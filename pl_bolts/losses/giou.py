"""
Generalized Intersection over Union (GIoU) loss (Rezatofighi et. al)
"""

import torch


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the generalized intersection over union loss.

    Args:
        pred: batch of prediction bounding boxes with representation
              [x_min, y_min, x_max, y_max]
        target: batch of target bounding boxes with representation
                [x_min, y_min, x_max, y_max]

    Returns:
        loss
    """
    eps = 1e-6
    x_min = torch.max(pred[:, 0], target[:, 0])
    y_min = torch.max(pred[:, 1], target[:, 1])
    x_max = torch.min(pred[:, 2], target[:, 2])
    y_max = torch.min(pred[:, 3], target[:, 3])
    intersection = (x_max - x_min) * (y_max - y_min)
    pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = pred_area + target_area - intersection
    C_x_min = torch.min(pred[:, 0], target[:, 0])
    C_y_min = torch.min(pred[:, 1], target[:, 1])
    C_x_max = torch.max(pred[:, 2], target[:, 2])
    C_y_max = torch.max(pred[:, 3], target[:, 3])
    C_area = (C_x_max - C_x_min) * (C_y_max - C_y_min)
    iou = torch.true_divide(intersection, union + eps)
    giou = iou - torch.true_divide((C_area - union), C_area + eps)

    return 1 - giou
