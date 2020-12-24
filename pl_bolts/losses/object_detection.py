"""
Generalized Intersection over Union (GIoU) loss (Rezatofighi et. al)
"""

import torch

from pl_bolts.metrics.object_detection import giou


def giou_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the generalized intersection over union loss.

    It has been proposed in `Generalized Intersection over Union: A Metric and A
    Loss for Bounding Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        preds: an Nx4 batch of prediction bounding boxes with representation ``[x_min, y_min, x_max, y_max]``
        target: an Mx4 batch of target bounding boxes with representation ``[x_min, y_min, x_max, y_max]``

    Example:

        >>> import torch
        >>> from pl_bolts.losses.object_detection import giou_loss
        >>> preds = torch.tensor([[100, 100, 200, 200]])
        >>> target = torch.tensor([[150, 150, 250, 250]])
        >>> giou_loss(preds, target)
        tensor([[1.0794]])

    Returns:
        GIoU loss in an NxM tensor containing the pairwise GIoU loss for every element in preds and target,
        where N is the number of prediction bounding boxes and M is the number of target bounding boxes
    """
    loss = 1 - giou(preds, target)
    return loss
