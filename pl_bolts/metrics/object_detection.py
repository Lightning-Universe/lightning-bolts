import torch
from torch import Tensor

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.ops import box_iou as iou
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


def _evaluate_iou(preds: torch.Tensor, target: torch.Tensor):
    """Evaluate intersection over union (IOU) for target from dataset and output prediction from model."""

    if preds["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=preds["boxes"].device)
    return iou(target["boxes"], preds["boxes"]).diag().mean()


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

    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (preds[:, 2:] >= preds[:, :2]).all()
    assert (target[:, 2:] >= target[:, :2]).all()

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
