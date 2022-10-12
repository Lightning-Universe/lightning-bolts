from typing import List

import torch
from torch import Tensor

from pl_bolts.utils import _TORCH_MESHGRID_REQUIRES_INDEXING, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.ops import box_iou
else:
    warn_missing_pkg("torchvision")

# PyTorch 1.10 introduced the argument "indexing" and deprecated calling without the argument. Since we call it inside
# a "@torch.jit.script" function, it's difficult to make this decision at call time.
if _TORCH_MESHGRID_REQUIRES_INDEXING:

    def meshgrid(x: Tensor, y: Tensor) -> List[Tensor]:
        return torch.meshgrid((x, y), indexing="ij")  # type: ignore

else:
    meshgrid = torch.meshgrid  # type: ignore


def grid_offsets(grid_size: Tensor) -> Tensor:
    """Given a grid size, returns a tensor containing offsets to the grid cells.

    Args:
        The width and height of the grid in a tensor.

    Returns:
        A ``[height, width, 2]`` tensor containing the grid cell `(x, y)` offsets.
    """
    x_range = torch.arange(grid_size[0].item(), device=grid_size.device)
    y_range = torch.arange(grid_size[1].item(), device=grid_size.device)
    grid_y, grid_x = meshgrid(y_range, x_range)
    return torch.stack((grid_x, grid_y), -1)


def grid_centers(grid_size: Tensor) -> Tensor:
    """Given a grid size, returns a tensor containing coordinates to the centers of the grid cells.

    Returns:
        A ``[height, width, 2]`` tensor containing coordinates to the centers of the grid cells.
    """
    return grid_offsets(grid_size) + 0.5


@torch.jit.script
def global_xy(xy: Tensor, image_size: Tensor) -> Tensor:
    """Adds offsets to the predicted box center coordinates to obtain global coordinates to the image.

    The predicted coordinates are interpreted as coordinates inside a grid cell whose width and height is 1. Adding
    offset to the cell, dividing by the grid size, and multiplying by the image size, we get global coordinates in the
    image scale.

    The function needs the ``@torch.jit.script`` decorator in order for ONNX generation to work. The tracing based
    generator will loose track of e.g. ``xy.shape[1]`` and treat it as a Python variable and not a tensor. This will
    cause the dimension to be treated as a constant in the model, which prevents dynamic input sizes.

    Args:
        xy: The predicted center coordinates before scaling. Values from zero to one in a tensor sized
            ``[batch_size, height, width, boxes_per_cell, 2]``.
        image_size: Width and height in a vector that will be used to scale the coordinates.

    Returns:
        Global coordinates scaled to the size of the network input image, in a tensor with the same shape as the input
        tensor.
    """
    height = xy.shape[1]
    width = xy.shape[2]
    grid_size = torch.tensor([width, height], device=xy.device)
    # Scripting requires explicit conversion to a floating point type.
    offset = grid_offsets(grid_size).to(xy.dtype).unsqueeze(2)  # [height, width, 1, 2]
    scale = torch.true_divide(image_size, grid_size)
    return (xy + offset) * scale


def aligned_iou(dims1: Tensor, dims2: Tensor) -> Tensor:
    """Calculates a matrix of intersections over union from box dimensions, assuming that the boxes are located at
    the same coordinates.

    Args:
        dims1: Width and height of `N` boxes. Tensor of size ``[N, 2]``.
        dims2: Width and height of `M` boxes. Tensor of size ``[M, 2]``.

    Returns:
        Tensor of size ``[N, M]`` containing the pairwise IoU values for every element in ``dims1`` and ``dims2``
    """
    area1 = dims1[:, 0] * dims1[:, 1]  # [N]
    area2 = dims2[:, 0] * dims2[:, 1]  # [M]

    inter_wh = torch.min(dims1[:, None, :], dims2)  # [N, M, 2]
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter  # [N, M]

    return inter / union


def iou_below(pred_boxes: Tensor, target_boxes: Tensor, threshold: float) -> Tensor:
    """Creates a binary mask whose value will be ``True``, unless the predicted box overlaps any target
    significantly (IoU greater than ``threshold``).

    Args:
        pred_boxes: The predicted corner coordinates. Tensor of size ``[height, width, boxes_per_cell, 4]``.
        target_boxes: Corner coordinates of the target boxes. Tensor of size ``[height, width, boxes_per_cell, 4]``.

    Returns:
        A boolean tensor sized ``[height, width, boxes_per_cell]``, with ``False`` where the predicted box overlaps a
        target significantly and ``True`` elsewhere.
    """
    shape = pred_boxes.shape[:-1]
    pred_boxes = pred_boxes.view(-1, 4)
    ious = box_iou(pred_boxes, target_boxes)
    best_iou = ious.max(-1).values
    below_threshold = best_iou <= threshold
    return below_threshold.view(shape)


def is_inside_box(points: Tensor, boxes: Tensor) -> Tensor:
    """Get pairwise truth values of whether the point is inside the box.

    Args:
        points: point (x, y) coordinates, [points, 2]
        boxes: box (x1, y1, x2, y2) coordinates, [boxes, 4]

    Returns:
        A tensor shaped ``[boxes, points]`` containing pairwise truth values of whether the points are inside the boxes.
    """
    points = points.unsqueeze(0)  # [1, points, 2]
    boxes = boxes.unsqueeze(1)  # [boxes, 1, 4]
    lt = points - boxes[..., :2]  # [boxes, points, 2]
    rb = boxes[..., 2:] - points  # [boxes, points, 2]
    deltas = torch.cat((lt, rb), -1)  # [boxes, points, 4]
    return deltas.min(-1).values > 0.0  # [boxes, points]


@torch.jit.script
def get_image_size(images: Tensor) -> Tensor:
    """Get the image size from an input tensor.

    The function needs the ``@torch.jit.script`` decorator in order for ONNX generation to work. The tracing based
    generator will loose track of e.g. ``images.shape[1]`` and treat it as a Python variable and not a tensor. This will
    cause the dimension to be treated as a constant in the model, which prevents dynamic input sizes.

    Args:
        images: An image batch to take the width and height from.

    Returns:
        A tensor that contains the image width and height.
    """
    height = images.shape[2]
    width = images.shape[3]
    return torch.tensor([width, height], device=images.device)
