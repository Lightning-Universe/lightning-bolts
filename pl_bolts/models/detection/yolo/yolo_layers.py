from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.ops import box_iou

    try:
        from torchvision.ops import generalized_box_iou
    except ImportError:
        _GIOU_AVAILABLE = False
    else:
        _GIOU_AVAILABLE = True
else:
    warn_missing_pkg("torchvision")


@under_review()
def _corner_coordinates(xy: Tensor, wh: Tensor) -> Tensor:
    """Converts box center points and sizes to corner coordinates.

    Args:
        xy: Center coordinates. Tensor of size ``[..., 2]``.
        wh: Width and height. Tensor of size ``[..., 2]``.

    Returns:
        A matrix of `(x1, y1, x2, y2)` coordinates.
    """
    half_wh = wh / 2
    top_left = xy - half_wh
    bottom_right = xy + half_wh
    return torch.cat((top_left, bottom_right), -1)


@under_review()
def _aligned_iou(dims1: Tensor, dims2: Tensor) -> Tensor:
    """Calculates a matrix of intersections over union from box dimensions, assuming that the boxes are located at
    the same coordinates.

    Args:
        dims1: Width and height of `N` boxes. Tensor of size ``[N, 2]``.
        dims2: Width and height of `M` boxes. Tensor of size ``[M, 2]``.

    Returns:
        Tensor of size ``[N, M]`` containing the pairwise IoU values for every element in
        ``dims1`` and ``dims2``
    """
    area1 = dims1[:, 0] * dims1[:, 1]  # [N]
    area2 = dims2[:, 0] * dims2[:, 1]  # [M]

    inter_wh = torch.min(dims1[:, None, :], dims2)  # [N, M, 2]
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter  # [N, M]

    return inter / union


@under_review()
class SELoss(nn.MSELoss):
    def __init__(self):
        super().__init__(reduction="none")

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return super().forward(inputs, target).sum(1)


@under_review()
class IoULoss(nn.Module):
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return 1.0 - box_iou(inputs, target).diagonal()


@under_review()
class GIoULoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if not _GIOU_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                "A more recent version of `torchvision` is needed for generalized IoU loss."
            )

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return 1.0 - generalized_box_iou(inputs, target).diagonal()


@under_review()
class DetectionLayer(nn.Module):
    """A YOLO detection layer.

    A YOLO model has usually 1 - 3 detection layers at different
    resolutions. The loss should be summed from all of them.
    """

    def __init__(
        self,
        num_classes: int,
        anchor_dims: List[Tuple[int, int]],
        anchor_ids: List[int],
        xy_scale: float = 1.0,
        input_is_normalized: bool = False,
        ignore_threshold: float = 0.5,
        overlap_loss_func: Optional[Callable] = None,
        class_loss_func: Optional[Callable] = None,
        confidence_loss_func: Optional[Callable] = None,
        image_space_loss: bool = False,
        overlap_loss_multiplier: float = 1.0,
        class_loss_multiplier: float = 1.0,
        confidence_loss_multiplier: float = 1.0,
    ) -> None:
        """
        Args:
            num_classes: Number of different classes that this layer predicts.
            anchor_dims: A list of all the predefined anchor box dimensions. The list should
                contain (width, height) tuples in the network input resolution (relative to the
                width and height defined in the configuration file).
            anchor_ids: List of indices to ``anchor_dims`` that is used to select the (usually 3)
                anchors that this layer uses.
            xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor.
                Using a value > 1.0 helps to produce coordinate values close to one.
            input_is_normalized: The input is normalized by logistic activation in the previous
                layer. In this case the detection layer will not take the sigmoid of the coordinate
                and probability predictions, and the width and height are scaled up so that the
                maximum value is four times the anchor dimension
            ignore_threshold: If a predictor is not responsible for predicting any target, but the
                corresponding anchor has IoU with some target greater than this threshold, the
                predictor will not be taken into account when calculating the confidence loss.
            overlap_loss_func: Loss function for bounding box coordinates. Default is the sum of
                squared errors.
            class_loss_func: Loss function for class probability distribution. Default is the sum
                of squared errors.
            confidence_loss_func: Loss function for confidence score. Default is the sum of squared
                errors.
            image_space_loss: If set to ``True``, the overlap loss function will receive the
                bounding box `(x1, y1, x2, y2)` coordinates, scaled to the input image size. This is
                needed for the IoU losses introduced in YOLOv4. Otherwise the loss will be computed
                from the x, y, width, and height values, as predicted by the network (i.e. relative
                to the anchor box, and width and height are logarithmic).
            overlap_loss_multiplier: Multiply the overlap loss by this factor.
            class_loss_multiplier: Multiply the classification loss by this factor.
            confidence_loss_multiplier: Multiply the confidence loss by this factor.
        """
        super().__init__()

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("YOLO model uses `torchvision`, which is not installed yet.")

        self.num_classes = num_classes
        self.all_anchor_dims = anchor_dims
        self.anchor_dims = [anchor_dims[i] for i in anchor_ids]
        self.anchor_map = [anchor_ids.index(i) if i in anchor_ids else -1 for i in range(len(anchor_dims))]
        self.xy_scale = xy_scale
        self.input_is_normalized = input_is_normalized
        self.ignore_threshold = ignore_threshold

        self.overlap_loss_func = overlap_loss_func or SELoss()
        self.class_loss_func = class_loss_func or SELoss()
        self.confidence_loss_func = confidence_loss_func or nn.MSELoss(reduction="none")
        self.image_space_loss = image_space_loss
        self.overlap_loss_multiplier = overlap_loss_multiplier
        self.class_loss_multiplier = class_loss_multiplier
        self.confidence_loss_multiplier = confidence_loss_multiplier

    def forward(
        self, x: Tensor, image_size: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Runs a forward pass through this YOLO detection layer.

        Maps cell-local coordinates to global coordinates in the image space, scales the bounding
        boxes with the anchors, converts the center coordinates to corner coordinates, and maps
        probabilities to the `]0, 1[` range using sigmoid.

        If targets are given, computes also losses from the predictions and the targets. This layer
        is responsible only for the targets that best match one of the anchors assigned to this
        layer.

        Args:
            x: The output from the previous layer. Tensor of size
                ``[batch_size, boxes_per_cell * (num_classes + 5), height, width]``.
            image_size: Image width and height in a vector (defines the scale of the predicted and
                target coordinates).
            targets: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.

        Returns:
            output (Tensor), losses (Dict[str, Tensor]), hits (int): Layer output tensor, sized
            ``[batch_size, num_anchors * height * width, num_classes + 5]``. If training targets
            were provided, also returns a dictionary of losses and the number of targets that this
            layer was responsible for.
        """
        batch_size, num_features, height, width = x.shape
        num_attrs = self.num_classes + 5
        boxes_per_cell = num_features // num_attrs
        if boxes_per_cell != len(self.anchor_dims):
            raise MisconfigurationException(
                "The model predicts {} bounding boxes per cell, but {} anchor boxes are defined "
                "for this layer.".format(boxes_per_cell, len(self.anchor_dims))
            )

        # Reshape the output to have the bounding box attributes of each grid cell on its own row.
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, boxes_per_cell * num_attrs]
        x = x.view(batch_size, height, width, boxes_per_cell, num_attrs)

        # Take the sigmoid of the bounding box coordinates, confidence score, and class
        # probabilities, unless the input is normalized by the previous layer activation.
        if self.input_is_normalized:
            xy = x[..., :2]
            confidence = x[..., 4]
            classprob = x[..., 5:]
        else:
            xy = torch.sigmoid(x[..., :2])
            confidence = torch.sigmoid(x[..., 4])
            classprob = torch.sigmoid(x[..., 5:])
        wh = x[..., 2:4]

        # Eliminate grid sensitivity. The previous layer should output extremely high values for
        # the sigmoid to produce x/y coordinates close to one. YOLOv4 solves this by scaling the
        # x/y coordinates.
        xy = xy * self.xy_scale - 0.5 * (self.xy_scale - 1)

        image_xy = self._global_xy(xy, image_size)
        if self.input_is_normalized:
            image_wh = 4 * torch.square(wh) * torch.tensor(self.anchor_dims, dtype=wh.dtype, device=wh.device)
        else:
            image_wh = torch.exp(wh) * torch.tensor(self.anchor_dims, dtype=wh.dtype, device=wh.device)
        boxes = _corner_coordinates(image_xy, image_wh)
        output = torch.cat((boxes, confidence.unsqueeze(-1), classprob), -1)
        output = output.reshape(batch_size, height * width * boxes_per_cell, num_attrs)

        if targets is None:
            return output

        lc_mask = self._low_confidence_mask(boxes, targets)
        if not self.image_space_loss:
            boxes = torch.cat((xy, wh), -1)
        losses, hits = self._calculate_losses(boxes, confidence, classprob, targets, image_size, lc_mask)
        return output, losses, hits

    def _global_xy(self, xy: Tensor, image_size: Tensor) -> Tensor:
        """Adds offsets to the predicted box center coordinates to obtain global coordinates to the image.

        The predicted coordinates are interpreted as coordinates inside a grid cell whose width and
        height is 1. Adding offset to the cell, dividing by the grid size, and multiplying by the
        image size, we get global coordinates in the image scale.

        Args:
            xy: The predicted center coordinates before scaling. Values from zero to one in a
                tensor sized ``[batch_size, height, width, boxes_per_cell, 2]``.
            image_size: Width and height in a vector that will be used to scale the coordinates.

        Returns:
            Global coordinates scaled to the size of the network input image, in a tensor with the
            same shape as the input tensor.
        """
        height = xy.shape[1]
        width = xy.shape[2]
        grid_size = torch.tensor([width, height], device=xy.device)

        x_range = torch.arange(width, device=xy.device)
        y_range = torch.arange(height, device=xy.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range)
        offset = torch.stack((grid_x, grid_y), -1)  # [height, width, 2]
        offset = offset.unsqueeze(2)  # [height, width, 1, 2]

        scale = torch.true_divide(image_size, grid_size)
        return (xy + offset) * scale

    def _low_confidence_mask(self, boxes: Tensor, targets: List[Dict[str, Tensor]]) -> Tensor:
        """Initializes the mask that will be used to select predictors that are not predicting any ground-truth
        target. The value will be ``True``, unless the predicted box overlaps any target significantly (IoU greater
        than ``self.ignore_threshold``).

        Args:
            boxes: The predicted corner coordinates in the image space. Tensor of size
                ``[batch_size, height, width, boxes_per_cell, 4]``.
            targets: List of dictionaries of ground-truth targets, one dictionary per image.

        Returns:
            A boolean tensor shaped ``[batch_size, height, width, boxes_per_cell]`` with ``False``
            where the predicted box overlaps a target significantly and ``True`` elsewhere.
        """
        batch_size, height, width, boxes_per_cell, num_coords = boxes.shape
        num_preds = height * width * boxes_per_cell
        boxes = boxes.view(batch_size, num_preds, num_coords)

        results = torch.ones((batch_size, num_preds), dtype=torch.bool, device=boxes.device)
        for image_idx, (image_boxes, image_targets) in enumerate(zip(boxes, targets)):
            target_boxes = image_targets["boxes"]
            if target_boxes.shape[0] > 0:
                ious = box_iou(image_boxes, target_boxes)  # [num_preds, num_targets]
                best_iou = ious.max(-1).values  # [num_preds]
                results[image_idx] = best_iou <= self.ignore_threshold

        return results.view((batch_size, height, width, boxes_per_cell))

    def _calculate_losses(
        self,
        boxes: Tensor,
        confidence: Tensor,
        classprob: Tensor,
        targets: List[Dict[str, Tensor]],
        image_size: Tensor,
        lc_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """From the targets that are in the image space calculates the actual targets for the network predictions,
        and returns a dictionary of training losses.

        Args:
            boxes: The predicted bounding boxes. A tensor sized
                ``[batch_size, height, width, boxes_per_cell, 4]``.
            confidence: The confidence predictions, normalized to `[0, 1]`. A tensor sized
                ``[batch_size, height, width, boxes_per_cell]``.
            classprob: The class probability predictions, normalized to `[0, 1]`. A tensor sized
                ``[batch_size, height, width, boxes_per_cell, num_classes]``.
            targets: List of dictionaries of target values, one dictionary for each image.
            image_size: Width and height in a vector that defines the scale of the target
                coordinates.
            lc_mask: A boolean mask containing ``True`` where the predicted box does not overlap
                any target significantly.

        Returns:
            losses (Dict[str, Tensor]), hits (int): A dictionary of training losses and the number
            of targets that this layer was responsible for.
        """
        batch_size, height, width, boxes_per_cell, _ = boxes.shape
        device = boxes.device
        assert batch_size == len(targets)

        # A multiplier for scaling image coordinates to feature map coordinates
        grid_size = torch.tensor([width, height], device=device)
        image_to_grid = torch.true_divide(grid_size, image_size)

        anchor_wh = torch.tensor(self.all_anchor_dims, dtype=boxes.dtype, device=device)
        anchor_map = torch.tensor(self.anchor_map, dtype=torch.int64, device=device)

        # List of predicted and target values for the predictors that are responsible for
        # predicting a target.
        target_xy = []
        target_wh = []
        target_label = []
        size_compensation = []
        pred_boxes = []
        pred_classprob = []
        pred_confidence = []
        hits = 0

        for image_idx, image_targets in enumerate(targets):
            target_boxes = image_targets["boxes"]
            if target_boxes.shape[0] < 1:
                continue

            # Bounding box corner coordinates are converted to center coordinates, width, and
            # height.
            wh = target_boxes[:, 2:4] - target_boxes[:, 0:2]
            xy = target_boxes[:, 0:2] + (wh / 2)

            # The center coordinates are converted to the feature map dimensions so that the whole
            # number tells the cell index and the fractional part tells the location inside the cell.
            grid_xy = xy * image_to_grid
            cell_i = grid_xy[:, 0].to(torch.int64).clamp(0, width - 1)
            cell_j = grid_xy[:, 1].to(torch.int64).clamp(0, height - 1)

            # We want to know which anchor box overlaps a ground truth box more than any other
            # anchor box. We know that the anchor box is located in the same grid cell as the
            # ground truth box. For each prior shape (width, height), we calculate the IoU with
            # all ground truth boxes, assuming the boxes are at the same location. Then for each
            # target, we select the prior shape that gives the highest IoU.
            ious = _aligned_iou(wh, anchor_wh)
            best_anchors = ious.max(1).indices

            # ``anchor_map`` maps the anchor indices to the predictors in this layer, or to -1 if
            # it's not an anchor of this layer. We ignore the predictions if the best anchor is in
            # another layer.
            predictors = anchor_map[best_anchors]
            selected = predictors >= 0
            cell_i = cell_i[selected]
            cell_j = cell_j[selected]
            predictors = predictors[selected]
            wh = wh[selected]
            # sum() is equivalent to count_nonzero() and available before PyTorch 1.7.
            hits += selected.sum()

            # The "low-confidence" mask is used to select predictors that are not responsible for
            # predicting any object, for calculating the part of the confidence loss with zero as
            # the target confidence.
            lc_mask[image_idx, cell_j, cell_i, predictors] = False

            # IoU losses are calculated from the image space coordinates. The squared-error loss is
            # calculated from the raw predicted values.
            if self.image_space_loss:
                xy = xy[selected]
                target_xy.append(xy)
                target_wh.append(wh)
            else:
                grid_xy = grid_xy[selected]
                best_anchors = best_anchors[selected]
                relative_xy = grid_xy - grid_xy.floor()
                if self.input_is_normalized:
                    relative_wh = torch.sqrt(wh / (4 * anchor_wh[best_anchors] + 1e-16))
                else:
                    relative_wh = torch.log(wh / anchor_wh[best_anchors] + 1e-16)
                target_xy.append(relative_xy)
                target_wh.append(relative_wh)

            # Size compensation factor for bounding box overlap loss is calculated from unit width
            # and height.
            unit_wh = wh / image_size
            size_compensation.append(2 - (unit_wh[:, 0] * unit_wh[:, 1]))

            # The data may contain a different number of classes than this detection layer. In case
            # a label is greater than the number of classes that this layer predicts, it will be
            # mapped to the last class.
            labels = image_targets["labels"]
            labels = labels[selected]
            labels = torch.min(labels, torch.tensor(self.num_classes - 1, device=device))
            target_label.append(labels)

            pred_boxes.append(boxes[image_idx, cell_j, cell_i, predictors])
            pred_classprob.append(classprob[image_idx, cell_j, cell_i, predictors])
            pred_confidence.append(confidence[image_idx, cell_j, cell_i, predictors])

        losses = dict()

        if pred_boxes and target_xy and target_wh:
            size_compensation = torch.cat(size_compensation)
            pred_boxes = torch.cat(pred_boxes)
            if self.image_space_loss:
                target_boxes = _corner_coordinates(torch.cat(target_xy), torch.cat(target_wh))
            else:
                target_boxes = torch.cat((torch.cat(target_xy), torch.cat(target_wh)), -1)
            overlap_loss = self.overlap_loss_func(pred_boxes, target_boxes)
            overlap_loss = overlap_loss * size_compensation
            overlap_loss = overlap_loss.sum() / batch_size
            losses["overlap"] = overlap_loss * self.overlap_loss_multiplier
        else:
            losses["overlap"] = torch.tensor(0.0, device=device)

        if pred_classprob and target_label:
            pred_classprob = torch.cat(pred_classprob)
            target_label = torch.cat(target_label)
            target_classprob = torch.nn.functional.one_hot(target_label, self.num_classes)
            target_classprob = target_classprob.to(dtype=pred_classprob.dtype)
            class_loss = self.class_loss_func(pred_classprob, target_classprob)
            class_loss = class_loss.sum() / batch_size
            losses["class"] = class_loss * self.class_loss_multiplier
        else:
            losses["class"] = torch.tensor(0.0, device=device)

        pred_low_confidence = confidence[lc_mask]
        target_low_confidence = torch.zeros_like(pred_low_confidence)
        if pred_confidence:
            pred_high_confidence = torch.cat(pred_confidence)
            target_high_confidence = torch.ones_like(pred_high_confidence)
            pred_confidence = torch.cat((pred_low_confidence, pred_high_confidence))
            target_confidence = torch.cat((target_low_confidence, target_high_confidence))
        else:
            pred_confidence = pred_low_confidence
            target_confidence = target_low_confidence
        confidence_loss = self.confidence_loss_func(pred_confidence, target_confidence)
        confidence_loss = confidence_loss.sum() / batch_size
        losses["confidence"] = confidence_loss * self.confidence_loss_multiplier

        return losses, hits


@under_review()
class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


@under_review()
class RouteLayer(nn.Module):
    """Route layer concatenates the output (or part of it) from given layers."""

    def __init__(self, source_layers: List[int], num_chunks: int, chunk_idx: int) -> None:
        """
        Args:
            source_layers: Indices of the layers whose output will be concatenated.
            num_chunks: Layer outputs will be split into this number of chunks.
            chunk_idx: Only the chunks with this index will be concatenated.
        """
        super().__init__()
        self.source_layers = source_layers
        self.num_chunks = num_chunks
        self.chunk_idx = chunk_idx

    def forward(self, x, outputs):
        chunks = [torch.chunk(outputs[layer], self.num_chunks, dim=1)[self.chunk_idx] for layer in self.source_layers]
        return torch.cat(chunks, dim=1)


@under_review()
class ShortcutLayer(nn.Module):
    """Shortcut layer adds a residual connection from the source layer."""

    def __init__(self, source_layer: int) -> None:
        """
        Args:
            source_layer: Index of the layer whose output will be added to the output of the
                previous layer.
        """
        super().__init__()
        self.source_layer = source_layer

    def forward(self, x, outputs):
        return outputs[-1] + outputs[self.source_layer]
