from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn, Tensor

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision.ops import box_iou
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


def _corner_coordinates(xy, wh):
    """
    Converts box center points and sizes to corner coordinates.

    Args:
        xy (Tensor): Center coordinates. Tensor of size `[..., 2]`.
        wh (Tensor): Width and height. Tensor of size `[..., 2]`.

    Returns:
        boxes (Tensor): A matrix of (x1, y1, x2, y2) coordinates.
    """
    half_wh = wh / 2
    top_left = xy - half_wh
    bottom_right = xy + half_wh
    return torch.cat((top_left, bottom_right), -1)


def _area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _aligned_iou(dims1, dims2):
    """
    Calculates a matrix of intersections over union from box dimensions, assuming that the boxes
    are located at the same coordinates.

    Args:
        dims1 (Tensor[N, 2]): width and height of N boxes
        dims2 (Tensor[M, 2]): width and height of M boxes

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in
            `dims1` and `dims2`
    """
    area1 = dims1[:, 0] * dims1[:, 1]  # [N]
    area2 = dims2[:, 0] * dims2[:, 1]  # [M]

    inter_wh = torch.min(dims1[:, None, :], dims2)  # [N, M, 2]
    inter = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2 - inter  # [N, M]

    return inter / union


def _elementwise_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Returns the elementwise intersection-over-union between two sets of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[N, 4])

    Returns:
        iou (Tensor[N]): the vector containing the elementwise IoU values for every element in
        boxes1 and boxes2
    """
    area1 = _area(boxes1)
    area2 = _area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    iou = inter / (area1 + area2 - inter)
    return iou


def _elementwise_generalized_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Returns the elementwise generalized intersection-over-union between two sets of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[N, 4])

    Returns:
        generalized_iou (Tensor[N]): the vector containing the elementwise generalized IoU values
        for every element in boxes1 and boxes2
    """

    # Degenerate boxes give inf / nan results, so do an early check.
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    area1 = _area(boxes1)
    area2 = _area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union

    lti = torch.min(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    whi = (rbi - lti).clamp(min=0)  # [N,2]
    areai = whi[:, 0] * whi[:, 1]

    return iou - (areai - union) / areai


class IoULoss(nn.Module):
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return 1.0 - _elementwise_iou(inputs, target)


class GIoULoss(nn.Module):
    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        return 1.0 - _elementwise_generalized_iou(inputs, target)


class DetectionLayer(nn.Module):
    """
    A YOLO detection layer. A YOLO model has usually 1 - 3 detection layers at different
    resolutions. The loss should be summed from all of them.
    """

    def __init__(
        self,
        num_classes: int,
        image_width: int,
        image_height: int,
        anchor_dims: List[Tuple[int, int]],
        anchor_ids: List[int],
        xy_scale: float = 1.0,
        ignore_threshold: float = 0.5,
        overlap_loss_func: Callable = None,
        class_loss_func: Callable = None,
        confidence_loss_func: Callable = None,
        image_space_loss: bool = False,
        overlap_loss_multiplier: float = 1.0,
        class_loss_multiplier: float = 1.0,
        confidence_loss_multiplier: float = 1.0
    ):
        """
        Args:
            num_classes: Number of different classes that this layer predicts.
            image_width: Image width (defines the scale of the anchor box and target bounding box
                dimensions).
            image_height: Image height (defines the scale of the anchor box and target bounding box
                dimensions).
            anchor_dims: A list of all the predefined anchor box dimensions. The list should
                contain (width, height) tuples in the network input resolution (relative to the
                width and height defined in the configuration file).
            anchor_ids: List of indices to `anchor_dims` that is used to select the (usually 3)
                anchors that this layer uses.
            xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor.
                Using a value > 1.0 helps to produce coordinate values close to one.
            ignore_threshold: If a predictor is not responsible for predicting any target, but the
                corresponding anchor has IoU with some target greater than this threshold, the
                predictor will not be taken into account when calculating the confidence loss.
            overlap_loss_func: Loss function for bounding box coordinates. Default is the sum of
                squared errors.
            class_loss_func: Loss function for class probability distribution. Default is the sum
                of squared errors.
            confidence_loss_func: Loss function for confidence score. Default is the sum of squared
                errors.
            image_space_loss: If set to `True`, the overlap loss function will receive the bounding
                box (x1, y1, x2, y2) coordinate normalized to the [0, 1] range. This is needed for
                the IoU losses introduced in YOLOv4. Otherwise the loss will be computed from the x,
                y, width, and height values, as predicted by the network (i.e. relative to the
                anchor box, and width and height are logarithmic).
            coord_loss_multiplier: Multiply the coordinate/size loss by this factor.
            class_loss_multiplier: Multiply the classification loss by this factor.
            confidence_loss_multiplier: Multiply the confidence loss by this factor.
        """
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                'YOLO model uses `torchvision`, which is not installed yet.'
            )

        self.num_classes = num_classes
        self.image_width = image_width
        self.image_height = image_height
        self.anchor_dims = anchor_dims
        self.anchor_ids = anchor_ids
        self.anchor_map = [anchor_ids.index(i) if i in anchor_ids else -1 for i in range(9)]
        self.xy_scale = xy_scale
        self.ignore_threshold = ignore_threshold

        se_loss = nn.MSELoss(reduction='none')
        self.overlap_loss_func = overlap_loss_func or se_loss
        self.class_loss_func = class_loss_func or se_loss
        self.confidence_loss_func = confidence_loss_func or se_loss
        self.image_space_loss = image_space_loss
        self.overlap_loss_multiplier = overlap_loss_multiplier
        self.class_loss_multiplier = class_loss_multiplier
        self.confidence_loss_multiplier = confidence_loss_multiplier

    def forward(self, x: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Runs a forward pass through this YOLO detection layer.

        Maps cell-local coordinates to global coordinates in the [0, 1] range, scales the bounding
        boxes with the anchors, converts the center coordinates to corner coordinates, and maps
        probabilities to ]0, 1[ range using sigmoid.

        Args:
            x: The output from the previous layer. Tensor of size
                `[batch_size, boxes_per_cell * (num_classes + 5), height, width]`.
            targets: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.

        Returns:
            output (Tensor), losses (Dict[str, Tensor]): Layer output, and if training targets were
                provided, a dictionary of losses. Layer output is sized
                `[batch_size, num_anchors * height * width, num_classes + 5]`.
        """
        batch_size, num_features, height, width = x.shape
        num_attrs = self.num_classes + 5
        boxes_per_cell = num_features // num_attrs
        if boxes_per_cell != len(self.anchor_ids):
            raise MisconfigurationException(
                "The model predicts {} bounding boxes per cell, but {} anchor boxes are defined "
                "for this layer.".format(boxes_per_cell, len(self.anchor_ids))
            )

        # Reshape the output to have the bounding box attributes of each grid cell on its own row.
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, boxes_per_cell * num_attrs]
        x = x.view(batch_size, height, width, boxes_per_cell, num_attrs)

        # Take the sigmoid of the bounding box coordinates, confidence score, and class
        # probabilities.
        xy = torch.sigmoid(x[..., :2])
        wh = x[..., 2:4]
        confidence = torch.sigmoid(x[..., 4])
        classprob = torch.sigmoid(x[..., 5:])

        # Eliminate grid sensitivity. The previous layer should output extremely high values for
        # the sigmoid to produce x/y coordinates close to one. YOLOv4 solves this by scaling the
        # x/y coordinates.
        xy = xy * self.xy_scale - 0.5 * (self.xy_scale - 1)

        if not torch.isfinite(x).all():
            raise ValueError('Detection layer output contains nan or inf values.')

        image_xy = self._global_xy(xy)
        image_wh = self._scale_wh(wh)
        boxes = _corner_coordinates(image_xy, image_wh)
        output = torch.cat((boxes, confidence.unsqueeze(-1), classprob), -1)
        output = output.reshape(batch_size, height * width * boxes_per_cell, num_attrs)

        if targets is None:
            return output

        lc_mask = self._low_confidence_mask(boxes, targets)
        if not self.image_space_loss:
            boxes = torch.cat((xy, wh), -1)
        losses = self._calculate_losses(boxes, confidence, classprob, targets, lc_mask)
        return output, losses

    def _global_xy(self, xy):
        """
        Adds offsets to the predicted box center coordinates to obtain global coordinates to the
        image.

        The predicted coordinates are interpreted as coordinates inside a grid cell whose width and
        height is 1. Adding offset to the cell and dividing by the grid size, we get global
        coordinates in the [0, 1] range.

        Args:
            xy (Tensor): The predicted center coordinates before scaling. Values from zero to one
                in a tensor sized `[batch_size, height, width, boxes_per_cell, 2]`.

        Returns:
            result (Tensor): Global coordinates from zero to one, in a tensor with the same shape
                as the input tensor.
        """
        height = xy.shape[1]
        width = xy.shape[2]
        grid_size = torch.tensor([width, height], device=xy.device)

        x_range = torch.arange(width, dtype=xy.dtype, device=xy.device)
        y_range = torch.arange(height, dtype=xy.dtype, device=xy.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range)
        offset = torch.stack((grid_x, grid_y), -1)  # [height, width, 2]
        offset = offset.unsqueeze(2)  # [height, width, 1, 2]

        return (xy + offset) / grid_size

    def _scale_wh(self, wh):
        """
        Scales the box size predictions by the prior dimensions from the anchors.

        Args:
            wh (Tensor): The unnormalized width and height predictions. Tensor of size
                `[..., boxes_per_cell, 2]`.

        Returns:
            result (Tensor): A tensor with the same shape as the input tensor, but scaled sizes
                normalized to the [0, 1] range.
        """
        image_size = torch.tensor([self.image_width, self.image_height], device=wh.device)
        anchor_wh = [self.anchor_dims[i] for i in self.anchor_ids]
        anchor_wh = torch.tensor(anchor_wh, dtype=wh.dtype, device=wh.device)
        return torch.exp(wh) * anchor_wh / image_size

    def _low_confidence_mask(self, boxes, targets):
        """
        Initializes the mask that will be used to select predictors that are not predicting any
        ground-truth target. The value will be `True`, unless the predicted box overlaps any target
        significantly (IoU greater than `self.ignore_threshold`).

        Args:
            boxes (Tensor): The predicted corner coordinates, normalized to the [0, 1] range.
                Tensor of size `[batch_size, height, width, boxes_per_cell, 4]`.
            targets (List[Dict[str, Tensor]]): List of dictionaries of target values, one
                dictionary for each image.

        Returns:
            results (Tensor): A boolean tensor shaped `[batch_size, height, width, boxes_per_cell]`
                with `False` where the predicted box overlaps a target significantly and `True`
                elsewhere.
        """
        batch_size, height, width, boxes_per_cell, num_coords = boxes.shape
        num_preds = height * width * boxes_per_cell
        boxes = boxes.view(batch_size, num_preds, num_coords)

        scale = torch.tensor([self.image_width, self.image_height, self.image_width, self.image_height],
                             device=boxes.device)
        boxes = boxes * scale

        results = torch.ones((batch_size, num_preds), dtype=torch.bool, device=boxes.device)
        for image_idx, (image_boxes, image_targets) in enumerate(zip(boxes, targets)):
            target_boxes = image_targets['boxes']
            if target_boxes.shape[0] > 0:
                ious = box_iou(image_boxes, target_boxes)  # [num_preds, num_targets]
                best_iou = ious.max(-1).values  # [num_preds]
                results[image_idx] = best_iou <= self.ignore_threshold

        return results.view((batch_size, height, width, boxes_per_cell))

    def _calculate_losses(self, boxes, confidence, classprob, targets, lc_mask):
        """
        From the targets that are in the image space calculates the actual targets for the network
        predictions, and returns a dictionary of training losses.

        Args:
            boxes (Tensor): The predicted bounding boxes. A tensor sized
                `[batch_size, height, width, boxes_per_cell, 4]`.
            confidence (Tensor): The confidence predictions, normalized to [0, 1]. A tensor sized
                `[batch_size, height, width, boxes_per_cell]`.
            classprob (Tensor): The class probability predictions, normalized to [0, 1]. A tensor
                sized `[batch_size, height, width, boxes_per_cell, num_classes]`.
            targets (List[Dict[str, Tensor]]): List of dictionaries of target values, one
                dictionary for each image.
            lc_mask (Tensor): A boolean mask containing `True` where the predicted box does not
                overlap any target significantly.

        Returns:
            predicted (Dict[str, Tensor]): A dictionary of training losses.
        """
        batch_size, height, width, boxes_per_cell, _ = boxes.shape
        device = boxes.device
        assert batch_size == len(targets)

        # Divisor for converting targets from image coordinates to feature map coordinates
        image_to_feature_map = torch.tensor([self.image_width / width, self.image_height / height], device=device)
        # Divisor for converting targets from image coordinates to [0, 1] range
        image_to_unit = torch.tensor([self.image_width, self.image_height], device=device)

        anchor_wh = torch.tensor(self.anchor_dims, dtype=boxes.dtype, device=device)
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

        for image_idx, image_targets in enumerate(targets):
            target_boxes = image_targets['boxes']
            if target_boxes.shape[0] < 1:
                continue

            # Bounding box corner coordinates are converted to center coordinates, width, and
            # height, and normalized to [0, 1] range.
            wh = target_boxes[:, 2:4] - target_boxes[:, 0:2]
            xy = target_boxes[:, 0:2] + (wh / 2)
            unit_xy = xy / image_to_unit
            unit_wh = wh / image_to_unit

            # The center coordinates are converted to the feature map dimensions so that the whole
            # number tells the cell index and the fractional part tells the location inside the cell.
            xy = xy / image_to_feature_map
            cell_i = xy[:, 0].to(torch.int64).clamp(0, width - 1)
            cell_j = xy[:, 1].to(torch.int64).clamp(0, height - 1)

            # We want to know which anchor box overlaps a ground truth box more than any other
            # anchor box. We know that the anchor box is located in the same grid cell as the
            # ground truth box. For each prior shape (width, height), we calculate the IoU with
            # all ground truth boxes, assuming the boxes are at the same location. Then for each
            # target, we select the prior shape that gives the highest IoU.
            ious = _aligned_iou(wh, anchor_wh)
            best_anchors = ious.max(1).indices

            # `anchor_map` maps the anchor indices to the predictors in this layer, or to -1 if
            # it's not an anchor of this layer. We ignore the predictions if the best anchor is in
            # another layer.
            predictors = anchor_map[best_anchors]
            selected = predictors >= 0
            unit_xy = unit_xy[selected]
            unit_wh = unit_wh[selected]
            cell_i = cell_i[selected]
            cell_j = cell_j[selected]
            predictors = predictors[selected]
            best_anchors = best_anchors[selected]

            # The "low-confidence" mask is used to select predictors that are not responsible for
            # predicting any object, for calculating the part of the confidence loss with zero as
            # the target confidence.
            lc_mask[image_idx, cell_j, cell_i, predictors] = False

            # IoU losses are calculated from the image space coordinates normalized to [0, 1]
            # range. The squared-error loss is calculated from the raw predicted values.
            if self.image_space_loss:
                target_xy.append(unit_xy)
                target_wh.append(unit_wh)
            else:
                xy = xy[selected]
                wh = wh[selected]
                relative_xy = xy - xy.floor()
                relative_wh = torch.log(wh / anchor_wh[best_anchors] + 1e-16)
                target_xy.append(relative_xy)
                target_wh.append(relative_wh)

            # Size compensation factor for bounding box overlap loss is calculated from image space
            # width and height.
            size_compensation.append(2 - (unit_wh[:, 0] * unit_wh[:, 1]))

            # The data may contain a different number of classes than this detection layer. In case
            # a label is greater than the number of classes that this layer predicts, it will be
            # mapped to the last class.
            labels = image_targets['labels']
            labels = labels[selected]
            labels = torch.min(labels, torch.tensor(self.num_classes - 1, device=device))
            target_label.append(labels)

            pred_boxes.append(boxes[image_idx, cell_j, cell_i, predictors])
            pred_classprob.append(classprob[image_idx, cell_j, cell_i, predictors])
            pred_confidence.append(confidence[image_idx, cell_j, cell_i, predictors])

        losses = dict()

        if pred_boxes and target_xy and target_wh:
            size_compensation = torch.cat(size_compensation).unsqueeze(1)
            pred_boxes = torch.cat(pred_boxes)
            if self.image_space_loss:
                target_boxes = _corner_coordinates(torch.cat(target_xy), torch.cat(target_wh))
            else:
                target_boxes = torch.cat((torch.cat(target_xy), torch.cat(target_wh)), -1)
            overlap_loss = self.overlap_loss_func(pred_boxes, target_boxes)
            overlap_loss = overlap_loss * size_compensation
            overlap_loss = overlap_loss.sum() / batch_size
            losses['overlap'] = overlap_loss * self.overlap_loss_multiplier

        if pred_classprob and target_label:
            pred_classprob = torch.cat(pred_classprob)
            target_label = torch.cat(target_label)
            target_classprob = torch.nn.functional.one_hot(target_label, self.num_classes)
            target_classprob = target_classprob.to(dtype=pred_classprob.dtype)
            class_loss = self.class_loss_func(pred_classprob, target_classprob)
            class_loss = class_loss.sum() / batch_size
            losses['class'] = class_loss * self.class_loss_multiplier

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
        losses['confidence'] = confidence_loss * self.confidence_loss_multiplier

        return losses


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class RouteLayer(nn.Module):
    """Route layer concatenates the output (or part of it) from given layers."""

    def __init__(self, source_layers: List[int], num_chunks: int, chunk_idx: int):
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


class ShortcutLayer(nn.Module):
    """Shortcut layer adds a residual connection from the source layer."""

    def __init__(self, source_layer: int):
        """
        Args:
            source_layer: Index of the layer whose output will be added to the output of the
                previous layer.
        """
        super().__init__()
        self.source_layer = source_layer

    def forward(self, x, outputs):
        return outputs[-1] + outputs[self.source_layer]
