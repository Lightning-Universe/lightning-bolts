from typing import Dict, List, Optional, Tuple

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


def _aligned_iou(dims1, dims2):
    """
    Calculates a matrix of intersections over union from box dimensions, assuming that the boxes
    are located at the same coordinates.

    Arguments:
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


class DetectionLayer(nn.Module):
    """
    A YOLO detection layer. A YOLO model has usually 1 - 3 detection layers at different
    resolutions. The loss should be summed from all of them.
    """

    def __init__(self,
                 num_classes: int,
                 image_width: int,
                 image_height: int,
                 anchor_dims: List[Tuple[int, int]],
                 anchor_ids: List[int],
                 xy_scale: float = 1.0,
                 ignore_threshold: float = 0.5,
                 coord_loss_multiplier: float = 1.0,
                 class_loss_multiplier: float = 1.0,
                 confidence_loss_multiplier: float = 1.0):
        """
        Args:
            num_classes: Number of different classes that this layer predicts.
            image_width: Image width (defines the scale of the anchor box and target bounding
                box dimensions).
            image_height: Image height (defines the scale of the anchor box and target
                bounding box dimensions).
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
        self.coord_loss_multiplier = coord_loss_multiplier
        self.class_loss_multiplier = class_loss_multiplier
        self.confidence_loss_multiplier = confidence_loss_multiplier
        self.se_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        x: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Runs a forward pass through this YOLO detection layer.

        Maps cell-local coordinates to global coordinates in the [0, 1] range, scales the bounding
        boxes with the anchors, converts the center coordinates to corner coordinates, and maps
        probabilities to ]0, 1[ range using sigmoid.

        Args:
            x : The output from the previous layer. Tensor of size
                `[batch_size, boxes_per_cell * (num_classes + 5), height, width]`.
            targets: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.

        Returns:
            result: Layer output, and if training targets were provided, a dictionary of losses.
                Layer output is sized `[batch_size, num_anchors * height * width, num_classes + 5]`.
        """
        batch_size, num_features, height, width = x.shape
        num_attrs = self.num_classes + 5
        boxes_per_cell = num_features // num_attrs
        if boxes_per_cell != len(self.anchor_ids):
            raise MisconfigurationException(
                "The model predicts {} bounding boxes per cell, but {} anchor boxes are defined "
                "for this layer.".format(boxes_per_cell, len(self.anchor_ids)))

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
        corners = self._corner_coordinates(image_xy, image_wh)
        output = torch.cat((corners, confidence.unsqueeze(-1), classprob), -1)
        output = output.reshape(batch_size, height * width * boxes_per_cell, num_attrs)

        if targets is None:
            return output
        else:
            np_mask = self._no_prediction_mask(corners, targets)
            losses = self._calculate_losses(xy, wh, confidence, classprob, targets, np_mask)
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

    def _corner_coordinates(self, xy, wh):
        """
        Converts box center points and sizes to corner coordinates.

        Args:
            xy (Tensor): Center coordinates. Tensor of size `[..., 2]`.
            wh (Tensor): Width and height. Tensor of size `[..., 2]`.

        Returns:
            corners (Tensor): A matrix of (x1, y1, x2, y2) coordinates.
        """
        half_wh = wh / 2
        top_left = xy - half_wh
        bottom_right = xy + half_wh
        return torch.cat((top_left, bottom_right), -1)

    def _no_prediction_mask(self, preds, targets):
        """
        Initializes the mask that will be used to select predictors that are not responsible for
        predicting any target. The value will be `True`, unless the predicted box overlaps any
        target significantly (IoU greater than `self.ignore_threshold`).

        Args:
            preds (Tensor): The predicted corner coordinates, normalized to the [0, 1] range.
                Tensor of size `[batch_size, height, width, boxes_per_cell, 4]`.
            targets (List[Dict[str, Tensor]]): List of dictionaries of target values, one
                dictionary for each image.

        Returns:
            results (Tensor): A boolean tensor shaped `[batch_size, height, width, boxes_per_cell]`
                with `False` where the predicted box overlaps a target significantly and `True`
                elsewhere.
        """
        shape = preds.shape
        preds = preds.view(shape[0], -1, shape[-1])

        scale = torch.tensor([self.image_width,
                              self.image_height,
                              self.image_width,
                              self.image_height],
                             device=preds.device)
        preds = preds * scale

        results = torch.ones(preds.shape[:-1], dtype=torch.bool, device=preds.device)
        for image_idx, (image_preds, image_targets) in enumerate(zip(preds, targets)):
            target_boxes = image_targets['boxes']
            if target_boxes.shape[0] > 0:
                ious = box_iou(image_preds, target_boxes)
                best_ious = ious.max(-1).values
                results[image_idx] = best_ious <= self.ignore_threshold
        results = results.view(shape[:-1])
        return results

    def _calculate_losses(self, xy, wh, confidence, classprob, targets, np_mask):
        """
        From the targets that are in the image space calculates the actual targets for the network
        predictions, and returns a dictionary of training losses.

        Args:
            xy (Tensor): The predicted center coordinates before scaling. Values from zero to one
                in a tensor sized `[batch_size, height, width, boxes_per_cell, 2]`.
            wh (Tensor): The unnormalized width and height predictions. Tensor of size
                `[batch_size, height, width, boxes_per_cell, 2]`.
            confidence (Tensor): The confidence predictions, normalized to [0, 1]. A tensor sized
                `[batch_size, height, width, boxes_per_cell]`.
            classprob (Tensor): The class probability predictions, normalized to [0, 1]. A tensor
                sized `[batch_size, height, width, boxes_per_cell, num_classes]`.
            targets (List[Dict[str, Tensor]]): List of dictionaries of target values, one
                dictionary for each image.
            np_mask: A boolean mask containing `True` where the predicted box does not overlap any
                target significantly.

        Returns:
            predicted (Dict[str, Tensor]): A dictionary of training losses.
        """
        batch_size, height, width, boxes_per_cell, _ = xy.shape
        device = xy.device
        assert batch_size == len(targets)

        # Divisor for converting targets from image coordinates to feature map coordinates
        image_to_feature_map = torch.tensor([self.image_width / width,
                                             self.image_height / height],
                                            device=device)
        # Divisor for converting targets from image coordinates to [0, 1] range
        image_to_unit = torch.tensor([self.image_width, self.image_height],
                                     device=device)

        anchor_wh = torch.tensor(self.anchor_dims, dtype=wh.dtype, device=device)
        anchor_map = torch.tensor(self.anchor_map, dtype=torch.int64, device=device)

        # List of predicted and target values for the predictors that are responsible for
        # predicting a target.
        target_xy = []
        target_wh = []
        target_label = []
        size_compensation = []
        pred_xy = []
        pred_wh = []
        pred_classprob = []
        pred_confidence = []

        for image_idx, image_targets in enumerate(targets):
            boxes = image_targets['boxes']
            if boxes.shape[0] < 1:
                continue

            # Bounding box corner coordinates are converted to center coordinates, width, and
            # height.
            box_wh = boxes[:, 2:4] - boxes[:, 0:2]
            box_xy = boxes[:, 0:2] + (box_wh / 2)

            # The center coordinates are converted to the feature map dimensions so that the whole
            # number tells the cell index and the fractional part tells the location inside the cell.
            box_xy = box_xy / image_to_feature_map
            cell_i = box_xy[:, 0].to(torch.int64).clamp(0, width - 1)
            cell_j = box_xy[:, 1].to(torch.int64).clamp(0, height - 1)

            # We want to know which anchor box overlaps a ground truth box more than any other
            # anchor box. We know that the anchor box is located in the same grid cell as the
            # ground truth box. For each prior shape (width, height), we calculate the IoU with
            # all ground truth boxes, assuming the boxes are at the same location. Then for each
            # target, we select the prior shape that gives the highest IoU.
            ious = _aligned_iou(box_wh, anchor_wh)
            best_anchors = ious.max(1).indices

            # `anchor_map` maps the anchor indices to the predictors in this layer, or to -1 if
            # it's not an anchor of this layer. We ignore the predictions if the best anchor is in
            # another layer.
            predictors = anchor_map[best_anchors]
            selected = predictors >= 0
            box_xy = box_xy[selected]
            box_wh = box_wh[selected]
            cell_i = cell_i[selected]
            cell_j = cell_j[selected]
            predictors = predictors[selected]
            best_anchors = best_anchors[selected]

            # The "no-prediction" mask is used to select predictors that are not responsible for
            # predicting any object for calculating the confidence loss.
            np_mask[image_idx, cell_j, cell_i, predictors] = False

            # Bounding box targets
            relative_xy = box_xy - box_xy.floor()
            relative_wh = torch.log(box_wh / anchor_wh[best_anchors] + 1e-16)
            target_xy.append(relative_xy)
            target_wh.append(relative_wh)

            # Size compensation factor for bounding box loss
            unit_wh = box_wh / image_to_unit
            size_compensation.append(2 - (unit_wh[:, 0] * unit_wh[:, 1]))

            # The data may contain a different number of classes than this detection layer. In case
            # a label is greater than the number of classes that this layer predicts, it will be
            # mapped to the last class.
            labels = image_targets['labels']
            labels = labels[selected]
            labels = torch.minimum(labels, torch.tensor(self.num_classes - 1, device=device))
            target_label.append(labels)

            pred_xy.append(xy[image_idx, cell_j, cell_i, predictors])
            pred_wh.append(wh[image_idx, cell_j, cell_i, predictors])
            pred_classprob.append(classprob[image_idx, cell_j, cell_i, predictors])
            pred_confidence.append(confidence[image_idx, cell_j, cell_i, predictors])

        losses = dict()

        if pred_xy and pred_wh and target_xy and target_wh:
            size_compensation = torch.cat(size_compensation).unsqueeze(1)
            pred_xy = torch.cat(pred_xy)
            target_xy = torch.cat(target_xy)
            location_loss = self.se_loss(pred_xy, target_xy)
            location_loss = location_loss * size_compensation
            location_loss = location_loss.sum() / batch_size
            losses['location'] = location_loss * self.coord_loss_multiplier

            pred_wh = torch.cat(pred_wh)
            target_wh = torch.cat(target_wh)
            size_loss = self.se_loss(pred_wh, target_wh)
            size_loss = size_loss * size_compensation
            size_loss = size_loss.sum() / batch_size
            losses['size'] = size_loss * self.coord_loss_multiplier

        class_loss = None
        if pred_classprob and target_label:
            pred_classprob = torch.cat(pred_classprob)
            target_label = torch.cat(target_label)
            target_classprob = torch.nn.functional.one_hot(target_label, self.num_classes)
            target_classprob = target_classprob.to(dtype=pred_classprob.dtype)
            class_loss = self.se_loss(pred_classprob, target_classprob)
            class_loss = class_loss.sum() / batch_size
            losses['class'] = class_loss * self.class_loss_multiplier

        np_confidence = confidence[np_mask]
        np_target_confidence = torch.zeros_like(np_confidence)
        np_confidence_loss = self.se_loss(np_confidence, np_target_confidence)
        np_confidence_loss = np_confidence_loss.sum() / batch_size
        losses['np_confidence'] = np_confidence_loss * self.confidence_loss_multiplier

        if pred_confidence:
            p_confidence = torch.cat(pred_confidence)
            p_target_confidence = torch.ones_like(p_confidence)
            p_confidence_loss = self.se_loss(p_confidence, p_target_confidence)
            p_confidence_loss = p_confidence_loss.sum() / batch_size
            losses['p_confidence'] = p_confidence_loss * self.confidence_loss_multiplier

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
        chunks = [torch.chunk(outputs[layer], self.num_chunks, dim=1)[self.chunk_idx]
                  for layer in self.source_layers]
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
