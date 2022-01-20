from typing import Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn
from torchvision.ops import box_convert

from pl_bolts.models.detection.yolo.utils import global_xy
from pl_bolts.models.detection.yolo.yolo_loss import LossFunction
from pl_bolts.utils import _TORCHVISION_AVAILABLE


class DetectionLayer(nn.Module):
    """A YOLO detection layer.

    A YOLO model has usually 1 - 3 detection layers at different
    resolutions. The loss should be summed from all of them.
    """

    def __init__(
        self,
        num_classes: int,
        anchor_dims: List[Tuple[int, int]],
        matching_func: Callable,
        loss_func: LossFunction,
        xy_scale: float = 1.0,
        input_is_normalized: bool = False,
    ) -> None:
        """
        Args:
            num_classes: Number of different classes that this layer predicts.
            anchor_dims: A list of the anchor box dimensions for this layer. The list should
                contain (width, height) tuples in the network input resolution (relative to the
                width and height defined in the configuration file).
            matching_func: The matching algorithm to be used for assigning targets to anchors.
            loss_func: ``LossFunction`` object for calculating the losses.
            xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor.
                Using a value > 1.0 helps to produce coordinate values close to one.
            input_is_normalized: The input is normalized by logistic activation in the previous
                layer. In this case the detection layer will not take the sigmoid of the coordinate
                and probability predictions, and the width and height are scaled up so that the
                maximum value is four times the anchor dimension
        """
        super().__init__()

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("YOLO model uses `torchvision`, which is not installed yet.")

        self.num_classes = num_classes
        self.anchor_dims = anchor_dims
        self.matching_func = matching_func
        self.loss_func = loss_func
        self.xy_scale = xy_scale
        self.input_is_normalized = input_is_normalized

    def forward(self, x: Tensor, image_size: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tensor:
        """Runs a forward pass through this YOLO detection layer.

        Maps cell-local coordinates to global coordinates in the image space, scales the bounding boxes with the
        anchors, converts the center coordinates to corner coordinates, and maps probabilities to the `]0, 1[` range
        using sigmoid.

        If targets are given, computes also losses from the predictions and the targets. This layer is responsible only
        for the targets that best match one of the anchors assigned to this layer. Training losses will be saved to the
        ``losses`` attribute. ``hits`` attribute will be set to the number of targets that this layer was responsible
        for. ``losses`` is a tensor of three elements: the overlap, confidence, and classification loss.

        Args:
            x: The output from the previous layer. Tensor of size
                ``[batch_size, boxes_per_cell * (num_classes + 5), height, width]``.
            image_size: Image width and height in a vector (defines the scale of the predicted and target coordinates).
            targets: If set, computes losses from detection layers against these targets. A list of target dictionaries,
                one for each image.

        Returns:
            Layer output tensor, sized ``[batch_size, num_anchors * height * width, num_classes + 5]``.
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
        # probabilities, unless the input is normalized by the previous layer activation. Confidence
        # and class losses use the unnormalized values if possible.
        norm_x = x if self.input_is_normalized else torch.sigmoid(x)
        xy = norm_x[..., :2]
        wh = x[..., 2:4]
        confidence = x[..., 4]
        classprob = x[..., 5:]
        norm_confidence = norm_x[..., 4]
        norm_classprob = norm_x[..., 5:]

        # Eliminate grid sensitivity. The previous layer should output extremely high values for
        # the sigmoid to produce x/y coordinates close to one. YOLOv4 solves this by scaling the
        # x/y coordinates.
        xy = xy * self.xy_scale - 0.5 * (self.xy_scale - 1)

        image_xy = global_xy(xy, image_size)
        if self.input_is_normalized:
            image_wh = 4 * torch.square(wh) * torch.tensor(self.anchor_dims, dtype=wh.dtype, device=wh.device)
        else:
            image_wh = torch.exp(wh) * torch.tensor(self.anchor_dims, dtype=wh.dtype, device=wh.device)
        box = torch.cat((image_xy, image_wh), -1)
        box = box_convert(box, in_fmt="cxcywh", out_fmt="xyxy")
        output = torch.cat((box, norm_confidence.unsqueeze(-1), norm_classprob), -1)
        output = output.reshape(batch_size, height * width * boxes_per_cell, num_attrs)

        if targets is not None:
            # We want to use binary_cross_entropy_with_logits, so we'll use the unnormalized confidence and classprob,
            # if possible.
            preds = [{"boxes": b, "confidences": c, "classprobs": p} for b, c, p in zip(box, confidence, classprob)]
            self._calculate_losses(preds, targets, image_size)

        return output

    def _calculate_losses(
        self,
        preds: List[Dict[str, Tensor]],
        targets: List[Dict[str, Tensor]],
        image_size: Tensor,
    ):
        """Matches the predictions to targets and calculates the losses. Creates the attributes ``losses`` and
        ``hits``. ``losses`` is a tensor of three elements: the overlap, confidence, and classification loss.
        ``hits`` is the number of targets that this layer was responsible for.

        Args:
            preds: List of predictions for each image.
            targets: List of training targets for each image.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
        """
        batch_size = len(preds)
        if batch_size != len(targets):
            raise ValueError("Different batch size for predictions and targets.")

        matches = []
        for image_preds, image_targets in zip(preds, targets):
            if image_targets["boxes"].shape[0] > 0:
                matched_preds, matched_targets = self.matching_func(image_preds, image_targets, image_size)
            else:
                device = image_preds["confidences"].device
                matched_preds = {
                    "boxes": torch.empty((0, 4), device=device),
                    "confidences": torch.empty(0, device=device),
                    "bg_confidences": image_preds["confidences"].flatten(),
                    "classprobs": torch.empty((0, self.num_classes), device=device),
                }
                matched_targets = {
                    "boxes": torch.empty((0, 4), device=device),
                    "labels": torch.empty(0, dtype=torch.int64, device=device),
                }
            matches.append((matched_preds, matched_targets))

        matched_preds = {
            "boxes": torch.cat(tuple(m[0]["boxes"] for m in matches)),
            "confidences": torch.cat(tuple(m[0]["confidences"] for m in matches)),
            "bg_confidences": torch.cat(tuple(m[0]["bg_confidences"] for m in matches)),
            "classprobs": torch.cat(tuple(m[0]["classprobs"] for m in matches)),
        }
        matched_targets = {
            "boxes": torch.cat(tuple(m[1]["boxes"] for m in matches)),
            "labels": torch.cat(tuple(m[1]["labels"] for m in matches)),
        }
        self.loss_func(matched_preds, matched_targets, self.input_is_normalized, image_size)
        overlap_loss, confidence_loss, class_loss = self.loss_func.sums()
        self.losses = torch.stack((overlap_loss, confidence_loss, class_loss)) / batch_size
        self.hits = len(matched_targets["boxes"])


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


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
