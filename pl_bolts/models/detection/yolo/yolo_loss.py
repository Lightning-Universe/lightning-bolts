import math
from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import box_iou, generalized_box_iou


def _upcast(t: Tensor) -> Tensor:
    """Protects from numerical overflows in multiplications by upcasting to the equivalent higher type."""
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def complete_iou(boxes1: Tensor, boxes2: Tensor, distance_only: bool = False) -> Tensor:
    """Returns the complete intersection-over-union between two sets of boxes. Both sets of boxes are expected to
    be in `(x1, y1, x2, y2)` format.

    Args:
        boxes1: Box coordinates in a tensor of size ``[N, 4]``.
        boxes2: Box coordinates in a tensor of size ``[M, 4]``.
        distance_only: If set to ``True``, returns the Distance IoU.

    Returns:
        A matrix containing the `NxM` complete IoU values between boxes from ``boxes1`` and ``boxes2``.
    """

    # Degenerate boxes give inf / nan results, so do an early check.
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou = box_iou(boxes1, boxes2)

    boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
    boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]

    lti = torch.min(boxes1[..., :2], boxes2[..., :2])
    rbi = torch.max(boxes1[..., 2:], boxes2[..., 2:])

    whi = _upcast(rbi - lti).clamp(min=0)  # [N, M, 2]
    wi = whi[..., 0]
    hi = whi[..., 1]
    sqr_length = wi * wi + hi * hi  # [N, M]

    wh1 = boxes1[..., 2:] - boxes1[..., :2]
    wh2 = boxes2[..., 2:] - boxes2[..., :2]
    center1 = boxes1[..., :2] + (wh1 / 2)
    center2 = boxes2[..., :2] + (wh2 / 2)
    offset = center2 - center1  # [N, M, 2]
    dx = offset[..., 0]
    dy = offset[..., 1]
    sqr_distance = dx * dx + dy * dy  # [N, M]

    diou = torch.where(sqr_length > 0.0, iou - (sqr_distance / sqr_length), iou)
    if distance_only:
        return diou

    w1 = wh1[..., 0]
    h1 = wh1[..., 1]
    w2 = wh2[..., 0]
    h2 = wh2[..., 1]
    daspect = torch.atan(w2 / h2) - torch.atan(w1 / h1)  # [N, M]
    aspect_loss = 4 / (math.pi * math.pi) * (daspect * daspect)

    with torch.no_grad():
        alpha = aspect_loss / (1 - iou + aspect_loss + 1e-6)

    return diou - (alpha * aspect_loss)


def distance_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    return complete_iou(boxes1, boxes2, distance_only=True)


class LossFunction:
    """A class for calculating the YOLO losses from predictions and targets.

    Args:
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a tensor with as many elements as there are input boxes. Valid values for a string are
            "iou", "giou", "diou", and "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that target
            confidence is one if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
    """

    def __init__(
        self,
        overlap_func: Union[str, Callable] = "ciou",
        predict_overlap: Optional[float] = None,
        overlap_multiplier: float = 1.0,
        confidence_multiplier: float = 1.0,
        class_multiplier: float = 1.0,
    ):
        if overlap_func == "iou":
            self.overlap_func = box_iou
        elif overlap_func == "giou":
            self.overlap_func = generalized_box_iou
        elif overlap_func == "diou":
            self.overlap_func = distance_iou
        elif overlap_func == "ciou":
            self.overlap_func = complete_iou
        elif callable(overlap_func):
            self.overlap_func = overlap_func
        else:
            raise ValueError(f"Overlap function type `{overlap_func}´ is unknown.")

        self.predict_overlap = predict_overlap

        self.overlap_multiplier = overlap_multiplier
        self.confidence_multiplier = confidence_multiplier
        self.class_multiplier = class_multiplier

    def _calculate_overlap(
        self, preds: Tensor, targets: Tensor, image_size: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Calculates the overlap and overlap loss.

        The overlap is calculated using ``self.overlap_func``. Overlap loss is ``1 - overlap``. If ``image_size`` is
        given, the loss is scaled by a factor that is large for small boxes (the maximum value is 2) and small for large
        boxes (the minimum value is 1).

        Args:
            preds: An ``[N, 4]`` matrix of predicted `(x1, y1, x2, y2)` coordinates.
            targets: An ``[M, 4]`` matrix of target `(x1, y1, x2, y2)` coordinates.
            image_size: If given,

        Returns:
            overlap, overlap_loss: Two ``[M, N]`` matrices: the overlap and the overlap loss between all combinations of
                a target and a prediction.
        """
        overlap = self.overlap_func(targets, preds)
        overlap_loss = 1.0 - overlap
        if image_size is not None:
            unit_wh = targets[:, 2:] / image_size
            size_compensation = 2 - (unit_wh[:, 0] * unit_wh[:, 1])
            overlap_loss = overlap_loss * size_compensation
        return overlap, overlap_loss

    def _calculate_confidence(self, preds: Tensor, overlap: Tensor, bce_func: Callable):
        """Calculates the confidence loss for foreground anchors.

        If ``self.predict_overlap`` is ``True``, ``overlap`` will be used as the target confidence. Otherwise the target
        confidence is 1. The method returns a matrix of losses for target/prediction pairs.

        Args:
            preds: An ``[N]`` vector of predicted confidences.
            overlap: An ``[M, N]`` matrix of the overlap between all combinations of a target bounding box and a
                predicted bounding box.
            bce_func: A function for calculating binary cross entropy.

        Returns:
            An ``[M, N]`` matrix of confidence loss between all combinations of a target and a prediction.
        """
        if self.predict_overlap is not None:
            # When predicting overlap, target confidence is different for each pair of a prediction and a target. The
            # tensors have to be broadcasted to [M, N].
            preds = preds.unsqueeze(0)
            preds = torch.broadcast_to(preds, overlap.shape)
            targets = torch.ones_like(preds) - self.predict_overlap
            # Distance-IoU may return negative "overlaps", so we have to make sure that the targets are not negative.
            targets = targets + (self.predict_overlap * overlap.detach().clamp(min=0))
        else:
            targets = torch.ones_like(preds)

        result = bce_func(preds, targets, reduction="none")

        if result.ndim == 1:
            # When not predicting overlap, target confidence is the same for every target, but we should still return a
            # matrix.
            result = result.unsqueeze(0)
            torch.broadcast_to(result, overlap.shape)

        return result

    def _calculate_bg_confidence(self, preds: Tensor, bce_func: Callable):
        """Calculates the confidence loss for background anchors."""
        targets = torch.zeros_like(preds)
        return bce_func(preds, targets, reduction="none")

    def _calculate_class(self, preds: Tensor, targets: Tensor, bce_func: Callable) -> Tensor:
        """Calculates the classification losses.

        If ``targets`` is a vector of class labels, converts it to a matrix of one-hot class probabilities. Then
        calculates the classification losses between the predictions and the targets. If ``all_pairs`` is ``True``,
        returns a matrix of losses between all combinations of a target and a prediction.

        Args:
            preds: An ``[N, C]`` matrix of predicted class probabilities.
            targets: An ``[M, C]`` matrix of target class probabilities or an ``[M]`` vector of class labels.
            bce_func: A function for calculating binary cross entropy.

        Returns:
            An ``[M, N]`` matrix of losses between all combinations of a target and a prediction.
        """
        num_classes = preds.shape[-1]
        if targets.ndim == 1:
            # The data may contain a different number of classes than what the model predicts. In case a label is
            # greater than the number of predicted classes, it will be mapped to the last class.
            last_class = torch.tensor(num_classes - 1, device=targets.device)
            targets = torch.min(targets, last_class)
            targets = torch.nn.functional.one_hot(targets, num_classes)
        elif targets.shape[-1] != num_classes:
            raise ValueError(
                f"The number of classes in the data ({targets.shape[-1]}) doesn't match the number of classes "
                f"predicted by the model ({num_classes})."
            )
        targets = targets.to(dtype=preds.dtype)

        preds = preds.unsqueeze(0)  # [1, preds, classes]
        targets = targets.unsqueeze(1)  # [targets, 1, classes]
        preds, targets = torch.broadcast_tensors(preds, targets)
        return bce_func(preds, targets, reduction="none").sum(-1)

    def __call__(self, preds, targets, input_is_normalized: bool, image_size: Optional[Tensor] = None):
        """Calculates the losses for all pairs of a predictions and a target, and if `bg_confidences` appears in
        ``preds``, calculates the confidence loss for background predictions.

        This method is called before taking the final losses using ``sums()``, and for obtaining costs for SimOTA
        matching.

        Args:
            preds: A dictionary of predictions, containing "boxes", "confidences", and "classprobs".
            targets: A dictionary of training targets, containing "boxes" and "labels".
            input_is_normalized: If ``False``, input is logits, if ``True``, input is normalized to `0..1`.
            image_size: Width and height in a vector that defines the scale of the target coordinates.
        """
        bce_func = binary_cross_entropy if input_is_normalized else binary_cross_entropy_with_logits

        overlap, overlap_loss = self._calculate_overlap(preds["boxes"], targets["boxes"], image_size)
        self.overlap = overlap
        self.overlap_loss = overlap_loss * self.overlap_multiplier

        confidence_loss = self._calculate_confidence(preds["confidences"], overlap, bce_func)
        self.confidence_loss = confidence_loss * self.confidence_multiplier

        if "bg_confidences" in preds:
            bg_confidence_loss = self._calculate_bg_confidence(preds["bg_confidences"], bce_func)
            self.bg_confidence_loss = bg_confidence_loss * self.confidence_multiplier

        class_loss = self._calculate_class(preds["classprobs"], targets["labels"], bce_func)
        self.class_loss = class_loss * self.class_multiplier

    def sums(self):
        """Returns the sums of the losses over prediction/target pairs, assuming the predictions and targets have
        been matched (there are as many predictions and targets)."""
        overlap_loss = self.overlap_loss.diagonal().sum()
        confidence_loss = self.confidence_loss.diagonal().sum() + self.bg_confidence_loss.sum()
        class_loss = self.class_loss.diagonal().sum()
        return overlap_loss, confidence_loss, class_loss
