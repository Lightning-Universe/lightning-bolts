from copy import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, optim

# It seems to be impossible to avoid mypy errors if using import instead of getattr().
# See https://github.com/python/mypy/issues/8823
try:
    LRScheduler: Any = getattr(optim.lr_scheduler, "LRScheduler")
except AttributeError:
    LRScheduler = getattr(optim.lr_scheduler, "_LRScheduler")

from pl_bolts.datamodules import VOCDetectionDataModule
from pl_bolts.datamodules.vocdetection_datamodule import Compose
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils import _TORCHMETRICS_DETECTION_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

from .darknet_network import DarknetNetwork
from .torch_networks import YOLOV4Network
from .types import BATCH, IMAGES, PRED, TARGET, TARGETS

if _TORCHMETRICS_DETECTION_AVAILABLE:
    try:
        from torchmetrics.detection import MeanAveragePrecision

        _MEAN_AVERAGE_PRECISION_AVAILABLE = True
    except ImportError:
        _MEAN_AVERAGE_PRECISION_AVAILABLE = False
else:
    _MEAN_AVERAGE_PRECISION_AVAILABLE = False

if _TORCHVISION_AVAILABLE:
    from torchvision.ops import batched_nms
    from torchvision.transforms import functional as T  # noqa: N812
else:
    warn_missing_pkg("torchvision")


class YOLO(LightningModule):
    """PyTorch Lightning implementation of YOLO that supports the most important features of YOLOv3, YOLOv4, YOLOv5,
    YOLOv7, Scaled-YOLOv4, and YOLOX.

    *YOLOv3 paper*: `Joseph Redmon and Ali Farhadi <https://arxiv.org/abs/1804.02767>`__

    *YOLOv4 paper*: `Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2004.10934>`__

    *YOLOv7 paper*: `Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2207.02696>`__

    *Scaled-YOLOv4 paper*: `Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao
    <https://arxiv.org/abs/2011.08036>`__

    *YOLOX paper*: `Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun <https://arxiv.org/abs/2107.08430>`__

    *Implementation*: `Seppo Enarvi <https://github.com/senarvi>`__

    The network architecture can be written in PyTorch, or read from a Darknet configuration file using the
    :class:`~.darknet_network.DarknetNetwork` class. ``DarknetNetwork`` is also able to read weights that have been
    saved by Darknet. See the :class:`~.yolo_module.CLIYOLO` command-line application for an example of how to specify
    a network architecture.

    The input is expected to be a list of images. Each image is a tensor with shape ``[channels, height, width]``. The
    images from a single batch will be stacked into a single tensor, so the sizes have to match. Different batches can
    have different image sizes, as long as the size is divisible by the ratio in which the network downsamples the
    input.

    During training, the model expects both the image tensors and a list of targets. It's possible to train a model
    using one integer class label per box, but the YOLO model supports also multiple labels per box. For multi-label
    training, simply use a boolean matrix that indicates which classes are assigned to which boxes, in place of the
    class labels. *Each target is a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
    - labels (``Int64Tensor[N]`` or ``BoolTensor[N, classes]``): the class label or a boolean class mask for each
      ground-truth box

    :func:`~.yolo_module.YOLO.forward` method returns all predictions from all detection layers in one tensor with shape
    ``[N, anchors, classes + 5]``, where ``anchors`` is the total number of anchors in all detection layers. The
    coordinates are scaled to the input image size. During training it also returns a dictionary containing the
    classification, box overlap, and confidence losses.

    During inference, the model requires only the image tensor. :func:`~.yolo_module.YOLO.infer` method filters and
    processes the predictions. If a prediction has a high score for more than one class, it will be duplicated. *The
    processed output is returned in a dictionary containing the following tensors*:

    - boxes (``FloatTensor[N, 4]``): predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
    - scores (``FloatTensor[N]``): detection confidences
    - labels (``Int64Tensor[N]``): the predicted labels for each object

    Args:
        network: A module that represents the network layers. This can be obtained from a Darknet configuration using
            :func:`~.darknet_network.DarknetNetwork`, or it can be defined as PyTorch code.
        optimizer: Which optimizer class to use for training.
        optimizer_params: Parameters to pass to the optimizer constructor. Weight decay will be applied only to
            convolutional layer weights.
        lr_scheduler: Which learning rate scheduler class to use for training.
        lr_scheduler_params: Parameters to pass to the learning rate scheduler constructor.
        confidence_threshold: Postprocessing will remove bounding boxes whose confidence score is not higher than this
            threshold.
        nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher confidence box is
            higher than this threshold, if the predicted categories are equal.
        detections_per_image: Keep at most this number of highest-confidence detections per image.

    """

    def __init__(
        self,
        network: nn.Module,
        optimizer: Type[optim.Optimizer] = optim.SGD,
        optimizer_params: Optional[Dict[str, Any]] = None,
        lr_scheduler: Type[LRScheduler] = LinearWarmupCosineAnnealingLR,
        lr_scheduler_params: Optional[Dict[str, Any]] = None,
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        detections_per_image: int = 300,
    ) -> None:
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError("YOLO model uses `torchvision`, which is not installed yet.")  # pragma: no-cover

        self.network = network
        self.optimizer_class = optimizer
        if optimizer_params is not None:
            self.optimizer_params = optimizer_params
        else:
            self.optimizer_params = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005}
        self.lr_scheduler_class = lr_scheduler
        if lr_scheduler_params is not None:
            self.lr_scheduler_params = lr_scheduler_params
        else:
            self.lr_scheduler_params = {"warmup_epochs": 5, "max_epochs": 300, "warmup_start_lr": 0.0}
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detections_per_image = detections_per_image

        if _MEAN_AVERAGE_PRECISION_AVAILABLE:
            self._val_map = MeanAveragePrecision()
            self._test_map = MeanAveragePrecision()

    def forward(
        self, images: Union[Tensor, IMAGES], targets: Optional[TARGETS] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Runs a forward pass through the network (all layers listed in ``self.network``), and if training targets
        are provided, computes the losses from the detection layers.

        Detections are concatenated from the detection layers. Each detection layer will produce a number of detections
        that depends on the size of the feature map and the number of anchors per feature map cell.

        Args:
            images: A tensor of size ``[batch_size, channels, height, width]`` containing a batch of images or a list of
                image tensors.
            targets: If given, computes losses from detection layers against these targets. A list of target
                dictionaries, one for each image.

        Returns:
            detections (:class:`~torch.Tensor`), losses (:class:`~torch.Tensor`): Detections, and if targets were
            provided, a dictionary of losses. Detections are shaped ``[batch_size, anchors, classes + 5]``, where
            ``anchors`` is the feature map size (width * height) times the number of anchors per cell. The predicted box
            coordinates are in `(x1, y1, x2, y2)` format and scaled to the input image size.
        """
        self.validate_batch(images, targets)
        images_tensor = images if isinstance(images, Tensor) else torch.stack(images)
        detections, losses, hits = self.network(images_tensor, targets)

        detections = torch.cat(detections, 1)
        if targets is None:
            return detections

        total_hits = sum(hits)
        for layer_idx, layer_hits in enumerate(hits):
            hit_rate: Union[Tensor, float] = torch.true_divide(layer_hits, total_hits) if total_hits > 0 else 1.0
            self.log(f"layer_{layer_idx}_hit_rate", hit_rate, sync_dist=True, batch_size=len(images))

        losses = torch.stack(losses).sum(0)
        return detections, losses

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[LRScheduler]]:
        """Constructs the optimizer and learning rate scheduler based on ``self.optimizer_params`` and
        ``self.lr_scheduler_params``.

        If weight decay is specified, it will be applied only to convolutional layer weights, as they contain much more
        parameters than the biases and batch normalization parameters. Regularizing all parameters could lead to
        underfitting.
        """
        if ("weight_decay" in self.optimizer_params) and (self.optimizer_params["weight_decay"] != 0):
            defaults = copy(self.optimizer_params)
            weight_decay = defaults.pop("weight_decay")

            default_group = []
            wd_group = []
            for name, tensor in self.named_parameters():
                if not tensor.requires_grad:
                    continue
                if name.endswith(".conv.weight"):
                    wd_group.append(tensor)
                else:
                    default_group.append(tensor)

            params = [
                {"params": default_group, "weight_decay": 0.0},
                {"params": wd_group, "weight_decay": weight_decay},
            ]
            optimizer = self.optimizer_class(params, **defaults)
        else:
            optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: BATCH, batch_idx: int) -> STEP_OUTPUT:
        """Computes the training loss.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        Returns:
            A dictionary that includes the training loss in 'loss'.

        """
        images, targets = batch
        _, losses = self(images, targets)

        self.log("train/overlap_loss", losses[0], prog_bar=True, sync_dist=True)
        self.log("train/confidence_loss", losses[1], prog_bar=True, sync_dist=True)
        self.log("train/class_loss", losses[2], prog_bar=True, sync_dist=True)
        self.log("train/total_loss", losses.sum(), sync_dist=True)

        return {"loss": losses.sum()}

    def validation_step(self, batch: BATCH, batch_idx: int) -> Optional[STEP_OUTPUT]:
        """Evaluates a batch of data from the validation set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        """
        images, targets = batch
        detections, losses = self(images, targets)

        self.log("val/overlap_loss", losses[0], sync_dist=True, batch_size=len(images))
        self.log("val/confidence_loss", losses[1], sync_dist=True, batch_size=len(images))
        self.log("val/class_loss", losses[2], sync_dist=True, batch_size=len(images))
        self.log("val/total_loss", losses.sum(), sync_dist=True, batch_size=len(images))

        if _MEAN_AVERAGE_PRECISION_AVAILABLE:
            detections = self.process_detections(detections)
            targets = self.process_targets(targets)
            self._val_map.update(detections, targets)

    def on_validation_epoch_end(self) -> None:
        # When continuing training from a checkpoint, it may happen that epoch_end is called without detections. In this
        # case the metrics cannot be computed.
        if (not _MEAN_AVERAGE_PRECISION_AVAILABLE) or (not self._val_map.detections):
            return

        map_scores = self._val_map.compute()
        map_scores = {"val/" + k: v for k, v in map_scores.items()}
        self.log_dict(map_scores, sync_dist=True)
        self._val_map.reset()

    def test_step(self, batch: BATCH, batch_idx: int) -> Optional[STEP_OUTPUT]:
        """Evaluates a batch of data from the test set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.

        """
        images, targets = batch
        detections, losses = self(images, targets)

        self.log("test/overlap_loss", losses[0], sync_dist=True)
        self.log("test/confidence_loss", losses[1], sync_dist=True)
        self.log("test/class_loss", losses[2], sync_dist=True)
        self.log("test/total_loss", losses.sum(), sync_dist=True)

        if _MEAN_AVERAGE_PRECISION_AVAILABLE:
            detections = self.process_detections(detections)
            targets = self.process_targets(targets)
            self._test_map.update(detections, targets)

    def on_test_epoch_end(self) -> None:
        # When continuing training from a checkpoint, it may happen that epoch_end is called without detections. In this
        # case the metrics cannot be computed.
        if (not _MEAN_AVERAGE_PRECISION_AVAILABLE) or (not self._test_map.detections):
            return

        map_scores = self._test_map.compute()
        map_scores = {"test/" + k: v for k, v in map_scores.items()}
        self.log_dict(map_scores, sync_dist=True)
        self._test_map.reset()

    def predict_step(self, batch: BATCH, batch_idx: int, dataloader_idx: int = 0) -> List[PRED]:
        """Feeds a batch of images to the network and returns the detected bounding boxes, confidence scores, and class
        labels.

        If a prediction has a high score for more than one class, it will be duplicated.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors. Targets is a list of target
                dictionaries.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            A list of dictionaries containing tensors "boxes", "scores", and "labels". "boxes" is a matrix of detected
            bounding box `(x1, y1, x2, y2)` coordinates. "scores" is a vector of confidence scores for the bounding box
            detections. "labels" is a vector of predicted class labels.

        """
        images, _ = batch
        detections = self(images)
        return self.process_detections(detections)

    def infer(self, image: Tensor) -> PRED:
        """Feeds an image to the network and returns the detected bounding boxes, confidence scores, and class labels.

        If a prediction has a high score for more than one class, it will be duplicated.

        Args:
            image: An input image, a tensor of uint8 values sized ``[channels, height, width]``.

        Returns:
            A dictionary containing tensors "boxes", "scores", and "labels". "boxes" is a matrix of detected bounding
            box `(x1, y1, x2, y2)` coordinates. "scores" is a vector of confidence scores for the bounding box
            detections. "labels" is a vector of predicted class labels.

        """
        if not isinstance(image, Tensor):
            image = T.to_tensor(image)

        was_training = self.training
        self.eval()

        detections = self([image])
        detections = self.process_detections(detections)
        detections = detections[0]

        if was_training:
            self.train()
        return detections

    def process_detections(self, preds: Tensor) -> List[PRED]:
        """Splits the detection tensor returned by a forward pass into a list of prediction dictionaries, and filters
        them based on confidence threshold, non-maximum suppression (NMS), and maximum number of predictions.

        If for any single detection there are multiple categories whose score is above the confidence threshold, the
        detection will be duplicated to create one detection for each category. NMS processes one category at a time,
        iterating over the bounding boxes in descending order of confidence score, and removes lower scoring boxes that
        have an IoU greater than the NMS threshold with a higher scoring box.

        The returned detections are sorted by descending confidence. The items of the dictionaries are as follows:
        - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
        - scores (``Tensor[batch_size, N]``): detection confidences
        - labels (``Int64Tensor[batch_size, N]``): the predicted class IDs

        Args:
            preds: A tensor of detected bounding boxes and their attributes.

        Returns:
            Filtered detections. A list of prediction dictionaries, one for each image.

        """

        def process(boxes: Tensor, confidences: Tensor, classprobs: Tensor) -> Dict[str, Any]:
            scores = classprobs * confidences[:, None]

            # Select predictions with high scores. If a prediction has a high score for more than one class, it will be
            # duplicated.
            idxs, labels = (scores > self.confidence_threshold).nonzero().T
            boxes = boxes[idxs]
            scores = scores[idxs, labels]

            keep = batched_nms(boxes, scores, labels, self.nms_threshold)
            keep = keep[: self.detections_per_image]
            return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

        return [process(p[..., :4], p[..., 4], p[..., 5:]) for p in preds]

    def process_targets(self, targets: TARGETS) -> List[TARGET]:
        """Duplicates multi-label targets to create one target for each label.

        Args:
            targets: List of target dictionaries. Each dictionary must contain "boxes" and "labels". "labels" is either
                a one-dimensional list of class IDs, or a two-dimensional boolean class map.

        Returns:
            Single-label targets. A list of target dictionaries, one for each image.

        """

        def process(boxes: Tensor, labels: Tensor, **other: Any) -> Dict[str, Any]:
            if labels.ndim == 2:
                idxs, labels = labels.nonzero().T
                boxes = boxes[idxs]
            return {"boxes": boxes, "labels": labels, **other}

        return [process(**t) for t in targets]

    def validate_batch(self, images: Union[Tensor, IMAGES], targets: Optional[TARGETS]) -> None:
        """Validates the format of a batch of data.

        Args:
            images: A tensor containing a batch of images or a list of image tensors.
            targets: A list of target dictionaries or ``None``. If a list is provided, there should be as many target
                dictionaries as there are images.

        """
        if not isinstance(images, Tensor):
            if not isinstance(images, (tuple, list)):
                raise TypeError(f"Expected images to be a Tensor, tuple, or a list, got {type(images).__name__}.")
            if not images:
                raise ValueError("No images in batch.")
            shape = images[0].shape
            for image in images:
                if not isinstance(image, Tensor):
                    raise ValueError(f"Expected image to be of type Tensor, got {type(image).__name__}.")
                if image.shape != shape:
                    raise ValueError(f"Images with different shapes in one batch: {shape} and {image.shape}")

        if targets is None:
            if self.training:
                raise ValueError("Targets should be given in training mode.")
            return

        if not isinstance(targets, (tuple, list)):
            raise TypeError(f"Expected targets to be a tuple or a list, got {type(images).__name__}.")
        if len(images) != len(targets):
            raise ValueError(f"Got {len(images)} images, but targets for {len(targets)} images.")

        for target in targets:
            if "boxes" not in target:
                raise ValueError("Target dictionary doesn't contain boxes.")
            boxes = target["boxes"]
            if not isinstance(boxes, Tensor):
                raise TypeError(f"Expected target boxes to be of type Tensor, got {type(boxes).__name__}.")
            if (boxes.ndim != 2) or (boxes.shape[-1] != 4):
                raise ValueError(f"Expected target boxes to be tensors of shape [N, 4], got {list(boxes.shape)}.")
            if "labels" not in target:
                raise ValueError("Target dictionary doesn't contain labels.")
            labels = target["labels"]
            if not isinstance(labels, Tensor):
                raise ValueError(f"Expected target labels to be of type Tensor, got {type(labels).__name__}.")
            if (labels.ndim < 1) or (labels.ndim > 2) or (len(labels) != len(boxes)):
                raise ValueError(
                    f"Expected target labels to be tensors of shape [N] or [N, num_classes], got {list(labels.shape)}."
                )


class CLIYOLO(YOLO):
    """A subclass of YOLO that can be easily configured using LightningCLI.

    Either loads a Darknet configuration file, or constructs a YOLOv4 network. This is just an example of how to use the
    model. Various other network architectures from ``torch_networks.py`` can be used. Note that if you change the
    resolution of the input images, you should also scale the prior shapes (a.k.a. anchors). They are specified in the
    Darknet configuration file or provided in the network constructor parameters.

    CLI command::

        # Darknet network configuration
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
        python yolo_module.py fit \
            --model.network_config yolov4-tiny-3l.cfg \
            --data.batch_size 8 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 8 \
            --trainer.accumulate_grad_batches 2 \
            --trainer.gradient_clip_val 5.0 \
            --trainer.max_epochs=100

        # YOLOv4
        python yolo_module.py fit \
            --data.batch_size 8 \
            --data.num_workers 4 \
            --trainer.accelerator gpu \
            --trainer.devices 8 \
            --trainer.accumulate_grad_batches 2 \
            --trainer.gradient_clip_val 5.0 \
            --trainer.max_epochs=100

    Args:
        network_config: Path to a Darknet configuration file that defines the network architecture. If not given, a
            YOLOv4 network will be constructed.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Valid values are
            "iou", "giou", "diou", and "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that target
            confidence is one if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    """

    def __init__(
        self,
        network_config: Optional[str] = None,
        darknet_weights: Optional[str] = None,
        matching_algorithm: Optional[str] = None,
        matching_threshold: Optional[float] = None,
        spatial_range: float = 5.0,
        size_range: float = 4.0,
        ignore_bg_threshold: Optional[float] = None,
        overlap_func: Optional[str] = None,
        predict_overlap: Optional[float] = None,
        overlap_loss_multiplier: Optional[float] = None,
        confidence_loss_multiplier: Optional[float] = None,
        class_loss_multiplier: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if network_config is not None:
            network: nn.Module = DarknetNetwork(
                network_config,
                darknet_weights,
                matching_algorithm=matching_algorithm,
                matching_threshold=matching_threshold,
                spatial_range=spatial_range,
                size_range=size_range,
                ignore_bg_threshold=ignore_bg_threshold,
                overlap_func=overlap_func,
                predict_overlap=predict_overlap,
                overlap_loss_multiplier=overlap_loss_multiplier,
                confidence_loss_multiplier=confidence_loss_multiplier,
                class_loss_multiplier=class_loss_multiplier,
            )
        else:
            # We need to set some defaults, since we don't get the default values from a configuration file.
            if ignore_bg_threshold is None:
                ignore_bg_threshold = 0.7
            if overlap_func is None:
                overlap_func = "ciou"
            if overlap_loss_multiplier is None:
                overlap_loss_multiplier = 5.0
            if confidence_loss_multiplier is None:
                confidence_loss_multiplier = 1.0
            if class_loss_multiplier is None:
                class_loss_multiplier = 1.0

            network = YOLOV4Network(
                num_classes=21,  # The number of classes in Pascal VOC dataset.
                matching_algorithm=matching_algorithm,
                matching_threshold=matching_threshold,
                spatial_range=spatial_range,
                size_range=size_range,
                ignore_bg_threshold=ignore_bg_threshold,
                overlap_func=overlap_func,
                predict_overlap=predict_overlap,
                overlap_loss_multiplier=overlap_loss_multiplier,
                confidence_loss_multiplier=confidence_loss_multiplier,
                class_loss_multiplier=class_loss_multiplier,
            )
        super().__init__(**kwargs, network=network)


class ResizedVOCDetectionDataModule(VOCDetectionDataModule):
    """A subclass of ``VOCDetectionDataModule`` that resizes the images to a specific size. YOLO expectes the image
    size to be divisible by the ratio in which the network downsamples the image.

    Args:
        width: Resize images to this width.
        height: Resize images to this height.
    """

    def __init__(self, width: int = 608, height: int = 608, **kwargs: Any):
        super().__init__(**kwargs)
        self.image_size = (height, width)

    def default_transforms(self) -> Callable:
        transforms = [
            lambda image, target: (T.to_tensor(image), target),
            self._resize,
        ]
        if self.normalize:
            transforms += [
                lambda image, target: (
                    T.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    target,
                )
            ]
        return Compose(transforms)

    def _resize(self, image: Tensor, target: TARGET) -> Tuple[Tensor, TARGET]:
        """Rescales the image and target to ``self.image_size``.

        Args:
            tensor: Image tensor to be resized.
            target: Dictionary of detection targets.

        Returns:
            Resized image tensor.
        """
        device = target["boxes"].device
        height, width = image.shape[-2:]
        original_size = torch.tensor([height, width], device=device)
        scale_y, scale_x = torch.tensor(self.image_size, device=device) / original_size
        scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=device)
        image = T.resize(image, self.image_size)
        target["boxes"] = target["boxes"] * scale
        return image, target


if __name__ == "__main__":
    from pytorch_lightning.cli import LightningCLI

    LightningCLI(CLIYOLO, ResizedVOCDetectionDataModule, seed_everything_default=42)
