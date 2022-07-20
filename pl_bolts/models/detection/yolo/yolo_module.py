import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor, optim

from pl_bolts.models.detection.yolo.yolo_layers import DetectionLayer, RouteLayer, ShortcutLayer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.ops import nms
    from torchvision.transforms import functional as F
else:
    warn_missing_pkg("torchvision")

log = logging.getLogger(__name__)


@under_review()
class YOLO(LightningModule):
    """PyTorch Lightning implementation of YOLOv3 and YOLOv4.

    *YOLOv3 paper*: `Joseph Redmon and Ali Farhadi <https://arxiv.org/abs/1804.02767>`_

    *YOLOv4 paper*: `Alexey Bochkovskiy, Chien-Yao Wang, and Hong-Yuan Mark Liao <https://arxiv.org/abs/2004.10934>`_

    *Implementation*: `Seppo Enarvi <https://github.com/senarvi>`_

    The network architecture can be read from a Darknet configuration file using the
    :class:`~pl_bolts.models.detection.yolo.yolo_config.YOLOConfiguration` class, or created by
    some other means, and provided as a list of PyTorch modules.

    The input from the data loader is expected to be a list of images. Each image is a tensor with
    shape ``[channels, height, width]``. The images from a single batch will be stacked into a
    single tensor, so the sizes have to match. Different batches can have different image sizes, as
    long as the size is divisible by the ratio in which the network downsamples the input.

    During training, the model expects both the input tensors and a list of targets. *Each target is
    a dictionary containing*:

    - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in `(x1, y1, x2, y2)` format
    - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    :func:`~pl_bolts.models.detection.yolo.yolo_module.YOLO.forward` method returns all
    predictions from all detection layers in all images in one tensor with shape
    ``[images, predictors, classes + 5]``. The coordinates are scaled to the input image size.
    During training it also returns a dictionary containing the classification, box overlap, and
    confidence losses.

    During inference, the model requires only the input tensors.
    :func:`~pl_bolts.models.detection.yolo.yolo_module.YOLO.infer` method filters and processes the
    predictions. *The processed output includes the following tensors*:

    - boxes (``FloatTensor[N, 4]``): predicted bounding box `(x1, y1, x2, y2)` coordinates in image space
    - scores (``FloatTensor[N]``): detection confidences
    - labels (``Int64Tensor[N]``): the predicted labels for each image

    Weights can be loaded from a Darknet model file using ``load_darknet_weights()``.

    CLI command::

        # PascalVOC
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny-3l.cfg
        python yolo_module.py --config yolov4-tiny-3l.cfg --data_dir . --gpus 8 --batch_size 8
    """

    def __init__(
        self,
        network: nn.ModuleList,
        optimizer: Type[optim.Optimizer] = optim.SGD,
        optimizer_params: Dict[str, Any] = {"lr": 0.001, "momentum": 0.9, "weight_decay": 0.0005},
        lr_scheduler: Type[optim.lr_scheduler._LRScheduler] = LinearWarmupCosineAnnealingLR,
        lr_scheduler_params: Dict[str, Any] = {"warmup_epochs": 1, "max_epochs": 300, "warmup_start_lr": 0.0},
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        max_predictions_per_image: int = -1,
    ) -> None:
        """
        Args:
            network: A list of network modules. This can be obtained from a Darknet configuration
                using the :func:`~pl_bolts.models.detection.yolo.yolo_config.YOLOConfiguration.get_network`
                method.
            optimizer: Which optimizer class to use for training.
            optimizer_params: Parameters to pass to the optimizer constructor.
            lr_scheduler: Which learning rate scheduler class to use for training.
            lr_scheduler_params: Parameters to pass to the learning rate scheduler constructor.
            confidence_threshold: Postprocessing will remove bounding boxes whose
                confidence score is not higher than this threshold.
            nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU with a higher
                confidence box is higher than this threshold, if the predicted categories are equal.
            max_predictions_per_image: If non-negative, keep at most this number of
                highest-confidence predictions per image.
        """
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError("YOLO model uses `torchvision`, which is not installed yet.")  # pragma: no-cover

        self.network = network
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler_class = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_predictions_per_image = max_predictions_per_image

    def forward(
        self, images: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Runs a forward pass through the network (all layers listed in ``self.network``), and if training targets
        are provided, computes the losses from the detection layers.

        Detections are concatenated from the detection layers. Each image will produce
        `N * num_anchors * grid_height * grid_width` detections, where `N` depends on the number of
        detection layers. For one detection layer `N = 1`, and each detection layer increases it by
        a number that depends on the size of the feature map on that layer. For example, if the
        feature map is twice as wide and high as the grid, the layer will add four times more
        features.

        Args:
            images: Images to be processed. Tensor of size
                ``[batch_size, num_channels, height, width]``.
            targets: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.

        Returns:
            detections (:class:`~torch.Tensor`), losses (Dict[str, :class:`~torch.Tensor`]):
            Detections, and if targets were provided, a dictionary of losses. Detections are shaped
            ``[batch_size, num_predictors, num_classes + 5]``, where ``num_predictors`` is the
            total number of cells in all detection layers times the number of boxes predicted by
            one cell. The predicted box coordinates are in `(x1, y1, x2, y2)` format and scaled to
            the input image size.
        """
        outputs = []  # Outputs from all layers
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_height = images.shape[2]
        image_width = images.shape[3]
        image_size = torch.tensor([image_width, image_height], device=images.device)

        x = images
        for module in self.network:
            if isinstance(module, (RouteLayer, ShortcutLayer)):
                x = module(x, outputs)
            elif isinstance(module, DetectionLayer):
                if targets is None:
                    x = module(x, image_size)
                    detections.append(x)
                else:
                    x, layer_losses, layer_hits = module(x, image_size, targets)
                    detections.append(x)
                    losses.append(layer_losses)
                    hits.append(layer_hits)
            else:
                x = module(x)

            outputs.append(x)

        detections = torch.cat(detections, 1)
        if targets is None:
            return detections

        total_hits = sum(hits)
        num_targets = sum(len(image_targets["boxes"]) for image_targets in targets)
        if total_hits != num_targets:
            log.warning(
                f"{num_targets} training targets were matched a total of {total_hits} times by detection layers. "
                "Anchors may have been configured incorrectly."
            )
        for layer_idx, layer_hits in enumerate(hits):
            hit_rate = torch.true_divide(layer_hits, total_hits) if total_hits > 0 else 1.0
            self.log(f"layer_{layer_idx}_hit_rate", hit_rate, sync_dist=False)

        def total_loss(loss_name):
            """Returns the sum of the loss over detection layers."""
            loss_tuple = tuple(layer_losses[loss_name] for layer_losses in losses)
            return torch.stack(loss_tuple).sum()

        losses = {loss_name: total_loss(loss_name) for loss_name in losses[0].keys()}
        return detections, losses

    def configure_optimizers(self) -> Tuple[List, List]:
        """Constructs the optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int) -> Dict[str, Tensor]:
        """Computes the training loss.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors.
                Targets is a list of dictionaries that contain ground-truth boxes, labels, etc.
            batch_idx: The index of this batch.

        Returns:
            A dictionary that includes the training loss in 'loss'.
        """
        images, targets = self._validate_batch(batch)
        _, losses = self(images, targets)
        total_loss = torch.stack(tuple(losses.values())).sum()

        # sync_dist=True is broken in some versions of Lightning and may cause the sum of the loss
        # across GPUs to be returned.
        for name, value in losses.items():
            self.log(f"train/{name}_loss", value, prog_bar=True, sync_dist=False)
        self.log("train/total_loss", total_loss, sync_dist=False)

        return {"loss": total_loss}

    def validation_step(self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int):
        """Evaluates a batch of data from the validation set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors.
                Targets is a list of dictionaries that contain ground-truth boxes, labels, etc.
            batch_idx: The index of this batch
        """
        images, targets = self._validate_batch(batch)
        detections, losses = self(images, targets)
        detections = self._split_detections(detections)
        detections = self._filter_detections(detections)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log(f"val/{name}_loss", value, sync_dist=True)
        self.log("val/total_loss", total_loss, sync_dist=True)

    def test_step(self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int):
        """Evaluates a batch of data from the test set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors.
                Targets is a list of dictionaries that contain ground-truth boxes, labels, etc.
            batch_idx: The index of this batch.
        """
        images, targets = self._validate_batch(batch)
        detections, losses = self(images, targets)
        detections = self._split_detections(detections)
        detections = self._filter_detections(detections)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log(f"test/{name}_loss", value, sync_dist=True)
        self.log("test/total_loss", total_loss, sync_dist=True)

    def infer(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Feeds an image to the network and returns the detected bounding boxes, confidence scores, and class
        labels.

        Args:
            image: An input image, a tensor of uint8 values sized ``[channels, height, width]``.

        Returns:
            boxes (:class:`~torch.Tensor`), confidences (:class:`~torch.Tensor`), labels (:class:`~torch.Tensor`):
            A matrix of detected bounding box `(x1, y1, x2, y2)` coordinates, a vector of
            confidences for the bounding box detections, and a vector of predicted class labels.
        """
        if not isinstance(image, torch.Tensor):
            image = F.to_tensor(image)

        self.eval()
        detections = self(image.unsqueeze(0))
        detections = self._split_detections(detections)
        detections = self._filter_detections(detections)
        boxes = detections["boxes"][0]
        scores = detections["scores"][0]
        labels = detections["labels"][0]
        return boxes, scores, labels

    def load_darknet_weights(self, weight_file):
        """Loads weights to layer modules from a pretrained Darknet model.

        One may want to continue training from the pretrained weights, on a dataset with a
        different number of object categories. The number of kernels in the convolutional layers
        just before each detection layer depends on the number of output classes. The Darknet
        solution is to truncate the weight file and stop reading weights at the first incompatible
        layer. For this reason the function silently leaves the rest of the layers unchanged, when
        the weight file ends.

        Args:
            weight_file: A file object containing model weights in the Darknet binary format.
        """
        version = np.fromfile(weight_file, count=3, dtype=np.int32)
        images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
        rank_zero_info(
            f"Loading weights from Darknet model version {version[0]}.{version[1]}.{version[2]} "
            f"that has been trained on {images_seen[0]} images."
        )

        def read(tensor):
            """Reads the contents of ``tensor`` from the current position of ``weight_file``.

            If there's no more data in ``weight_file``, returns without error.
            """
            x = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            if x.shape[0] == 0:
                return
            x = torch.from_numpy(x).view_as(tensor)
            with torch.no_grad():
                tensor.copy_(x)

        for module in self.network:
            # Weights are loaded only to convolutional layers
            if not isinstance(module, nn.Sequential):
                continue

            conv = module[0]
            assert isinstance(conv, nn.Conv2d)

            # Convolution may be followed by batch normalization, in which case we read the batch
            # normalization parameters and not the convolution bias.
            if len(module) > 1 and isinstance(module[1], nn.BatchNorm2d):
                bn = module[1]
                read(bn.bias)
                read(bn.weight)
                read(bn.running_mean)
                read(bn.running_var)
            else:
                read(conv.bias)

            read(conv.weight)

    def _validate_batch(
        self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]]
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """Reads a batch of data, validates the format, and stacks the images into a single tensor.

        Args:
            batch: The batch of data read by the :class:`~torch.utils.data.DataLoader`.

        Returns:
            The input batch with images stacked into a single tensor.
        """
        images, targets = batch

        if len(images) != len(targets):
            raise ValueError(f"Got {len(images)} images, but targets for {len(targets)} images.")

        for image in images:
            if not isinstance(image, Tensor):
                raise ValueError(f"Expected image to be of type Tensor, got {type(image)}.")

        for target in targets:
            boxes = target["boxes"]
            if not isinstance(boxes, Tensor):
                raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
            if (len(boxes.shape) != 2) or (boxes.shape[-1] != 4):
                raise ValueError(f"Expected target boxes to be tensors of shape [N, 4], got {list(boxes.shape)}.")
            labels = target["labels"]
            if not isinstance(labels, Tensor):
                raise ValueError(f"Expected target labels to be of type Tensor, got {type(labels)}.")
            if len(labels.shape) != 1:
                raise ValueError(f"Expected target labels to be tensors of shape [N], got {list(labels.shape)}.")

        images = torch.stack(images)
        return images, targets

    def _split_detections(self, detections: Tensor) -> Dict[str, Tensor]:
        """Splits the detection tensor returned by a forward pass into a dictionary.

        The fields of the dictionary are as follows:
            - boxes (``Tensor[batch_size, N, 4]``): detected bounding box `(x1, y1, x2, y2)` coordinates
            - scores (``Tensor[batch_size, N]``): detection confidences
            - classprobs (``Tensor[batch_size, N]``): probabilities of the best classes
            - labels (``Int64Tensor[batch_size, N]``): the predicted labels for each image

        Args:
            detections: A tensor of detected bounding boxes and their attributes.

        Returns:
            A dictionary of detection results.
        """
        boxes = detections[..., :4]
        scores = detections[..., 4]
        classprobs = detections[..., 5:]
        classprobs, labels = torch.max(classprobs, -1)
        return {"boxes": boxes, "scores": scores, "classprobs": classprobs, "labels": labels}

    def _filter_detections(self, detections: Dict[str, Tensor]) -> Dict[str, List[Tensor]]:
        """Filters detections based on confidence threshold. Then for every class performs non-maximum suppression
        (NMS). NMS iterates the bounding boxes that predict this class in descending order of confidence score, and
        removes lower scoring boxes that have an IoU greater than the NMS threshold with a higher scoring box.
        Finally the detections are sorted by descending confidence and possible truncated to the maximum number of
        predictions.

        Args:
            detections: All detections. A dictionary of tensors, each containing the predictions
                from all images.

        Returns:
            Filtered detections. A dictionary of lists, each containing a tensor per image.
        """
        boxes = detections["boxes"]
        scores = detections["scores"]
        classprobs = detections["classprobs"]
        labels = detections["labels"]

        out_boxes = []
        out_scores = []
        out_classprobs = []
        out_labels = []

        for img_boxes, img_scores, img_classprobs, img_labels in zip(boxes, scores, classprobs, labels):
            # Select detections with high confidence score.
            selected = img_scores > self.confidence_threshold
            img_boxes = img_boxes[selected]
            img_scores = img_scores[selected]
            img_classprobs = img_classprobs[selected]
            img_labels = img_labels[selected]

            img_out_boxes = boxes.new_zeros((0, 4))
            img_out_scores = scores.new_zeros(0)
            img_out_classprobs = classprobs.new_zeros(0)
            img_out_labels = labels.new_zeros(0)

            # Iterate through the unique object classes detected in the image and perform non-maximum
            # suppression for the objects of the class in question.
            for cls_label in labels.unique():
                selected = img_labels == cls_label
                cls_boxes = img_boxes[selected]
                cls_scores = img_scores[selected]
                cls_classprobs = img_classprobs[selected]
                cls_labels = img_labels[selected]

                # NMS will crash if there are too many boxes.
                cls_boxes = cls_boxes[:100000]
                cls_scores = cls_scores[:100000]
                selected = nms(cls_boxes, cls_scores, self.nms_threshold)

                img_out_boxes = torch.cat((img_out_boxes, cls_boxes[selected]))
                img_out_scores = torch.cat((img_out_scores, cls_scores[selected]))
                img_out_classprobs = torch.cat((img_out_classprobs, cls_classprobs[selected]))
                img_out_labels = torch.cat((img_out_labels, cls_labels[selected]))

            # Sort by descending confidence and limit the maximum number of predictions.
            indices = torch.argsort(img_out_scores, descending=True)
            if self.max_predictions_per_image >= 0:
                indices = indices[: self.max_predictions_per_image]
            out_boxes.append(img_out_boxes[indices])
            out_scores.append(img_out_scores[indices])
            out_classprobs.append(img_out_classprobs[indices])
            out_labels.append(img_out_labels[indices])

        return {"boxes": out_boxes, "scores": out_scores, "classprobs": out_classprobs, "labels": out_labels}


@under_review()
class Resize:
    """Rescales the image and target to given dimensions.

    Args:
        output_size (tuple or int): Desired output size. If tuple (height, width), the output is
            matched to ``output_size``. If int, the smaller of the image edges is matched to
            ``output_size``, keeping the aspect ratio the same.
    """

    def __init__(self, output_size: tuple) -> None:
        self.output_size = output_size

    def __call__(self, image: Tensor, target: Dict[str, Any]):
        """
        Args:
            tensor: Tensor image to be resized.
            target: Dictionary of detection targets.

        Returns:
            Resized Tensor image.
        """
        height, width = image.shape[-2:]
        original_size = torch.tensor([height, width])
        scale_y, scale_x = torch.tensor(self.output_size) / original_size
        scale = torch.tensor([scale_x, scale_y, scale_x, scale_y], device=target["boxes"].device)
        image = F.resize(image, self.output_size)
        target["boxes"] = target["boxes"] * scale
        return image, target


@under_review()
def run_cli():
    from argparse import ArgumentParser

    from pytorch_lightning import Trainer, seed_everything

    from pl_bolts.datamodules import VOCDetectionDataModule
    from pl_bolts.datamodules.vocdetection_datamodule import Compose
    from pl_bolts.models.detection.yolo.yolo_config import YOLOConfiguration

    seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        required=True,
        help="read model configuration from PATH",
    )
    parser.add_argument(
        "--darknet-weights",
        type=str,
        metavar="PATH",
        help="read the initial model weights from PATH in Darknet format",
    )
    parser.add_argument(
        "--lr",
        type=float,
        metavar="LR",
        default=0.0013,
        help="learning rate after the warmup period",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        metavar="GAMMA",
        default=0.9,
        help="if nonzero, the optimizer uses momentum with factor GAMMA",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        metavar="LAMBDA",
        default=0.0005,
        help="if nonzero, the optimizer uses weight decay (L2 penalty) with factor LAMBDA",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        metavar="N",
        default=1,
        help="learning rate warmup period is N epochs",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        metavar="N",
        default=300,
        help="train at most N epochs",
    )
    parser.add_argument(
        "--initial-lr",
        type=float,
        metavar="LR",
        default=0.0,
        help="learning rate before the warmup period",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        metavar="THRESHOLD",
        default=0.001,
        help="keep predictions only if the confidence is above THRESHOLD",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        metavar="THRESHOLD",
        default=0.45,
        help="non-maximum suppression removes predicted boxes that have IoU greater than "
        "THRESHOLD with a higher scoring box",
    )
    parser.add_argument(
        "--max-predictions-per-image",
        type=int,
        metavar="N",
        default=100,
        help="keep at most N best predictions",
    )

    parser = VOCDetectionDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = YOLOConfiguration(args.config)

    transforms = [lambda image, target: (F.to_tensor(image), target), Resize((config.height, config.width))]
    transforms = Compose(transforms)
    datamodule = VOCDetectionDataModule.from_argparse_args(args, train_transforms=transforms, val_transforms=transforms)

    optimizer_params = {"lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay}
    lr_scheduler_params = {
        "warmup_epochs": args.warmup_epochs,
        "max_epochs": args.max_epochs,
        "warmup_start_lr": args.initial_lr,
    }
    model = YOLO(
        network=config.get_network(),
        optimizer_params=optimizer_params,
        lr_scheduler_params=lr_scheduler_params,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        max_predictions_per_image=args.max_predictions_per_image,
    )
    if args.darknet_weights is not None:
        with open(args.darknet_weights) as weight_file:
            model.load_darknet_weights(weight_file)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    run_cli()
