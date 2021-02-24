from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim, Tensor

from pl_bolts.models.detection.yolo.yolo_config import YOLOConfiguration
from pl_bolts.models.detection.yolo.yolo_layers import DetectionLayer, RouteLayer, ShortcutLayer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.ops import nms
    from torchvision.transforms import functional as F
else:
    warn_missing_pkg('torchvision')


class YOLO(pl.LightningModule):
    """
    PyTorch Lightning implementation of `YOLOv3 <https://arxiv.org/abs/1804.02767>`_ with some
    improvements from `YOLOv4 <https://arxiv.org/abs/2004.10934>`_.

    YOLOv3 paper authors: Joseph Redmon and Ali Farhadi

    YOLOv4 paper authors: Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao

    Model implemented by:
        - `Seppo Enarvi <https://github.com/senarvi>`_

    The network architecture can be read from a Darknet configuration file using the
    :class:`~pl_bolts.models.detection.yolo.yolo_config.YoloConfiguration` class, or created by
    some other means, and provided as a list of PyTorch modules. Supports loading weights from a
    Darknet model file too, if you don't want to start training from a randomly initialized model.
    During training, the model expects both the images (list of tensors), as well as targets (list
    of dictionaries).

    The target dictionaries should contain:
        - boxes (``FloatTensor[N, 4]``): the ground truth boxes in ``[x1, y1, x2, y2]`` format.
        - labels (``LongTensor[N]``): the class label for each ground truh box

    CLI command::

        # PascalVOC
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
        python yolo_module.py --config yolov4-tiny.cfg --data_dir . --gpus 8 --batch-size 8
    """

    def __init__(
        self,
        network: nn.ModuleList,
        optimizer: Type[optim.Optimizer] = optim.SGD,
        optimizer_params: Dict[str, Any] = {'lr': 0.0013, 'momentum': 0.9, 'weight_decay': 0.0005},
        lr_scheduler: Type[optim.lr_scheduler._LRScheduler] = LinearWarmupCosineAnnealingLR,
        lr_scheduler_params: Dict[str, Any] = {'warmup_epochs': 1, 'max_epochs': 271, 'warmup_start_lr': 0.0},
        confidence_threshold: float = 0.2,
        nms_threshold: float = 0.45,
        max_predictions_per_image: int = -1
    ) -> None:
        """
        Args:
            network: A list of network modules. This can be obtained from a Darknet configuration
                using the ``YoloConfiguration.get_network()`` method.
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
            raise ModuleNotFoundError(  # pragma: no-cover
                'YOLO model uses `torchvision`, which is not installed yet.'
            )

        self.network = network
        self.optimizer_class = optimizer
        self.optimizer_params = optimizer_params
        self.lr_scheduler_class = lr_scheduler
        self.lr_scheduler_params = lr_scheduler_params
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_predictions_per_image = max_predictions_per_image

    def forward(
        self,
        images: Tensor,
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Runs a forward pass through the network (all layers listed in ``self.network``), and if
        training targets are provided, computes the losses from the detection layers.

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
            boxes (Tensor), confidences (Tensor), classprobs (Tensor), losses (Dict[str, Tensor]):
                Detections, and if targets were provided, a dictionary of losses. The first
                dimension of the detections is the index of the image in the batch and the second
                dimension is the detection within the image. ``boxes`` contains the predicted
                (x1, y1, x2, y2) coordinates, normalized to [0, 1].
        """
        outputs = []  # Outputs from all layers
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers

        x = images
        for module in self.network:
            if isinstance(module, (RouteLayer, ShortcutLayer)):
                x = module(x, outputs)
            elif isinstance(module, DetectionLayer):
                if targets is None:
                    x = module(x)
                    detections.append(x)
                else:
                    x, layer_losses = module(x, targets)
                    detections.append(x)
                    losses.append(layer_losses)
            else:
                x = module(x)

            outputs.append(x)

        def mean_loss(loss_name):
            loss_tuple = tuple(layer_losses[loss_name] for layer_losses in losses)
            return torch.stack(loss_tuple).sum() / images.shape[0]

        detections = torch.cat(detections, 1)
        boxes = detections[..., :4]
        confidences = detections[..., 4]
        classprobs = detections[..., 5:]

        if targets is None:
            return boxes, confidences, classprobs

        losses = {loss_name: mean_loss(loss_name) for loss_name in losses[0].keys()}
        return boxes, confidences, classprobs, losses

    def configure_optimizers(self) -> Tuple[List, List]:
        """Constructs the optimizer and learning rate scheduler."""
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_params)
        lr_scheduler = self.lr_scheduler_class(optimizer, **self.lr_scheduler_params)
        return [optimizer], [lr_scheduler]

    def training_step(self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int) -> Dict[str, Tensor]:
        """
        Computes the training loss.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors.
                Targets is a list of dictionaries that contain ground-truth boxes, labels, etc.
            batch_idx: The index of this batch.

        Returns:
            A dictionary that includes the training loss in 'loss'.
        """
        images, targets = self._validate_batch(batch)
        _, _, _, losses = self(images, targets)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log('train/{}_loss'.format(name), value)
        self.log('train/total_loss', total_loss)

        return {'loss': total_loss}

    def validation_step(self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int) -> Dict[str, Tensor]:
        """
        Evaluates a batch of data from the validation set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors.
                Targets is a list of dictionaries that contain ground-truth boxes, labels, etc.
            batch_idx: The index of this batch
        """
        images, targets = self._validate_batch(batch)
        boxes, confidences, classprobs, losses = self(images, targets)
        classprobs, labels = torch.max(classprobs, -1)
        boxes, confidences, classprobs, labels = self._filter_detections(boxes, confidences, classprobs, labels)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log('val/{}_loss'.format(name), value)
        self.log('val/total_loss', total_loss)

    def test_step(self, batch: Tuple[List[Tensor], List[Dict[str, Tensor]]], batch_idx: int) -> Dict[str, Tensor]:
        """
        Evaluates a batch of data from the test set.

        Args:
            batch: A tuple of images and targets. Images is a list of 3-dimensional tensors.
                Targets is a list of dictionaries that contain ground-truth boxes, labels, etc.
            batch_idx: The index of this batch.
        """
        images, targets = self._validate_batch(batch)
        boxes, confidences, classprobs, losses = self(images, targets)
        classprobs, labels = torch.max(classprobs, -1)
        boxes, confidences, classprobs, labels = self._filter_detections(boxes, confidences, classprobs, labels)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log('test/{}_loss'.format(name), value)
        self.log('test/total_loss', total_loss)

    def infer(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Resizes given image to the network input size and feeds it to the network. Returns the
        detected bounding boxes, confidences, and class labels.

        Args:
            image: An input image, a tensor of uint8 values sized ``[channels, height, width]``.

        Returns:
            boxes (:class:`~torch.Tensor`), confidences (:class:`~torch.Tensor`), labels (:class:`~torch.Tensor`):
                A matrix of detected bounding box (x1, y1, x2, y2) coordinates, a vector of
                confidences for the bounding box detections, and a vector of predicted class
                labels.
        """
        network_input = image.float().div(255.0)
        network_input = network_input.unsqueeze(0)
        self.eval()
        boxes, confidences, classprobs = self(network_input)
        classprobs, labels = torch.max(classprobs, -1)
        boxes, confidences, classprobs, labels = self._filter_detections(boxes, confidences, classprobs, labels)
        assert len(boxes) == 1
        boxes = boxes[0]
        confidences = confidences[0]
        labels = labels[0]

        height = image.shape[1]
        width = image.shape[2]
        scale = torch.tensor([width, height, width, height], device=boxes.device)
        boxes = boxes * scale
        boxes = torch.round(boxes).int()
        return boxes, confidences, labels

    def load_darknet_weights(self, weight_file):
        """
        Loads weights to layer modules from a pretrained Darknet model.

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
        print(
            'Loading weights from Darknet model version {}.{}.{} that has been trained on {} '
            'images.'.format(version[0], version[1], version[2], images_seen[0])
        )

        def read(tensor):
            """
            Reads the contents of ``tensor`` from the current position of ``weight_file``.
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

            if len(module) > 1:
                bn = module[1]
                assert isinstance(bn, nn.BatchNorm2d)

                read(bn.bias)
                read(bn.weight)
                read(bn.running_mean)
                read(bn.running_var)
            else:
                read(conv.bias)

            read(conv.weight)

    def _validate_batch(
        self,
        batch: Tuple[List[Tensor], List[Dict[str, Tensor]]]
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """
        Reads a batch of data, validates the format, and stacks the images into a single tensor.

        Args:
            batch: The batch of data read by the :class:`~torch.utils.data.DataLoader`.

        Returns:
            batch: The input batch with images stacked into a single tensor.
        """
        images, targets = batch

        if len(images) != len(targets):
            raise ValueError("Got {} images, but targets for {} images.".format(len(images), len(targets)))

        for image in images:
            if not isinstance(image, Tensor):
                raise ValueError("Expected image to be of type Tensor, got {}.".format(type(image)))

        for target in targets:
            boxes = target['boxes']
            if not isinstance(boxes, Tensor):
                raise ValueError("Expected target boxes to be of type Tensor, got {}.".format(type(boxes)))
            if (len(boxes.shape) != 2) or (boxes.shape[-1] != 4):
                raise ValueError(
                    "Expected target boxes to be tensors of shape [N, 4], got {}.".format(list(boxes.shape))
                )
            labels = target['labels']
            if not isinstance(labels, Tensor):
                raise ValueError("Expected target labels to be of type Tensor, got {}.".format(type(labels)))
            if len(labels.shape) != 1:
                raise ValueError(
                    "Expected target labels to be tensors of shape [N], got {}.".format(list(labels.shape))
                )

        images = torch.stack(images)
        return images, targets

    def _filter_detections(
        self,
        boxes: Tensor,
        confidences: Tensor,
        classprobs: Tensor,
        labels: Tensor
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """
        Filters detections based on confidence threshold. Then for every class performs non-maximum
        suppression (NMS). NMS iterates the bounding boxes that predict this class in descending
        order of confidence score, and removes the bounding box, if its IoU with the next one is
        higher than the NMS threshold.

        Args:
            boxes: Detected bounding box (x1, y1, x2, y2) coordinates in a tensor sized
                ``[batch_size, N, 4]``.
            confidences: Detection confidences in a tensor sized ``[batch_size, N]``.
            classprobs: Probabilities of the best classes in a tensor sized ``[batch_size, N]``.
            labels: Indices of the best classes in a tensor sized ``[batch_size, N]``.

        Returns:
            boxes (List[Tensor]), confidences (List[Tensor]), classprobs (List[Tensor]), labels (List[Tensor]):
                Four lists, each containing one tensor per image - bounding box (x1, y1, x2, y2)
                coordinates, detection confidences, probabilities of the best class of each
                prediction, and the predicted class labels.
        """
        out_boxes = []
        out_confidences = []
        out_classprobs = []
        out_labels = []

        for img_boxes, img_confidences, img_classprobs, img_labels in zip(boxes, confidences, classprobs, labels):
            # Select detections with high confidence score.
            selected = img_confidences > self.confidence_threshold
            img_boxes = img_boxes[selected]
            img_confidences = img_confidences[selected]
            img_classprobs = img_classprobs[selected]
            img_labels = img_labels[selected]

            img_out_boxes = boxes.new_zeros((0, 4))
            img_out_confidences = confidences.new_zeros(0)
            img_out_classprobs = classprobs.new_zeros(0)
            img_out_labels = labels.new_zeros(0)

            # Iterate through the unique object classes detected in the image and perform non-maximum
            # suppression for the objects of the class in question.
            for cls_label in labels.unique():
                selected = img_labels == cls_label
                cls_boxes = img_boxes[selected]
                cls_confidences = img_confidences[selected]
                cls_classprobs = img_classprobs[selected]
                cls_labels = img_labels[selected]

                selected = nms(cls_boxes, cls_confidences, self.nms_threshold)
                img_out_boxes = torch.cat((img_out_boxes, cls_boxes[selected]))
                img_out_confidences = torch.cat((img_out_confidences, cls_confidences[selected]))
                img_out_classprobs = torch.cat((img_out_classprobs, cls_classprobs[selected]))
                img_out_labels = torch.cat((img_out_labels, cls_labels[selected]))

            # Sort by descending confidence and limit the maximum number of predictions.
            indices = torch.argsort(img_out_confidences, descending=True)
            if self.max_predictions_per_image >= 0:
                indices = indices[:self.max_predictions_per_image]
            out_boxes.append(img_out_boxes[indices])
            out_confidences.append(img_out_confidences[indices])
            out_classprobs.append(img_out_classprobs[indices])
            out_labels.append(img_out_labels[indices])

        return out_boxes, out_confidences, out_classprobs, out_labels


class Resize:
    """Rescales the image and target to given dimensions.

    Args:
        output_size (tuple or int): Desired output size. If tuple (height, width), the output is
            matched to ``output_size``. If int, the smaller of the image edges is matched to
            ``output_size``, keeping the aspect ratio the same.
    """

    def __init__(self, output_size: tuple) -> None:
        self.output_size = output_size

    def __call__(self, image, target):
        width, height = image.size
        original_size = torch.tensor([height, width])
        resize_ratio = torch.tensor(self.output_size) / original_size
        image = F.resize(image, self.output_size)
        scale = torch.tensor(
            [
                resize_ratio[1],  # y
                resize_ratio[0],  # x
                resize_ratio[1],  # y
                resize_ratio[0]  # x
            ],
            device=target['boxes'].device
        )
        target['boxes'] = target['boxes'] * scale
        return image, target


def run_cli():
    from argparse import ArgumentParser

    from pl_bolts.datamodules import VOCDetectionDataModule

    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, metavar='PATH', required=True,
        help='read model configuration from PATH'
    )
    parser.add_argument(
        '--darknet-weights', type=str, metavar='PATH',
        help='read the initial model weights from PATH in Darknet format'
    )
    parser.add_argument(
        '--batch-size', type=int, metavar='N', default=16,
        help='batch size is N image'
    )
    parser.add_argument(
        '--lr', type=float, metavar='LR', default=0.0013,
        help='learning rate after the warmup period'
    )
    parser.add_argument(
        '--momentum', type=float, metavar='GAMMA', default=0.9,
        help='if nonzero, the optimizer uses momentum with factor GAMMA'
    )
    parser.add_argument(
        '--weight-decay', type=float, metavar='LAMBDA', default=0.0005,
        help='if nonzero, the optimizer uses weight decay (L2 penalty) with factor LAMBDA'
    )
    parser.add_argument(
        '--warmup-epochs', type=int, metavar='N', default=1,
        help='learning rate warmup period is N epochs'
    )
    parser.add_argument(
        '--max-epochs', type=int, metavar='N', default=300,
        help='train at most N epochs'
    )
    parser.add_argument(
        '--initial-lr', type=float, metavar='LR', default=0.0,
        help='learning rate before the warmup period'
    )
    parser.add_argument(
        '--confidence-threshold', type=float, metavar='THRESHOLD', default=0.001,
        help='keep predictions only if the confidence is above THRESHOLD'
    )
    parser.add_argument(
        '--nms-threshold', type=float, metavar='THRESHOLD', default=0.45,
        help='non-maximum suppression removes predicted boxes that have IoU greater than '
             'THRESHOLD with a higher scoring box'
    )
    parser.add_argument(
        '--max-predictions-per-image', type=int, metavar='N', default=100,
        help='keep at most N best predictions'
    )

    parser = VOCDetectionDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = YOLOConfiguration(args.config)

    transforms = [Resize((config.height, config.width))]
    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    datamodule.prepare_data()

    optimizer_params = {
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay
    }
    lr_scheduler_params = {
        'warmup_epochs': args.warmup_epochs,
        'max_epochs': args.max_epochs,
        'warmup_start_lr': args.initial_lr
    }
    model = YOLO(
        network=config.get_network(),
        optimizer_params=optimizer_params,
        lr_scheduler_params=lr_scheduler_params,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        max_predictions_per_image=args.max_predictions_per_image
    )
    if args.darknet_weights is not None:
        with open(args.darknet_weights, 'r') as weight_file:
            model.load_darknet_weights(weight_file)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(
        model, datamodule.train_dataloader(args.batch_size, transforms),
        datamodule.val_dataloader(args.batch_size, transforms)
    )


if __name__ == "__main__":
    run_cli()
