import inspect
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities import argparse_utils
from torch import optim, Tensor

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.utils.warnings import warn_missing_pkg
from pl_bolts.models.detection.yolo.yolo_config import YoloConfiguration
from pl_bolts.models.detection.yolo.yolo_layers import DetectionLayer, Mish, RouteLayer, ShortcutLayer

try:
    import torchvision.transforms as T
    from torchvision.ops import nms
    from torchvision.transforms import functional as F
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class Yolo(pl.LightningModule):
    """
    PyTorch Lightning implementation of `YOLOv3 <https://arxiv.org/abs/1804.02767>`_ with some
    improvements from `YOLOv4 <https://arxiv.org/abs/2004.10934>`_.

    YOLOv3 paper authors: Joseph Redmon and Ali Farhadi

    YOLOv4 paper authors: Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao

    Model implemented by:
        - `Seppo Enarvi <https://github.com/senarvi>`_

    The network architecture is read from a configuration file in the same format as in the Darknet
    implementation. Supports loading weights from a Darknet model file too, if you don't want to
    start training from a randomly initialized model. During training, the model expects both the
    images (list of tensors), as well as targets (list of dictionaries).

    The target dictionaries should contain:
        - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
        - labels (`LongTensor[N]`): the class label for each ground truh box

    CLI command::

        # PascalVOC
        wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg
        python yolo_module.py --config yolov4-tiny.cfg --data_dir . --gpus 8 --batch-size 8
    """

    def __init__(self,
                 configuration: YoloConfiguration,
                 optimizer: str = 'sgd',
                 momentum: float = 0.9,
                 weight_decay: float = 0.0005,
                 learning_rate: float = 0.0013,
                 warmup_epochs: int = 1,
                 warmup_start_lr: float = 0.0001,
                 annealing_epochs: int = 271,
                 confidence_threshold: float = 0.2,
                 nms_threshold: float = 0.45):
        """
        Args:
            configuration: The model configuration.
            optimizer: Which optimizer to use for training; either 'sgd' or 'adam'.
            momentum: Momentum factor for SGD with momentum.
            weight_decay: Weight decay (L2 penalty).
            learning_rate: Learning rate after the warmup period.
            warmup_epochs: Length of the learning rate warmup period in the beginning of
                training. During this number of epochs, the learning rate will be raised from
                `warmup_start_lr` to `learning_rate`.
            warmup_start_lr: Learning rate in the beginning of the warmup period.
            annealing_epochs: Length of the learning rate annealing period, during which the
                learning rate will go to zero.
            confidence_threshold: Postprocessing will remove bounding boxes whose
                confidence score is not higher than this threshold.
            nms_threshold: Non-maximum suppression will remove bounding boxes whose IoU
                with the next best bounding box in that class is higher than this threshold.
        """
        super().__init__()

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                'YOLO model uses `torchvision`, which is not installed yet.'
            )

        self.config = configuration
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.annealing_epochs = annealing_epochs
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self._create_modules()

    def forward(
        self,
        images: Tensor,
        targets: List[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """
        Runs a forward pass through the network (all layers listed in `self._module_list`), and if
        training targets are provided, computes the losses from the detection layers.

        Detections are concatenated from the detection layers. Each image will produce
        `N * num_anchors * grid_height * grid_width` detections, where `N` depends on the number of
        detection layers. For one detection layer `N = 1`, and each detection layer increases it by
        a number that depends on the size of the feature map on that layer. For example, if the
        feature map is twice as wide and high as the grid, the layer will add four times more
        features.

        Args:
            images: Images to be processed. Tensor of size
                `[batch_size, num_channels, height, width]`.
            targets: If set, computes losses from detection layers against these targets. A list of
                dictionaries, one for each image.

        Returns:
            boxes (Tensor), confidences (Tensor), classprobs (Tensor), losses (Dict[str, Tensor]):
                Detections, and if targets were provided, a dictionary of losses. The first
                dimension of the detections is the index of the image in the batch and the second
                dimension is the detection within the image. `boxes` contains the predicted
                (x1, y1, x2, y2) coordinates, normalized to [0, 1].
        """
        outputs = []     # Outputs from all layers
        detections = []  # Outputs from detection layers
        losses = []      # Losses from detection layers

        x = images
        for module in self._module_list:
            if isinstance(module, RouteLayer) or isinstance(module, ShortcutLayer):
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

        if targets is not None:
            losses = {loss_name: mean_loss(loss_name) for loss_name in losses[0].keys()}
            return boxes, confidences, classprobs, losses
        else:
            return boxes, confidences, classprobs

    def configure_optimizers(self) -> Tuple[List, List]:
        """Constructs the optimizer and learning rate scheduler."""
        if self.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.learning_rate
            )
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.annealing_epochs,
            warmup_start_lr=self.warmup_start_lr)
        return [optimizer], [lr_scheduler]

    def training_step(
        self,
        batch: Tuple[List[Tensor], List[Dict[str, Tensor]]],
        batch_idx: int
    ) -> Dict[str, Tensor]:
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

    def validation_step(
        self,
        batch: Tuple[List[Tensor], List[Dict[str, Tensor]]],
        batch_idx: int
    ) -> Dict[str, Tensor]:
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
        boxes, confidences, classprobs, labels = self._filter_detections(
            boxes, confidences, classprobs, labels)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log('val/{}_loss'.format(name), value)
        self.log('val/total_loss', total_loss)

    def test_step(
        self,
        batch: Tuple[List[Tensor], List[Dict[str, Tensor]]],
        batch_idx: int
    ) -> Dict[str, Tensor]:
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
        boxes, confidences, classprobs, labels = self._filter_detections(
            boxes, confidences, classprobs, labels)
        total_loss = torch.stack(tuple(losses.values())).sum()

        for name, value in losses.items():
            self.log('test/{}_loss'.format(name), value)
        self.log('test/total_loss', total_loss)

    def infer(self, image: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Resizes given image to the network input size and feeds it to the network. Returns the
        detected bounding boxes, confidences, and class labels.

        Args:
            image: An input image, a tensor of uint8 values sized `[channels, height, width]`.

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
        boxes, confidences, classprobs, labels = self._filter_detections(
            boxes, confidences, classprobs, labels)
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

        Args:
            weight_file: A file object containing model weights in the Darknet binary format.
        """
        version = np.fromfile(weight_file, count=3, dtype=np.int32)
        images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
        print('Loading weights from Darknet model version {}.{}.{} that has been trained on {} '
              'images.'.format(version[0], version[1], version[2], images_seen[0]))

        def read(tensor):
            x = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            x = torch.from_numpy(x).view_as(tensor)
            with torch.no_grad():
                tensor.copy_(x)

        for module in self._module_list:
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

    @classmethod
    def get_deprecated_arg_names(cls) -> List:
        """Returns a list with deprecated constructor arguments."""
        depr_arg_names = []
        for name, val in cls.__dict__.items():
            if name.startswith('DEPRECATED') and isinstance(val, (tuple, list)):
                depr_arg_names.extend(val)
        return depr_arg_names

    def _create_modules(self):
        """
        Creates a list of network modules based on parsed configuration file.
        """
        self._module_list = nn.ModuleList()
        num_outputs = 3     # Number of channels in the previous layer output
        layer_outputs = []  # Number of channels in the output of every layer

        # Iterate through the modules from the configuration and generate required components.
        for index, config in enumerate(self.config.modules):
            if config['type'] == 'convolutional':
                module = nn.Sequential()

                batch_normalize = config.get('batch_normalize', False)
                padding = (config['size'] - 1) // 2 if config['pad'] else 0

                conv = nn.Conv2d(
                    num_outputs,
                    config['filters'],
                    config['size'],
                    config['stride'],
                    padding,
                    bias=not batch_normalize)
                module.add_module("conv_{0}".format(index), conv)
                num_outputs = config['filters']

                if batch_normalize:
                    bn = nn.BatchNorm2d(config['filters'])
                    module.add_module("batch_norm_{0}".format(index), bn)

                if config['activation'] == 'leaky':
                    leakyrelu = nn.LeakyReLU(0.1, inplace=True)
                    module.add_module('leakyrelu_{0}'.format(index), leakyrelu)
                elif config['activation'] == 'mish':
                    mish = Mish()
                    module.add_module("mish_{0}".format(index), mish)

            elif config['type'] == 'upsample':
                module = nn.Upsample(scale_factor=config["stride"], mode='nearest')

            elif config['type'] == 'route':
                num_chunks = config.get('groups', 1)
                chunk_idx = config.get('group_id', 0)
                source_layers = [layer if layer >= 0 else index + layer
                                 for layer in config['layers']]
                module = RouteLayer(source_layers, num_chunks, chunk_idx)
                num_outputs = sum(layer_outputs[layer] // num_chunks
                                  for layer in source_layers)

            elif config['type'] == 'shortcut':
                module = ShortcutLayer(config['from'])

            elif config['type'] == 'yolo':
                # The "anchors" list alternates width and height.
                anchor_dims = config['anchors']
                anchor_dims = [(anchor_dims[i], anchor_dims[i + 1])
                               for i in range(0, len(anchor_dims), 2)]

                xy_scale = config.get('scale_x_y', 1.0)
                ignore_threshold = config.get('ignore_thresh', 1.0)
                overlap_loss_multiplier = config.get('iou_normalizer', 1.0)
                class_loss_multiplier = config.get('cls_normalizer', 1.0)
                confidence_loss_multiplier = config.get('obj_normalizer', 1.0)

                module = DetectionLayer(
                    num_classes=config['classes'],
                    image_width=self.config.width,
                    image_height=self.config.height,
                    anchor_dims=anchor_dims,
                    anchor_ids=config['mask'],
                    xy_scale=xy_scale,
                    ignore_threshold=ignore_threshold,
                    overlap_loss_multiplier=overlap_loss_multiplier,
                    class_loss_multiplier=class_loss_multiplier,
                    confidence_loss_multiplier=confidence_loss_multiplier)

            elif config['type'] == 'maxpool':
                padding = (config['size'] - 1) // 2
                module = nn.MaxPool2d(config['size'], config['stride'], padding)

            self._module_list.append(module)
            layer_outputs.append(num_outputs)

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
            raise ValueError("Got {} images, but targets for {} images."
                             .format(len(images), len(targets)))

        for image in images:
            if not isinstance(image, Tensor):
                raise ValueError("Expected image to be of type Tensor, got {}."
                                 .format(type(image)))
            expected_shape = torch.Size((self.config.channels,
                                         self.config.height,
                                         self.config.width))
            if image.shape != expected_shape:
                raise ValueError("Expected images to be tensors of shape {}, got {}."
                                 .format(list(expected_shape), list(image.shape)))

        for target in targets:
            boxes = target['boxes']
            if not isinstance(boxes, Tensor):
                raise ValueError("Expected target boxes to be of type Tensor, got {}."
                                 .format(type(boxes)))
            if (len(boxes.shape) != 2) or (boxes.shape[-1] != 4):
                raise ValueError("Expected target boxes to be tensors of shape [N, 4], got {}."
                                 .format(list(boxes.shape)))
            labels = target['labels']
            if not isinstance(labels, Tensor):
                raise ValueError("Expected target labels to be of type Tensor, got {}."
                                 .format(type(labels)))
            if len(labels.shape) != 1:
                raise ValueError("Expected target labels to be tensors of shape [N], got {}."
                                 .format(list(labels.shape)))

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
                `[batch_size, N, 4]`.
            confidences: Detection confidences in a tensor sized `[batch_size, N]`.
            classprobs: Probabilities of the best classes in a tensor sized `[batch_size, N]`.
            labels: Indices of the best classes in a tensor sized `[batch_size, N]`.

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

            out_boxes.append(img_out_boxes)
            out_confidences.append(img_out_confidences)
            out_classprobs.append(img_out_classprobs)
            out_labels.append(img_out_labels)

        return out_boxes, out_confidences, out_classprobs, out_labels


class Resize:
    """Rescales the image and target to given dimensions.

    Args:
        output_size (tuple or int): Desired output size. If tuple (height, width), the output is
            matched to `output_size`. If int, the smaller of the image edges is matched to
            `output_size`, keeping the aspect ratio the same.
    """

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, image, target):
        width, height = image.size
        original_size = torch.tensor([height, width])
        resize_ratio = torch.tensor(self.output_size) / original_size
        image = F.resize(image, self.output_size)
        scale = torch.tensor([resize_ratio[1],   # y
                              resize_ratio[0],   # x
                              resize_ratio[1],   # y
                              resize_ratio[0]],  # x
                             device=target['boxes'].device)
        target['boxes'] = target['boxes'] * scale
        return image, target


def run_cli():
    from pytorch_lightning.utilities import argparse_utils

    from pl_bolts.datamodules import VOCDetectionDataModule
    from pl_bolts.datamodules.vocdetection_datamodule import Compose

    pl.seed_everything(42)

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help='model configuration file', required=True)
    parser.add_argument('--darknet-weights', type=str, help='initialize the model weights from this Darknet model file')
    parser.add_argument('--batch-size', type=int, help='number of images in one batch', default=16)
    parser = VOCDetectionDataModule.add_argparse_args(parser)
    parser = argparse_utils.add_argparse_args(Yolo, parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = YoloConfiguration(args.config)

    transforms = [Resize((config.height, config.width))]
    image_transforms = T.ToTensor()
    datamodule = VOCDetectionDataModule.from_argparse_args(args)
    datamodule.prepare_data()

    params = vars(args)
    valid_kwargs = inspect.signature(Yolo.__init__).parameters
    kwargs = dict((name, params[name]) for name in valid_kwargs if name in params)
    model = Yolo(configuration=config, **kwargs)
    if args.darknet_weights is not None:
        with open(args.darknet_weights, 'r') as weight_file:
            model.load_darknet_weights(weight_file)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(
        model,
        datamodule.train_dataloader(args.batch_size, transforms, image_transforms),
        datamodule.val_dataloader(args.batch_size, transforms, image_transforms))


if __name__ == "__main__":
    run_cli()
