from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from pl_bolts.models.detection.yolo.utils import get_image_size
from pl_bolts.models.detection.yolo.yolo_layers import Conv, create_detection_layer


class Bottleneck(nn.Module):
    """A bottleneck from YOLOv5.

    Args:
        in_channels: Number of input channels that the bottleneck expects.
        out_channels: Number of output channels that the bottleneck produces.
        shortcut: Whether the bottleneck should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut: bool = True,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()
        self.convs = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm),
            Conv(out_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm),
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.convs(x)
        return x + y if self.shortcut else y


class TinyStage(nn.Module):
    """One stage of the "tiny" network architecture from YOLOv4.

    Args:
        num_channels: Number of channels in the stage input and output.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        num_channels: int,
        activation: Optional[str] = "leaky",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()

        hidden_channels = num_channels // 2
        self.conv1 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.mix = Conv(num_channels, num_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x):
        x = torch.chunk(x, 2, dim=1)[1]
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        return self.mix(torch.cat((y2, y1), dim=1))


class CSPStage(nn.Module):
    """One stage of a Cross Stage Partial Network (CSPNet).

    Args:
        in_channels: Number of input channels that the stage expects.
        out_channels: Number of output channels that the stage produces.
        depth: Number of bottlenecks that the stage contains.
        shortcut: Whether the bottlenecks should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 1,
        shortcut: bool = True,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()
        # Instead of splitting the N output channels of a convolution into two parts, we can equivalently perform two
        # convolutions with N/2 output channels.
        self.split1 = Conv(in_channels, out_channels // 2, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, out_channels // 2, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.mix = Conv(out_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.bottlenecks = nn.Sequential(
            *(
                Bottleneck(out_channels // 2, out_channels // 2, shortcut, norm=norm, activation=activation)
                for _ in range(depth)
            )
        )

    def forward(self, x):
        y1 = self.bottlenecks(self.split1(x))
        y2 = self.split2(x)
        return self.mix(torch.cat((y1, y2), dim=1))


class FastSPP(nn.Module):
    """Fast spatial pyramid pooling module.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        kernel_size: Kernel size for convolutional layers.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.mix = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x):
        y1 = self.conv(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        y4 = self.maxpool(y3)
        return self.mix(torch.cat((y1, y2, y3, y4), dim=1))


class TinyBackbone(nn.Module):
    """Backbone of the "tiny" network architecture from YOLOv4.

    Args:
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        width: int = 32,
        activation: Optional[str] = "leaky",
        normalization: Optional[str] = "batchnorm",
    ):
        super().__init__()

        def smooth(num_channels):
            return Conv(num_channels, num_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels, out_channels):
            conv = Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)
            return nn.Sequential(
                OrderedDict(
                    [
                        ("conv", conv),
                        ("smooth", smooth(out_channels)),
                    ]
                )
            )

        def maxpool(out_channels):
            return nn.Sequential(
                OrderedDict(
                    [
                        ("pad", nn.ZeroPad2d((0, 1, 0, 1))),
                        ("maxpool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)),
                        ("smooth", smooth(out_channels)),
                    ]
                )
            )

        self.stage1 = Conv(3, width, kernel_size=3, stride=2, activation=activation, norm=normalization)
        self.downsample2 = downsample(width, width * 2)
        self.stage2 = TinyStage(width * 2, activation=activation, norm=normalization)
        self.downsample3 = maxpool(width * 4)
        self.stage3 = TinyStage(width * 4, activation=activation, norm=normalization)
        self.downsample4 = maxpool(width * 8)
        self.stage4 = TinyStage(width * 8, activation=activation, norm=normalization)
        self.downsample5 = maxpool(width * 16)

    def forward(self, x):
        c1 = self.stage1(x)
        x = self.downsample2(c1)
        c2 = self.stage2(x)
        x = torch.cat((x, c2), dim=1)
        x = self.downsample3(x)
        c3 = self.stage3(x)
        x = torch.cat((x, c3), dim=1)
        x = self.downsample4(x)
        c4 = self.stage4(x)
        x = torch.cat((x, c4), dim=1)
        c5 = self.downsample5(x)
        return c1, c2, c3, c4, c5


class CSPBackbone(nn.Module):
    """The Cross Stage Partial Network backbone from YOLOv5.

    Args:
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper.
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
    """

    def __init__(
        self,
        depth: int = 3,
        width: int = 64,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def downsample(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def csp(num_channels, depth):
            return CSPStage(num_channels, num_channels, depth=depth)

        def spp(num_channels):
            return FastSPP(num_channels, num_channels, kernel_size=5, activation=activation, norm=normalization)

        self.stage1 = Conv(3, width, kernel_size=6, stride=2, padding=2, activation=activation, norm=normalization)
        self.stage2 = nn.Sequential(
            OrderedDict(
                [
                    ("downsample", downsample(width, width * 2)),
                    ("csp", csp(width * 2, depth)),
                ]
            )
        )
        self.stage3 = nn.Sequential(
            OrderedDict(
                [
                    ("downsample", downsample(width * 2, width * 4)),
                    ("csp", csp(width * 4, depth * 2)),
                ]
            )
        )
        self.stage4 = nn.Sequential(
            OrderedDict(
                [
                    ("downsample", downsample(width * 4, width * 8)),
                    ("csp", csp(width * 8, depth * 3)),
                ]
            )
        )
        self.stage5 = nn.Sequential(
            OrderedDict(
                [
                    ("downsample", downsample(width * 8, width * 16)),
                    ("csp", csp(width * 16, depth)),
                    ("spp", spp(width * 16)),
                ]
            )
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c1, c2, c3, c4, c5


class YOLOV4TinyNetwork(nn.Module):
    """The "tiny" network architecture from YOLOv4.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per grid cell. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a tensor with as many elements as there are input boxes. Valid values for a string are
            "iou", "giou", "diou", and "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that target
            confidence is one if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        width: int = 32,
        activation: Optional[str] = "leaky",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: List[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                (12, 16),
                (19, 36),
                (40, 28),
                (36, 75),
                (76, 55),
                (72, 146),
                (142, 110),
                (192, 243),
                (459, 401),
            ]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def conv1x1(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def conv3x3(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def linear(in_channels, out_channels):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        def detect(prior_shape_idxs):
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        self.backbone = backbone or TinyBackbone(width=width, normalization=normalization, activation=activation)

        self.lateral5 = conv1x1(width * 16, width * 8)
        self.out5 = nn.Sequential(
            conv3x3(width * 8, width * 16),
            linear(width * 16, num_outputs),
        )
        self.upsample5 = nn.Sequential(
            conv1x1(width * 8, width * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.lateral4 = conv3x3(width * 12, width * 8)
        self.out4 = linear(width * 8, num_outputs)
        self.upsample4 = nn.Sequential(
            conv1x1(width * 8, width * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )

        self.lateral3 = conv3x3(width * 6, width * 4)
        self.out3 = linear(width * 4, num_outputs)

        self.detect3 = detect([0, 1, 2])
        self.detect4 = detect([3, 4, 5])
        self.detect5 = detect([6, 7, 8])

    def forward(self, x: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Tensor, Tensor]:
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5 = self.backbone(x)[-3:]

        p5 = self.lateral5(c5)
        x = torch.cat((self.upsample5(p5), c4), dim=1)
        p4 = self.lateral4(x)
        x = torch.cat((self.upsample4(p4), c3), dim=1)
        p3 = self.lateral3(x)

        y = self.detect5(self.out5(p5), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect5.losses)
            hits.append(self.detect5.hits)

        y = self.detect4(self.out4(p4), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect4.losses)
            hits.append(self.detect4.hits)

        y = self.detect3(self.out3(p3), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect3.losses)
            hits.append(self.detect3.hits)

        return detections, losses, hits


class YOLOV5Network(nn.Module):
    """The YOLOv5 network architecture. Different variants (n/s/m/l/x) can be achieved by adjusting the ``depth``
    and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper.
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per grid cell. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a tensor with as many elements as there are input boxes. Valid values for a string are
            "iou", "giou", "diou", and "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that target
            confidence is one if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        depth: int = 3,
        width: int = 64,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: List[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                (12, 16),
                (19, 36),
                (40, 28),
                (36, 75),
                (76, 55),
                (72, 146),
                (142, 110),
                (192, 243),
                (459, 401),
            ]
            anchors_per_cell = 3
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def downsample(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv1x1(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def linear(in_channels, out_channels):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def bottleneck(in_channels, out_channels):
            return CSPStage(
                in_channels,
                out_channels,
                depth=depth,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def detect(prior_shape_idxs):
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        self.backbone = backbone or CSPBackbone(
            depth=depth, width=width, normalization=normalization, activation=activation
        )

        self.lateral3 = bottleneck(width * 8, width * 4)
        self.out3 = linear(width * 4, num_outputs)

        self.lateral4a = nn.Sequential(
            bottleneck(width * 16, width * 8),
            conv1x1(width * 8, width * 4),
        )
        self.lateral4b = bottleneck(width * 8, width * 8)
        self.out4 = linear(width * 8, num_outputs)

        self.lateral5a = conv1x1(width * 16, width * 8)
        self.lateral5b = bottleneck(width * 16, width * 16)
        self.out5 = linear(width * 16, num_outputs)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Tensor, Tensor]:
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5 = self.backbone(x)[-3:]

        p5 = self.lateral5a(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.lateral4a(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)

        n3 = self.lateral3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.lateral4b(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.lateral5b(x)

        y = self.detect3(self.out3(n3), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect3.losses)
            hits.append(self.detect3.hits)

        y = self.detect4(self.out4(n4), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect4.losses)
            hits.append(self.detect4.hits)

        y = self.detect5(self.out5(n5), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect5.losses)
            hits.append(self.detect5.hits)

        return detections, losses, hits


class YOLOXNetwork(nn.Module):
    """The YOLOX network architecture. Different variants (nano/tiny/s/m/l/x) can be achieved by adjusting the
    ``depth`` and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper.
        width: The number of channels in the narrowest convolutional layer. The wider convolutional layers will use a
            number of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per grid cell. They are
            assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning that
            you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a tensor with as many elements as there are input boxes. Valid values for a string are
            "iou", "giou", "diou", and "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that target
            confidence is one if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        depth: int = 3,
        width: int = 64,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: List[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # By default use one anchor per cell and the stride as the prior size.
        if prior_shapes is None:
            prior_shapes = [(8, 8), (16, 16), (32, 32)]
            anchors_per_cell = 1
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 3)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 3.")

        def downsample(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv1x1(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def conv3x3(in_channels, out_channels):
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def linear(in_channels, out_channels):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def bottleneck(in_channels, out_channels):
            return CSPStage(
                in_channels,
                out_channels,
                depth=depth,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def detect(prior_shape_idxs):
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        self.backbone = backbone or CSPBackbone(
            depth=depth, width=width, normalization=normalization, activation=activation
        )

        self.lateral3 = bottleneck(width * 8, width * 4)
        self.out3_stem = conv1x1(width * 4, width * 4)
        self.out3_feat = nn.Sequential(
            conv3x3(width * 4, width * 4),
            conv3x3(width * 4, width * 4),
        )
        self.out3_box = linear(width * 4, anchors_per_cell * 4)
        self.out3_confidence = linear(width * 4, anchors_per_cell)
        self.out3_classprob = nn.Sequential(
            conv3x3(width * 4, width * 4),
            conv3x3(width * 4, width * 4),
            linear(width * 4, anchors_per_cell * num_classes),
        )

        self.lateral4a = nn.Sequential(
            bottleneck(width * 16, width * 8),
            conv1x1(width * 8, width * 4),
        )
        self.lateral4b = bottleneck(width * 8, width * 8)
        self.out4_stem = conv1x1(width * 8, width * 4)
        self.out4_feat = nn.Sequential(
            conv3x3(width * 4, width * 4),
            conv3x3(width * 4, width * 4),
        )
        self.out4_box = linear(width * 4, anchors_per_cell * 4)
        self.out4_confidence = linear(width * 4, anchors_per_cell)
        self.out4_classprob = nn.Sequential(
            conv3x3(width * 4, width * 4),
            conv3x3(width * 4, width * 4),
            linear(width * 4, anchors_per_cell * num_classes),
        )

        self.lateral5a = conv1x1(width * 16, width * 8)
        self.lateral5b = bottleneck(width * 16, width * 16)
        self.out5_stem = conv1x1(width * 16, width * 4)
        self.out5_feat = nn.Sequential(
            conv3x3(width * 4, width * 4),
            conv3x3(width * 4, width * 4),
        )
        self.out5_box = linear(width * 4, anchors_per_cell * 4)
        self.out5_confidence = linear(width * 4, anchors_per_cell)
        self.out5_classprob = nn.Sequential(
            conv3x3(width * 4, width * 4),
            conv3x3(width * 4, width * 4),
            linear(width * 4, anchors_per_cell * num_classes),
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Tensor, Tensor]:
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5 = self.backbone(x)[-3:]

        p5 = self.lateral5a(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.lateral4a(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)

        n3 = self.lateral3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.lateral4b(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.lateral5b(x)

        x = self.out3_stem(n3)
        features = self.out3_feat(x)
        box = self.out3_box(features)
        confidence = self.out3_confidence(features)
        classprob = self.out3_classprob(x)
        y = self.detect3(torch.cat((box, confidence, classprob), dim=1), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect3.losses)
            hits.append(self.detect3.hits)

        x = self.out4_stem(n4)
        features = self.out4_feat(x)
        box = self.out4_box(features)
        confidence = self.out4_confidence(features)
        classprob = self.out4_classprob(x)
        y = self.detect4(torch.cat((box, confidence, classprob), dim=1), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect4.losses)
            hits.append(self.detect4.hits)

        x = self.out5_stem(n5)
        features = self.out5_feat(x)
        box = self.out5_box(features)
        confidence = self.out5_confidence(features)
        classprob = self.out5_classprob(x)
        y = self.detect5(torch.cat((box, confidence, classprob), dim=1), image_size, targets)
        detections.append(y)
        if targets is not None:
            losses.append(self.detect5.losses)
            hits.append(self.detect5.hits)

        return detections, losses, hits
