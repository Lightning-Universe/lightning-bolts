from collections import OrderedDict
from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .layers import Conv, DetectionLayer, MaxPool, ReOrg, create_detection_layer
from .types import NETWORK_OUTPUT, TARGETS
from .utils import get_image_size


def run_detection(
    detection_layer: DetectionLayer,
    layer_input: Tensor,
    targets: Optional[TARGETS],
    image_size: Tensor,
    detections: List[Tensor],
    losses: List[Tensor],
    hits: List[int],
) -> None:
    """Runs the detection layer on the inputs and appends the output to the ``detections`` list.

    If ``targets`` is given, also calculates the losses and appends to the ``losses`` list.

    Args:
        detection_layer: The detection layer.
        layer_input: Input to the detection layer.
        targets: List of training targets for each image.
        image_size: Width and height in a vector that defines the scale of the target coordinates.
        detections: A list where a tensor containing the detections will be appended to.
        losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.
        hits: A list where the number of targets that matched this layer will be appended to, if ``targets`` is given.
    """
    output, preds = detection_layer(layer_input, image_size)
    detections.append(output)

    if targets is not None:
        layer_losses, layer_hits = detection_layer.calculate_losses(preds, targets, image_size)
        losses.append(layer_losses)
        hits.append(layer_hits)


def run_detection_with_aux_head(
    detection_layer: DetectionLayer,
    aux_detection_layer: DetectionLayer,
    layer_input: Tensor,
    aux_input: Tensor,
    targets: Optional[TARGETS],
    image_size: Tensor,
    aux_weight: float,
    detections: List[Tensor],
    losses: List[Tensor],
    hits: List[int],
) -> None:
    """Runs the detection layer on the inputs and appends the output to the ``detections`` list.

    If ``targets`` is given, also runs the auxiliary detection layer on the auxiliary inputs, calculates the losses, and
    appends the losses to the ``losses`` list.

    Args:
        detection_layer: The lead detection layer.
        aux_detection_layer: The auxiliary detection layer.
        layer_input: Input to the lead detection layer.
        aux_input: Input to the auxiliary detection layer.
        targets: List of training targets for each image.
        image_size: Width and height in a vector that defines the scale of the target coordinates.
        aux_weight: Weight of the auxiliary loss.
        detections: A list where a tensor containing the detections will be appended to.
        losses: A list where a tensor containing the losses will be appended to, if ``targets`` is given.
        hits: A list where the number of targets that matched this layer will be appended to, if ``targets`` is given.
    """
    output, preds = detection_layer(layer_input, image_size)
    detections.append(output)

    if targets is not None:
        # Match lead head predictions to targets and calculate losses from lead head outputs.
        layer_losses, layer_hits = detection_layer.calculate_losses(preds, targets, image_size)
        losses.append(layer_losses)
        hits.append(layer_hits)

        # Match lead head predictions to targets and calculate losses from auxiliary head outputs.
        _, aux_preds = aux_detection_layer(aux_input, image_size)
        layer_losses, layer_hits = aux_detection_layer.calculate_losses(
            preds, targets, image_size, loss_preds=aux_preds
        )
        losses.append(layer_losses * aux_weight)
        hits.append(layer_hits)


class BottleneckBlock(nn.Module):
    """A residual block with a bottleneck layer.

    Args:
        in_channels: Number of input channels that the block expects.
        out_channels: Number of output channels that the block produces.
        hidden_channels: Number of output channels the (hidden) bottleneck layer produces. By default the number of
            output channels of the block.
        shortcut: Whether the block should include a shortcut connection.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        shortcut: bool = True,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.convs = nn.Sequential(
            Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm),
            Conv(hidden_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm),
        )
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        y = self.convs(x)
        return x + y if self.shortcut else y


class TinyStage(nn.Module):
    """One stage of the "tiny" network architecture from YOLOv4.

    Args:
        num_channels: Number of channels in the input of the stage. Partial output will have as many channels and full
            output will have twice as many channels.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        num_channels: int,
        activation: Optional[str] = "leaky",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        hidden_channels = num_channels // 2
        self.conv1 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.conv2 = Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=norm)
        self.mix = Conv(num_channels, num_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        partial = torch.chunk(x, 2, dim=1)[1]
        y1 = self.conv1(partial)
        y2 = self.conv2(y1)
        partial_output = self.mix(torch.cat((y2, y1), dim=1))
        full_output = torch.cat((x, partial_output), dim=1)
        return partial_output, full_output


class CSPStage(nn.Module):
    """One stage of a Cross Stage Partial Network (CSPNet).

    Encapsulates a number of bottleneck blocks in the "fusion first" CSP structure.

    `Chien-Yao Wang et al. <https://arxiv.org/abs/1911.11929>`_

    Args:
        in_channels: Number of input channels that the CSP stage expects.
        out_channels: Number of output channels that the CSP stage produces.
        depth: Number of bottleneck blocks that the CSP stage contains.
        shortcut: Whether the bottleneck blocks should include a shortcut connection.
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
    ) -> None:
        super().__init__()

        # Instead of splitting the N output channels of a convolution into two parts, we can equivalently perform two
        # convolutions with N/2 output channels.
        hidden_channels = out_channels // 2

        self.split1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        bottlenecks: List[nn.Module] = [
            BottleneckBlock(hidden_channels, hidden_channels, shortcut=shortcut, norm=norm, activation=activation)
            for _ in range(depth)
        ]
        self.bottlenecks = nn.Sequential(*bottlenecks)
        self.mix = Conv(hidden_channels * 2, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.bottlenecks(self.split1(x))
        y2 = self.split2(x)
        return self.mix(torch.cat((y1, y2), dim=1))


class ELANStage(nn.Module):
    """One stage of an Efficient Layer Aggregation Network (ELAN).

    `Chien-Yao Wang et al. <https://arxiv.org/abs/2211.04800>`_

    Args:
        in_channels: Number of input channels that the ELAN stage expects.
        out_channels: Number of output channels that the ELAN stage produces.
        hidden_channels: Number of output channels that the computational blocks produce. The default value is half the
            number of output channels of the block, as in YOLOv7-W6, but the value varies between the variants.
        split_channels: Number of channels in each part after splitting the input to the cross stage connection and the
            computational blocks. The default value is the number of hidden channels, as in all YOLOv7 backbones. Most
            YOLOv7 heads use twice the number of hidden channels.
        depth: Number of computational blocks that the ELAN stage contains. The default value is 2. YOLOv7 backbones use
            2 to 4 blocks per stage.
        block_depth: Number of convolutional layers in one computational block. The default value is 2. YOLOv7 backbones
            have two convolutions per block. YOLOv7 heads (except YOLOv7-X) have 2 to 8 blocks with only one convolution
            in each.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        split_channels: Optional[int] = None,
        depth: int = 2,
        block_depth: int = 2,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=norm)

        def block(in_channels: int, out_channels: int) -> nn.Module:
            convs = [conv3x3(in_channels, out_channels)]
            for _ in range(block_depth - 1):
                convs.append(conv3x3(out_channels, out_channels))
            return nn.Sequential(*convs)

        # Instead of splitting the N output channels of a convolution into two parts, we can equivalently perform two
        # convolutions with N/2 output channels. However, in many YOLOv7 architectures, the number of hidden channels is
        # not exactly half the number of output channels.
        if hidden_channels is None:
            hidden_channels = out_channels // 2

        if split_channels is None:
            split_channels = hidden_channels

        self.split1 = Conv(in_channels, split_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.split2 = Conv(in_channels, split_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

        blocks = [block(split_channels, hidden_channels)]
        for _ in range(depth - 1):
            blocks.append(block(hidden_channels, hidden_channels))
        self.blocks = nn.ModuleList(blocks)

        total_channels = (split_channels * 2) + (hidden_channels * depth)
        self.mix = Conv(total_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [self.split1(x), self.split2(x)]
        x = outputs[-1]
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return self.mix(torch.cat(outputs, dim=1))


class CSPSPP(nn.Module):
    """Spatial pyramid pooling module from the Cross Stage Partial Network from YOLOv4.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=kernel_size, stride=1, activation=activation, norm=norm)

        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels),
            conv(out_channels, out_channels, kernel_size=3),
            conv(out_channels, out_channels),
        )
        self.conv2 = conv(in_channels, out_channels)

        self.maxpool1 = MaxPool(kernel_size=5, stride=1)
        self.maxpool2 = MaxPool(kernel_size=9, stride=1)
        self.maxpool3 = MaxPool(kernel_size=13, stride=1)

        self.mix1 = nn.Sequential(
            conv(4 * out_channels, out_channels),
            conv(out_channels, out_channels, kernel_size=3),
        )
        self.mix2 = Conv(2 * out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)
        x3 = self.maxpool2(x1)
        x4 = self.maxpool3(x1)
        y1 = self.mix1(torch.cat((x1, x2, x3, x4), dim=1))
        y2 = self.conv2(x)
        return self.mix2(torch.cat((y1, y2), dim=1))


class FastSPP(nn.Module):
    """Fast spatial pyramid pooling module from YOLOv5.

    Args:
        in_channels: Number of input channels that the module expects.
        out_channels: Number of output channels that the module produces.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, activation=activation, norm=norm)
        self.maxpool = MaxPool(kernel_size=5, stride=1)
        self.mix = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, activation=activation, norm=norm)

    def forward(self, x: Tensor) -> Tensor:
        y1 = self.conv(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)
        y4 = self.maxpool(y3)
        return self.mix(torch.cat((y1, y2, y3, y4), dim=1))


class YOLOV4TinyBackbone(nn.Module):
    """Backbone of the "tiny" network architecture from YOLOv4.

    Args:
        in_channels: Number of channels in the input image.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 32,
        activation: Optional[str] = "leaky",
        normalization: Optional[str] = "batchnorm",
    ):
        super().__init__()

        def smooth(num_channels: int) -> nn.Module:
            return Conv(num_channels, num_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            conv_module = Conv(
                in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization
            )
            return nn.Sequential(OrderedDict([("downsample", conv_module), ("smooth", smooth(out_channels))]))

        def maxpool(out_channels: int) -> nn.Module:
            return nn.Sequential(
                OrderedDict(
                    [
                        ("pad", nn.ZeroPad2d((0, 1, 0, 1))),
                        ("maxpool", MaxPool(kernel_size=2, stride=2)),
                        ("smooth", smooth(out_channels)),
                    ]
                )
            )

        def stage(out_channels: int, use_maxpool: bool) -> nn.Module:
            downsample_module = maxpool(out_channels) if use_maxpool else downsample(out_channels // 2, out_channels)
            stage_module = TinyStage(out_channels, activation=activation, norm=normalization)
            return nn.Sequential(OrderedDict([("downsample", downsample_module), ("stage", stage_module)]))

        stages = [
            Conv(in_channels, width, kernel_size=3, stride=2, activation=activation, norm=normalization),
            stage(width * 2, False),
            stage(width * 4, True),
            stage(width * 8, True),
            maxpool(width * 16),
        ]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> List[Tensor]:
        c1 = self.stages[0](x)
        c2, x = self.stages[1](c1)
        c3, x = self.stages[2](x)
        c4, x = self.stages[3](x)
        c5 = self.stages[4](x)
        return [c1, c2, c3, c4, c5]


class YOLOV4Backbone(nn.Module):
    """A backbone that corresponds approximately to the Cross Stage Partial Network from YOLOv4.

    Args:
        in_channels: Number of channels in the input image.
        widths: Number of channels at each network stage. Typically ``(32, 64, 128, 256, 512, 1024)``. The P6 variant
            adds one more stage with 1024 channels.
        depths: Number of bottleneck layers at each network stage. Typically ``(1, 1, 2, 8, 8, 4)``. The P6 variant uses
            ``(1, 1, 3, 15, 15, 7, 7)``.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (32, 64, 128, 256, 512, 1024),
        depths: Sequence[int] = (1, 1, 2, 8, 8, 4),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        if len(widths) != len(depths):
            raise ValueError("Width and depth has to be given for an equal number of stages.")

        def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: int, out_channels: int, depth: int) -> nn.Module:
            csp = CSPStage(
                out_channels,
                out_channels,
                depth=depth,
                shortcut=True,
                activation=activation,
                norm=normalization,
            )
            return nn.Sequential(
                OrderedDict(
                    [
                        ("downsample", downsample(in_channels, out_channels)),
                        ("csp", csp),
                    ]
                )
            )

        convs = [conv3x3(in_channels, widths[0])] + [conv3x3(widths[0], widths[0]) for _ in range(depths[0] - 1)]
        self.stem = nn.Sequential(*convs)
        self.stages = nn.ModuleList(
            stage(in_channels, out_channels, depth)
            for in_channels, out_channels, depth in zip(widths[:-1], widths[1:], depths[1:])
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outputs: List[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


class YOLOV5Backbone(nn.Module):
    """The Cross Stage Partial Network backbone from YOLOv5.

    Args:
        in_channels: Number of channels in the input image.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 16 (yolov5n), 32
            (yolov5s), 48 (yolov5m), 64 (yolov5l), and 80 (yolov5x).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolov5n, yolov5s), 2 (yolov5m), 3 (yolov5l), and 4 (yolov5x).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 3,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def downsample(in_channels: int, out_channels: int, kernel_size: int = 3) -> nn.Module:
            return Conv(
                in_channels, out_channels, kernel_size=kernel_size, stride=2, activation=activation, norm=normalization
            )

        def stage(in_channels: int, out_channels: int, depth: int) -> nn.Module:
            csp = CSPStage(
                out_channels,
                out_channels,
                depth=depth,
                shortcut=True,
                activation=activation,
                norm=normalization,
            )
            return nn.Sequential(
                OrderedDict(
                    [
                        ("downsample", downsample(in_channels, out_channels)),
                        ("csp", csp),
                    ]
                )
            )

        stages = [
            downsample(in_channels, width, kernel_size=6),
            stage(width, width * 2, depth),
            stage(width * 2, width * 4, depth * 2),
            stage(width * 4, width * 8, depth * 3),
            stage(width * 8, width * 16, depth),
        ]
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Tensor) -> List[Tensor]:
        c1 = self.stages[0](x)
        c2 = self.stages[1](c1)
        c3 = self.stages[2](c2)
        c4 = self.stages[3](c3)
        c5 = self.stages[4](c4)
        return [c1, c2, c3, c4, c5]


class YOLOV7Backbone(nn.Module):
    """A backbone that corresponds to the W6 variant of the Efficient Layer Aggregation Network from YOLOv7.

    Args:
        in_channels: Number of channels in the input image.
        widths: Number of channels at each network stage. Before the first stage there will be one extra split of
            spatial resolution by a ``ReOrg`` layer, producing ``in_channels * 4`` channels.
        depth: Number of computational blocks at each network stage. YOLOv7-W6 backbone uses 2.
        block_depth: Number of convolutional layers in one computational block. YOLOv7-W6 backbone uses 2.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: Sequence[int] = (64, 128, 256, 512, 768, 1024),
        depth: int = 2,
        block_depth: int = 2,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def conv3x3(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def stage(in_channels: int, out_channels: int) -> nn.Module:
            elan = ELANStage(
                out_channels,
                out_channels,
                depth=depth,
                block_depth=block_depth,
                activation=activation,
                norm=normalization,
            )
            return nn.Sequential(
                OrderedDict(
                    [
                        ("downsample", downsample(in_channels, out_channels)),
                        ("elan", elan),
                    ]
                )
            )

        self.stem = nn.Sequential(*[ReOrg(), conv3x3(in_channels * 4, widths[0])])
        self.stages = nn.ModuleList(
            stage(in_channels, out_channels) for in_channels, out_channels in zip(widths[:-1], widths[1:])
        )

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outputs: List[Tensor] = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        return outputs


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
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
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
        prior_shapes: Optional[List[Tuple[int, int]]] = None,
        **kwargs: Any,
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

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=normalization)

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def outputs(in_channels: int) -> nn.Module:
            return nn.Conv2d(in_channels, num_outputs, kernel_size=1, stride=1, bias=True)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        self.backbone = backbone or YOLOV4TinyBackbone(width=width, activation=activation, normalization=normalization)

        self.fpn5 = conv(width * 16, width * 8)
        self.out5 = nn.Sequential(
            OrderedDict(
                [
                    ("channels", conv(width * 8, width * 16)),
                    (f"outputs_{num_outputs}", outputs(width * 16)),
                ]
            )
        )
        self.upsample5 = upsample(width * 8, width * 4)

        self.fpn4 = conv(width * 12, width * 8, kernel_size=3)
        self.out4 = nn.Sequential(OrderedDict([(f"outputs_{num_outputs}", outputs(width * 8))]))
        self.upsample4 = upsample(width * 8, width * 2)

        self.fpn3 = conv(width * 6, width * 4, kernel_size=3)
        self.out3 = nn.Sequential(OrderedDict([(f"outputs_{num_outputs}", outputs(width * 4))]))

        self.detect3 = detect([0, 1, 2])
        self.detect4 = detect([3, 4, 5])
        self.detect5 = detect([6, 7, 8])

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5 = self.backbone(x)[-3:]

        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample5(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), c3), dim=1)
        p3 = self.fpn3(x)

        run_detection(self.detect5, self.out5(p5), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(p4), targets, image_size, detections, losses, hits)
        run_detection(self.detect3, self.out3(p3), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV4Network(nn.Module):
    """Network architecture that corresponds approximately to the Cross Stage Partial Network from YOLOv4.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        widths: Sequence[int] = (32, 64, 128, 256, 512, 1024),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[List[Tuple[int, int]]] = None,
        **kwargs: Any,
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

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=2,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def out(in_channels: int) -> nn.Module:
            conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([("conv", conv), (f"outputs_{num_outputs}", outputs)]))

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV4Backbone(widths=widths, activation=activation, normalization=normalization)

        w3 = widths[-3]
        w4 = widths[-2]
        w5 = widths[-1]

        self.spp = spp(w5, w5)

        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5, w4 // 2)
        self.fpn4 = csp(w4, w4)

        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4, w3 // 2)
        self.fpn3 = csp(w3, w3)

        self.downsample3 = downsample(w3, w3)
        self.pan4 = csp(w3 + w4, w4)

        self.downsample4 = downsample(w4, w4)
        self.pan5 = csp(w4 + w5, w5)

        self.out3 = out(w3)
        self.out4 = out(w4)
        self.out5 = out(w5)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)

        x = torch.cat((self.upsample5(c5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), c5), dim=1)
        n5 = self.pan5(x)

        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV4P6Network(nn.Module):
    """Network architecture that corresponds approximately to the variant of YOLOv4 with four detection layers.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        widths: Sequence[int] = (32, 64, 128, 256, 512, 1024, 1024),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[List[Tuple[int, int]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                (13, 17),
                (31, 25),
                (24, 51),
                (61, 45),
                (61, 45),
                (48, 102),
                (119, 96),
                (97, 189),
                (97, 189),
                (217, 184),
                (171, 384),
                (324, 451),
                (324, 451),
                (545, 357),
                (616, 618),
                (1024, 1024),
            ]
            anchors_per_cell = 4
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 4)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 4.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=2,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def out(in_channels: int) -> nn.Module:
            conv = Conv(in_channels, in_channels, kernel_size=3, stride=1, activation=activation, norm=normalization)
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([("conv", conv), (f"outputs_{num_outputs}", outputs)]))

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV4Backbone(
                widths=widths, depths=(1, 1, 3, 15, 15, 7, 7), activation=activation, normalization=normalization
            )

        w3 = widths[-4]
        w4 = widths[-3]
        w5 = widths[-2]
        w6 = widths[-1]

        self.spp = spp(w6, w6)

        self.pre5 = conv(w5, w5 // 2)
        self.upsample6 = upsample(w6, w5 // 2)
        self.fpn5 = csp(w5, w5)

        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5, w4 // 2)
        self.fpn4 = csp(w4, w4)

        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4, w3 // 2)
        self.fpn3 = csp(w3, w3)

        self.downsample3 = downsample(w3, w3)
        self.pan4 = csp(w3 + w4, w4)

        self.downsample4 = downsample(w4, w4)
        self.pan5 = csp(w4 + w5, w5)

        self.downsample5 = downsample(w5, w5)
        self.pan6 = csp(w5 + w6, w6)

        self.out3 = out(w3)
        self.out4 = out(w4)
        self.out5 = out(w5)
        self.out6 = out(w6)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))
        self.detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5, x = self.backbone(x)[-4:]
        c6 = self.spp(x)

        x = torch.cat((self.upsample6(c6), self.pre5(c5)), dim=1)
        p5 = self.fpn5(x)
        x = torch.cat((self.upsample5(p5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        x = torch.cat((self.downsample5(n5), c6), dim=1)
        n6 = self.pan6(x)

        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        run_detection(self.detect6, self.out6(n6), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV5Network(nn.Module):
    """The YOLOv5 network architecture. Different variants (n/s/m/l/x) can be achieved by adjusting the ``depth``
    and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 16 (yolov5n), 32
            (yolov5s), 48 (yolov5m), 64 (yolov5l), and 80 (yolov5x).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolov5n, yolov5s), 2 (yolov5m), 3 (yolov5l), and 4 (yolov5x).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        width: int = 64,
        depth: int = 3,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[List[Tuple[int, int]]] = None,
        **kwargs: Any,
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

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return FastSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def out(in_channels: int) -> nn.Module:
            outputs = nn.Conv2d(in_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([(f"outputs_{num_outputs}", outputs)]))

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=depth,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        self.backbone = backbone or YOLOV5Backbone(
            depth=depth, width=width, activation=activation, normalization=normalization
        )

        self.spp = spp(width * 16, width * 16)

        self.pan3 = csp(width * 8, width * 4)
        self.out3 = out(width * 4)

        self.fpn4 = nn.Sequential(
            OrderedDict(
                [
                    ("csp", csp(width * 16, width * 8)),
                    ("conv", conv(width * 8, width * 4)),
                ]
            )
        )
        self.pan4 = csp(width * 8, width * 8)
        self.out4 = out(width * 8)

        self.fpn5 = conv(width * 16, width * 8)
        self.pan5 = csp(width * 16, width * 16)
        self.out5 = out(width * 16)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)

        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)

        n3 = self.pan3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)

        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits


class YOLOV7Network(nn.Module):
    """Network architecture that corresponds to the W6 variant of YOLOv7 with four detection layers.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        widths: Number of channels at each network stage.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        aux_weight: Weight for the loss from the auxiliary heads.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.

    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        widths: Sequence[int] = (64, 128, 256, 512, 768, 1024),
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[List[Tuple[int, int]]] = None,
        aux_weight: float = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.aux_weight = aux_weight

        # By default use the prior shapes that have been learned from the COCO data.
        if prior_shapes is None:
            prior_shapes = [
                (13, 17),
                (31, 25),
                (24, 51),
                (61, 45),
                (61, 45),
                (48, 102),
                (119, 96),
                (97, 189),
                (97, 189),
                (217, 184),
                (171, 384),
                (324, 451),
                (324, 451),
                (545, 357),
                (616, 618),
                (1024, 1024),
            ]
            anchors_per_cell = 4
        else:
            anchors_per_cell, modulo = divmod(len(prior_shapes), 4)
            if modulo != 0:
                raise ValueError("The number of provided prior shapes needs to be divisible by 4.")
        num_outputs = (5 + num_classes) * anchors_per_cell

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=1, stride=1, activation=activation, norm=normalization)

        def elan(in_channels: int, out_channels: int) -> nn.Module:
            return ELANStage(
                in_channels,
                out_channels,
                split_channels=out_channels,
                depth=4,
                block_depth=1,
                norm=normalization,
                activation=activation,
            )

        def out(in_channels: int, hidden_channels: int) -> nn.Module:
            conv = Conv(
                in_channels, hidden_channels, kernel_size=3, stride=1, activation=activation, norm=normalization
            )
            outputs = nn.Conv2d(hidden_channels, num_outputs, kernel_size=1)
            return nn.Sequential(OrderedDict([("conv", conv), (f"outputs_{num_outputs}", outputs)]))

        def upsample(in_channels: int, out_channels: int) -> nn.Module:
            channels = conv(in_channels, out_channels)
            upsample = nn.Upsample(scale_factor=2, mode="nearest")
            return nn.Sequential(OrderedDict([("channels", channels), ("upsample", upsample)]))

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def detect(prior_shape_idxs: Sequence[int], range: float) -> DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(
                prior_shapes,
                prior_shape_idxs,
                spatial_range=range,
                num_classes=num_classes,
                input_is_normalized=False,
                **kwargs,
            )

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = YOLOV7Backbone(
                widths=widths, depth=2, block_depth=2, activation=activation, normalization=normalization
            )

        w3 = widths[-4]
        w4 = widths[-3]
        w5 = widths[-2]
        w6 = widths[-1]

        self.spp = spp(w6, w6 // 2)

        self.pre5 = conv(w5, w5 // 2)
        self.upsample6 = upsample(w6 // 2, w5 // 2)
        self.fpn5 = elan(w5, w5 // 2)

        self.pre4 = conv(w4, w4 // 2)
        self.upsample5 = upsample(w5 // 2, w4 // 2)
        self.fpn4 = elan(w4, w4 // 2)

        self.pre3 = conv(w3, w3 // 2)
        self.upsample4 = upsample(w4 // 2, w3 // 2)
        self.fpn3 = elan(w3, w3 // 2)

        self.downsample3 = downsample(w3 // 2, w4 // 2)
        self.pan4 = elan(w4, w4 // 2)

        self.downsample4 = downsample(w4 // 2, w5 // 2)
        self.pan5 = elan(w5, w5 // 2)

        self.downsample5 = downsample(w5 // 2, w6 // 2)
        self.pan6 = elan(w6, w6 // 2)

        self.out3 = out(w3 // 2, w3)
        self.aux_out3 = out(w3 // 2, w3 + (w3 // 4))
        self.out4 = out(w4 // 2, w4)
        self.aux_out4 = out(w4 // 2, w4 + (w4 // 4))
        self.out5 = out(w5 // 2, w5)
        self.aux_out5 = out(w5 // 2, w5 + (w5 // 4))
        self.out6 = out(w6 // 2, w6)
        self.aux_out6 = out(w6 // 2, w6 + (w6 // 4))

        self.detect3 = detect(range(0, anchors_per_cell), 5.0)
        self.aux_detect3 = detect(range(0, anchors_per_cell), 3.0)
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2), 5.0)
        self.aux_detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2), 3.0)
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3), 5.0)
        self.aux_detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3), 3.0)
        self.detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4), 5.0)
        self.aux_detect6 = detect(range(anchors_per_cell * 3, anchors_per_cell * 4), 3.0)

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, c5, x = self.backbone(x)[-4:]
        c6 = self.spp(x)

        x = torch.cat((self.upsample6(c6), self.pre5(c5)), dim=1)
        p5 = self.fpn5(x)
        x = torch.cat((self.upsample5(p5), self.pre4(c4)), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample4(p4), self.pre3(c3)), dim=1)
        n3 = self.fpn3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)
        x = torch.cat((self.downsample5(n5), c6), dim=1)
        n6 = self.pan6(x)

        run_detection_with_aux_head(
            self.detect3,
            self.aux_detect3,
            self.out3(n3),
            self.aux_out3(n3),
            targets,
            image_size,
            self.aux_weight,
            detections,
            losses,
            hits,
        )
        run_detection_with_aux_head(
            self.detect4,
            self.aux_detect4,
            self.out4(n4),
            self.aux_out4(p4),
            targets,
            image_size,
            self.aux_weight,
            detections,
            losses,
            hits,
        )
        run_detection_with_aux_head(
            self.detect5,
            self.aux_detect5,
            self.out5(n5),
            self.aux_out5(p5),
            targets,
            image_size,
            self.aux_weight,
            detections,
            losses,
            hits,
        )
        run_detection_with_aux_head(
            self.detect6,
            self.aux_detect6,
            self.out6(n6),
            self.aux_out6(c6),
            targets,
            image_size,
            self.aux_weight,
            detections,
            losses,
            hits,
        )
        return detections, losses, hits


class YOLOXHead(nn.Module):
    """A module that produces features for YOLO detection layer, decoupling the classification and localization
    features.

    Args:
        in_channels: Number of input channels that the module expects.
        hidden_channels: Number of output channels in the hidden layers.
        anchors_per_cell: Number of detections made at each spatial location of the feature map.
        num_classes: Number of different classes that this model predicts.
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        norm: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        anchors_per_cell: int,
        num_classes: int,
        activation: Optional[str] = "silu",
        norm: Optional[str] = "batchnorm",
    ) -> None:
        super().__init__()

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=norm)

        def linear(in_channels: int, out_channels: int) -> nn.Module:
            return nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def features(num_channels: int) -> nn.Module:
            return nn.Sequential(
                conv(num_channels, num_channels, kernel_size=3),
                conv(num_channels, num_channels, kernel_size=3),
            )

        def classprob(num_channels: int) -> nn.Module:
            num_outputs = anchors_per_cell * num_classes
            outputs = linear(num_channels, num_outputs)
            return nn.Sequential(OrderedDict([("convs", features(num_channels)), (f"outputs_{num_outputs}", outputs)]))

        self.stem = conv(in_channels, hidden_channels)
        self.feat = features(hidden_channels)
        self.box = linear(hidden_channels, anchors_per_cell * 4)
        self.confidence = linear(hidden_channels, anchors_per_cell)
        self.classprob = classprob(hidden_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        features = self.feat(x)
        box = self.box(features)
        confidence = self.confidence(features)
        classprob = self.classprob(x)
        return torch.cat((box, confidence, classprob), dim=1)


class YOLOXNetwork(nn.Module):
    """The YOLOX network architecture. Different variants (nano/tiny/s/m/l/x) can be achieved by adjusting the
    ``depth`` and ``width`` parameters.

    Args:
        num_classes: Number of different classes that this model predicts.
        backbone: A backbone network that returns the output from each stage.
        width: Number of channels in the narrowest convolutional layer. The wider convolutional layers will use a number
            of channels that is a multiple of this value. The values used by the different variants are 24 (yolox-tiny),
            32 (yolox-s), 48 (yolox-m), and 64 (yolox-l).
        depth: Repeat the bottleneck layers this many times. Can be used to make the network deeper. The values used by
            the different variants are 1 (yolox-tiny, yolox-s), 2 (yolox-m), and 3 (yolox-l).
        activation: Which layer activation to use. Can be "relu", "leaky", "mish", "silu" (or "swish"), "logistic",
            "linear", or "none".
        normalization: Which layer normalization to use. Can be "batchnorm", "groupnorm", or "none".
        prior_shapes: A list of prior box dimensions, used for scaling the predicted dimensions and possibly for
            matching the targets to the anchors. The list should contain (width, height) tuples in the network input
            resolution. There should be `3N` tuples, where `N` defines the number of anchors per spatial location. They
            are assigned to the layers from the lowest (high-resolution) to the highest (low-resolution) layer, meaning
            that you typically want to sort the shapes from the smallest to the largest.
        matching_algorithm: Which algorithm to use for matching targets to anchors. "simota" (the SimOTA matching rule
            from YOLOX), "size" (match those prior shapes, whose width and height relative to the target is below given
            ratio), "iou" (match all prior shapes that give a high enough IoU), or "maxiou" (match the prior shape that
            gives the highest IoU, default).
        matching_threshold: Threshold for "size" and "iou" matching algorithms.
        spatial_range: The "simota" matching algorithm will restrict to anchors that are within an `N × N` grid cell
            area centered at the target, where `N` is the value of this parameter.
        size_range: The "simota" matching algorithm will restrict to anchors whose dimensions are no more than `N` and
            no less than `1/N` times the target dimensions, where `N` is the value of this parameter.
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has IoU
            with some target greater than this threshold, the predictor will not be taken into account when calculating
            the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou" (default).
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.
        xy_scale: Eliminate "grid sensitivity" by scaling the box coordinates by this factor. Using a value > 1.0 helps
            to produce coordinate values close to one.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[nn.Module] = None,
        width: int = 64,
        depth: int = 3,
        activation: Optional[str] = "silu",
        normalization: Optional[str] = "batchnorm",
        prior_shapes: Optional[List[Tuple[int, int]]] = None,
        **kwargs: Any,
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

        def spp(in_channels: int, out_channels: int) -> nn.Module:
            return FastSPP(in_channels, out_channels, activation=activation, norm=normalization)

        def downsample(in_channels: int, out_channels: int) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size=3, stride=2, activation=activation, norm=normalization)

        def conv(in_channels: int, out_channels: int, kernel_size: int = 1) -> nn.Module:
            return Conv(in_channels, out_channels, kernel_size, stride=1, activation=activation, norm=normalization)

        def csp(in_channels: int, out_channels: int) -> nn.Module:
            return CSPStage(
                in_channels,
                out_channels,
                depth=depth,
                shortcut=False,
                norm=normalization,
                activation=activation,
            )

        def head(in_channels: int, hidden_channels: int) -> YOLOXHead:
            return YOLOXHead(
                in_channels,
                hidden_channels,
                anchors_per_cell,
                num_classes,
                activation=activation,
                norm=normalization,
            )

        def detect(prior_shape_idxs: Sequence[int]) -> DetectionLayer:
            assert prior_shapes is not None
            return create_detection_layer(
                prior_shapes, prior_shape_idxs, num_classes=num_classes, input_is_normalized=False, **kwargs
            )

        self.backbone = backbone or YOLOV5Backbone(
            depth=depth, width=width, activation=activation, normalization=normalization
        )

        self.spp = spp(width * 16, width * 16)

        self.pan3 = csp(width * 8, width * 4)
        self.out3 = head(width * 4, width * 4)

        self.fpn4 = nn.Sequential(
            OrderedDict(
                [
                    ("csp", csp(width * 16, width * 8)),
                    ("conv", conv(width * 8, width * 4)),
                ]
            )
        )
        self.pan4 = csp(width * 8, width * 8)
        self.out4 = head(width * 8, width * 4)

        self.fpn5 = conv(width * 16, width * 8)
        self.pan5 = csp(width * 16, width * 16)
        self.out5 = head(width * 16, width * 4)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.downsample3 = downsample(width * 4, width * 4)
        self.downsample4 = downsample(width * 8, width * 8)

        self.detect3 = detect(range(0, anchors_per_cell))
        self.detect4 = detect(range(anchors_per_cell, anchors_per_cell * 2))
        self.detect5 = detect(range(anchors_per_cell * 2, anchors_per_cell * 3))

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        c3, c4, x = self.backbone(x)[-3:]
        c5 = self.spp(x)

        p5 = self.fpn5(c5)
        x = torch.cat((self.upsample(p5), c4), dim=1)
        p4 = self.fpn4(x)
        x = torch.cat((self.upsample(p4), c3), dim=1)

        n3 = self.pan3(x)
        x = torch.cat((self.downsample3(n3), p4), dim=1)
        n4 = self.pan4(x)
        x = torch.cat((self.downsample4(n4), p5), dim=1)
        n5 = self.pan5(x)

        run_detection(self.detect3, self.out3(n3), targets, image_size, detections, losses, hits)
        run_detection(self.detect4, self.out4(n4), targets, image_size, detections, losses, hits)
        run_detection(self.detect5, self.out5(n5), targets, image_size, detections, losses, hits)
        return detections, losses, hits
