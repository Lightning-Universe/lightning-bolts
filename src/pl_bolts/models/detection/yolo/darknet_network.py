import io
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_info
except ModuleNotFoundError:
    from pytorch_lightning.utilities.distributed import rank_zero_info

from torch import Tensor

from .layers import Conv, DetectionLayer, MaxPool, RouteLayer, ShortcutLayer, create_detection_layer
from .torch_networks import NETWORK_OUTPUT
from .types import TARGETS
from .utils import get_image_size

CONFIG = Dict[str, Any]
CREATE_LAYER_OUTPUT = Tuple[nn.Module, int]  # layer, num_outputs


class DarknetNetwork(nn.Module):
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation.

    Iterates through the layers from the configuration and creates corresponding PyTorch modules. If ``weights_path`` is
    given and points to a Darknet model file, loads the convolutional layer weights from the file.

    Args:
        config_path: Path to a Darknet configuration file that defines the network architecture.
        weights_path: Path to a Darknet model file. If given, the model weights will be read from this file.
        in_channels: Number of channels in the input image.
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
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    """

    def __init__(
        self, config_path: str, weights_path: Optional[str] = None, in_channels: Optional[int] = None, **kwargs: Any
    ) -> None:
        super().__init__()

        with open(config_path) as config_file:
            sections = self._read_config(config_file)

        if len(sections) < 2:
            raise MisconfigurationException("The model configuration file should include at least two sections.")

        self.__dict__.update(sections[0])
        global_config = sections[0]
        layer_configs = sections[1:]

        if in_channels is None:
            in_channels = global_config.get("channels", 3)
            assert isinstance(in_channels, int)

        self.layers = nn.ModuleList()
        # num_inputs will contain the number of channels in the input of every layer up to the current layer. It is
        # initialized with the number of channels in the input image.
        num_inputs = [in_channels]
        for layer_config in layer_configs:
            config = {**global_config, **layer_config}
            layer, num_outputs = _create_layer(config, num_inputs, **kwargs)
            self.layers.append(layer)
            num_inputs.append(num_outputs)

        if weights_path is not None:
            with open(weights_path) as weight_file:
                self.load_weights(weight_file)

    def forward(self, x: Tensor, targets: Optional[TARGETS] = None) -> NETWORK_OUTPUT:
        outputs: List[Tensor] = []  # Outputs from all layers
        detections: List[Tensor] = []  # Outputs from detection layers
        losses: List[Tensor] = []  # Losses from detection layers
        hits: List[int] = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        for layer in self.layers:
            if isinstance(layer, (RouteLayer, ShortcutLayer)):
                x = layer(outputs)
            elif isinstance(layer, DetectionLayer):
                x, preds = layer(x, image_size)
                detections.append(x)
                if targets is not None:
                    layer_losses, layer_hits = layer.calculate_losses(preds, targets, image_size)
                    losses.append(layer_losses)
                    hits.append(layer_hits)
            else:
                x = layer(x)

            outputs.append(x)

        return detections, losses, hits

    def load_weights(self, weight_file: io.IOBase) -> None:
        """Loads weights to layer modules from a pretrained Darknet model.

        One may want to continue training from pretrained weights, on a dataset with a different number of object
        categories. The number of kernels in the convolutional layers just before each detection layer depends on the
        number of output classes. The Darknet solution is to truncate the weight file and stop reading weights at the
        first incompatible layer. For this reason the function silently leaves the rest of the layers unchanged, when
        the weight file ends.

        Args:
            weight_file: A file-like object containing model weights in the Darknet binary format.

        """
        if not isinstance(weight_file, io.IOBase):
            raise ValueError("weight_file must be a file-like object.")

        version = np.fromfile(weight_file, count=3, dtype=np.int32)
        images_seen = np.fromfile(weight_file, count=1, dtype=np.int64)
        rank_zero_info(
            f"Loading weights from Darknet model version {version[0]}.{version[1]}.{version[2]} "
            f"that has been trained on {images_seen[0]} images."
        )

        def read(tensor: Tensor) -> int:
            """Reads the contents of ``tensor`` from the current position of ``weight_file``.

            Returns the number of elements read. If there's no more data in ``weight_file``, returns 0.
            """
            np_array = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            num_elements = np_array.size
            if num_elements > 0:
                source = torch.from_numpy(np_array).view_as(tensor)
                with torch.no_grad():
                    tensor.copy_(source)
            return num_elements

        for layer in self.layers:
            # Weights are loaded only to convolutional layers
            if not isinstance(layer, Conv):
                continue

            # If convolution is followed by batch normalization, read the batch normalization parameters. Otherwise we
            # read the convolution bias.
            if isinstance(layer.norm, nn.Identity):
                assert layer.conv.bias is not None
                read(layer.conv.bias)
            else:
                assert isinstance(layer.norm, nn.BatchNorm2d)
                assert layer.norm.running_mean is not None
                assert layer.norm.running_var is not None
                read(layer.norm.bias)
                read(layer.norm.weight)
                read(layer.norm.running_mean)
                read(layer.norm.running_var)

            read_count = read(layer.conv.weight)
            if read_count == 0:
                return

    def _read_config(self, config_file: Iterable[str]) -> List[Dict[str, Any]]:
        """Reads a Darnet network configuration file and returns a list of configuration sections.

        Args:
            config_file: The configuration file to read.

        Returns:
            A list of configuration sections.

        """
        section_re = re.compile(r"\[([^]]+)\]")
        list_variables = ("layers", "anchors", "mask", "scales")
        variable_types = {
            "activation": str,
            "anchors": int,
            "angle": float,
            "batch": int,
            "batch_normalize": bool,
            "beta_nms": float,
            "burn_in": int,
            "channels": int,
            "classes": int,
            "cls_normalizer": float,
            "decay": float,
            "exposure": float,
            "filters": int,
            "from": int,
            "groups": int,
            "group_id": int,
            "height": int,
            "hue": float,
            "ignore_thresh": float,
            "iou_loss": str,
            "iou_normalizer": float,
            "iou_thresh": float,
            "jitter": float,
            "layers": int,
            "learning_rate": float,
            "mask": int,
            "max_batches": int,
            "max_delta": float,
            "momentum": float,
            "mosaic": bool,
            "new_coords": int,
            "nms_kind": str,
            "num": int,
            "obj_normalizer": float,
            "pad": bool,
            "policy": str,
            "random": bool,
            "resize": float,
            "saturation": float,
            "scales": float,
            "scale_x_y": float,
            "size": int,
            "steps": str,
            "stride": int,
            "subdivisions": int,
            "truth_thresh": float,
            "width": int,
        }

        section = None
        sections = []

        def convert(key: str, value: str) -> Union[str, int, float, List[Union[str, int, float]]]:
            """Converts a value to the correct type based on key."""
            if key not in variable_types:
                warn(f"Unknown YOLO configuration variable: {key}")
                return value
            if key in list_variables:
                return [variable_types[key](v) for v in value.split(",")]
            return variable_types[key](value)

        for line in config_file:
            line = line.strip()
            if (not line) or (line[0] == "#"):
                continue

            section_match = section_re.match(line)
            if section_match:
                if section is not None:
                    sections.append(section)
                section = {"type": section_match.group(1)}
            else:
                if section is None:
                    raise RuntimeError("Darknet network configuration file does not start with a section header.")
                key, value = line.split("=")
                key = key.rstrip()
                value = value.lstrip()
                section[key] = convert(key, value)
        if section is not None:
            sections.append(section)

        return sections


def _create_layer(config: CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Calls one of the ``_create_<layertype>(config, num_inputs)`` functions to create a PyTorch module from the
    layer config.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    create_func: Dict[str, Callable[..., CREATE_LAYER_OUTPUT]] = {
        "convolutional": _create_convolutional,
        "maxpool": _create_maxpool,
        "route": _create_route,
        "shortcut": _create_shortcut,
        "upsample": _create_upsample,
        "yolo": _create_yolo,
    }
    return create_func[config["type"]](config, num_inputs, **kwargs)


def _create_convolutional(config: CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a convolutional layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    batch_normalize = config.get("batch_normalize", False)
    padding = (config["size"] - 1) // 2 if config["pad"] else 0

    layer = Conv(
        num_inputs[-1],
        config["filters"],
        kernel_size=config["size"],
        stride=config["stride"],
        padding=padding,
        bias=not batch_normalize,
        activation=config["activation"],
        norm="batchnorm" if batch_normalize else None,
    )
    return layer, config["filters"]


def _create_maxpool(config: CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a max pooling layer.

    Padding is added so that the output resolution will be the input resolution divided by stride, rounded upwards.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = MaxPool(config["size"], config["stride"])
    return layer, num_inputs[-1]


def _create_route(config: CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a routing layer.

    A routing layer concatenates the output (or part of it) from the layers specified by the "layers" configuration
    option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    num_chunks = config.get("groups", 1)
    chunk_idx = config.get("group_id", 0)

    # 0 is the first layer, -1 is the previous layer
    last = len(num_inputs) - 1
    source_layers = [layer if layer >= 0 else last + layer for layer in config["layers"]]

    layer = RouteLayer(source_layers, num_chunks, chunk_idx)

    # The number of outputs of a source layer is the number of inputs of the next layer.
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)

    return layer, num_outputs


def _create_shortcut(config: CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a shortcut layer.

    A shortcut layer adds a residual connection from the layer specified by the "from" configuration option.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = ShortcutLayer(config["from"])
    return layer, num_inputs[-1]


def _create_upsample(config: CONFIG, num_inputs: List[int], **kwargs: Any) -> CREATE_LAYER_OUTPUT:
    """Creates a layer that upsamples the data.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.

    """
    layer = nn.Upsample(scale_factor=config["stride"], mode="nearest")
    return layer, num_inputs[-1]


def _create_yolo(
    config: CONFIG,
    num_inputs: List[int],
    prior_shapes: Optional[List[Tuple[int, int]]] = None,
    matching_algorithm: Optional[str] = None,
    matching_threshold: Optional[float] = None,
    spatial_range: float = 5.0,
    size_range: float = 4.0,
    ignore_bg_threshold: Optional[float] = None,
    overlap_func: Optional[Union[str, Callable]] = None,
    predict_overlap: Optional[float] = None,
    label_smoothing: Optional[float] = None,
    overlap_loss_multiplier: Optional[float] = None,
    confidence_loss_multiplier: Optional[float] = None,
    class_loss_multiplier: Optional[float] = None,
    **kwargs: Any,
) -> CREATE_LAYER_OUTPUT:
    """Creates a YOLO detection layer.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer. Not used by the detection layer.
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
        ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the corresponding anchor
            has IoU with some target greater than this threshold, the predictor will not be taken into account when
            calculating the confidence loss.
        overlap_func: A function for calculating the pairwise overlaps between two sets of boxes. Either a string or a
            function that returns a matrix of pairwise overlaps. Valid string values are "iou", "giou", "diou", and
            "ciou".
        predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that the target
            confidence is 1 if there's an object, and 1.0 means that the target confidence is the output of
            ``overlap_func``.
        label_smoothing: The epsilon parameter (weight) for class label smoothing. 0.0 means no smoothing (binary
            targets), and 1.0 means that the target probabilities are always 0.5.
        overlap_loss_multiplier: Overlap loss will be scaled by this value.
        confidence_loss_multiplier: Confidence loss will be scaled by this value.
        class_loss_multiplier: Classification loss will be scaled by this value.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output (always 0 for a detection layer).

    """
    if prior_shapes is None:
        # The "anchors" list alternates width and height.
        dims = config["anchors"]
        prior_shapes = [(dims[i], dims[i + 1]) for i in range(0, len(dims), 2)]
    if ignore_bg_threshold is None:
        ignore_bg_threshold = config.get("ignore_thresh", 1.0)
        assert isinstance(ignore_bg_threshold, float)
    if overlap_func is None:
        overlap_func = config.get("iou_loss", "iou")
        assert isinstance(overlap_func, str)
    if overlap_loss_multiplier is None:
        overlap_loss_multiplier = config.get("iou_normalizer", 1.0)
        assert isinstance(overlap_loss_multiplier, float)
    if confidence_loss_multiplier is None:
        confidence_loss_multiplier = config.get("obj_normalizer", 1.0)
        assert isinstance(confidence_loss_multiplier, float)
    if class_loss_multiplier is None:
        class_loss_multiplier = config.get("cls_normalizer", 1.0)
        assert isinstance(class_loss_multiplier, float)

    layer = create_detection_layer(
        num_classes=config["classes"],
        prior_shapes=prior_shapes,
        prior_shape_idxs=config["mask"],
        matching_algorithm=matching_algorithm,
        matching_threshold=matching_threshold,
        spatial_range=spatial_range,
        size_range=size_range,
        ignore_bg_threshold=ignore_bg_threshold,
        overlap_func=overlap_func,
        predict_overlap=predict_overlap,
        label_smoothing=label_smoothing,
        overlap_loss_multiplier=overlap_loss_multiplier,
        confidence_loss_multiplier=confidence_loss_multiplier,
        class_loss_multiplier=class_loss_multiplier,
        xy_scale=config.get("scale_x_y", 1.0),
        input_is_normalized=config.get("new_coords", 0) > 0,
    )
    return layer, 0
