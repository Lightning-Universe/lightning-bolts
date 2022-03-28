import io
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.utilities.distributed import rank_zero_debug, rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor

from pl_bolts.models.detection.yolo import layers
from pl_bolts.models.detection.yolo.layers import MaxPool
from pl_bolts.models.detection.yolo.utils import get_image_size


class DarknetNetwork(nn.Module):
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation."""

    def __init__(self, config_path: str, weights_path: Optional[str] = None, **kwargs) -> None:
        """Parses a Darknet configuration file and creates the network structure.

        Iterates through the layers from the configuration and creates corresponding PyTorch modules. If
        ``weights_path`` is given and points to a Darknet model file, loads the convolutional layer weights from the
        file.

        Args:
            config_path: Path to a Darknet configuration file that defines the network architecture.
            weights_path: Path to a Darknet model file. If given, the model weights will be read from this file.
            match_sim_ota: If ``True``, matches a target to an anchor using the SimOTA algorithm from YOLOX.
            match_size_ratio: If specified, matches a target to an anchor if its width and height relative to the anchor
                is smaller than this ratio. If ``match_size_ratio`` or ``match_iou_threshold`` is not specified, selects
                for each target the anchor with the highest IoU.
            match_iou_threshold: If specified, matches a target to an anchor if the IoU is higher than this threshold.
            ignore_bg_threshold: If a predictor is not responsible for predicting any target, but the prior shape has
                IoU with some target greater than this threshold, the predictor will not be taken into account when
                calculating the confidence loss.
            overlap_func: Which function to use for calculating the overlap between boxes. Valid values are "iou",
                "giou", "diou", and "ciou".
            predict_overlap: Balance between binary confidence targets and predicting the overlap. 0.0 means that target
                confidence is one if there's an object, and 1.0 means that the target confidence is the output of
                ``overlap_func``.
            overlap_loss_multiplier: Overlap loss will be scaled by this value.
            confidence_loss_multiplier: Confidence loss will be scaled by this value.
            class_loss_multiplier: Classification loss will be scaled by this value.
        """
        super().__init__()

        with open(config_path) as config_file:
            sections = self._read_config(config_file)

        if len(sections) < 2:
            raise MisconfigurationException("The model configuration file should include at least two sections.")

        self.__dict__.update(sections[0])
        global_config = sections[0]
        layer_configs = sections[1:]

        self.layers = nn.ModuleList()
        # num_inputs will contain the number of channels in the input of every layer up to the current layer. It is
        # initialized with the number of channels in the input image.
        num_inputs = [global_config.get("channels", 3)]
        for layer_config in layer_configs:
            config = {**global_config, **layer_config}
            module, num_outputs = _create_layer(config, num_inputs, **kwargs)
            self.layers.append(module)
            num_inputs.append(num_outputs)

        if weights_path is not None:
            with open(weights_path) as weight_file:
                self.load_weights(weight_file)

    def forward(self, x: Tensor, targets: Optional[List[Dict[str, Tensor]]] = None) -> Tuple[Tensor, Tensor]:
        outputs = []  # Outputs from all layers
        detections = []  # Outputs from detection layers
        losses = []  # Losses from detection layers
        hits = []  # Number of targets each detection layer was responsible for

        image_size = get_image_size(x)

        for layer in self.layers:
            if isinstance(layer, (layers.RouteLayer, layers.ShortcutLayer)):
                x = layer(x, outputs)
            elif isinstance(layer, layers.DetectionLayer):
                x = layer(x, image_size, targets)
                detections.append(x)
                if targets is not None:
                    losses.append(layer.losses)
                    hits.append(layer.hits)
            else:
                x = layer(x)

            outputs.append(x)

        return detections, losses, hits

    def load_weights(self, weight_file: io.IOBase):
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

        def read(tensor):
            """Reads the contents of ``tensor`` from the current position of ``weight_file``.

            If there's no more data in ``weight_file``, returns without error.
            """
            x = np.fromfile(weight_file, count=tensor.numel(), dtype=np.float32)
            if x.size > 0:
                x = torch.from_numpy(x).view_as(tensor)
                with torch.no_grad():
                    tensor.copy_(x)
            return x.size

        for layer_idx, layer in enumerate(self.layers):
            # Weights are loaded only to convolutional layers
            if not isinstance(layer, layers.Conv):
                continue

            rank_zero_debug(f"Reading weights for layer {layer_idx}: {list(layer.conv.weight.shape)}")

            # If convolution is followed by batch normalization, read the batch normalization parameters. Otherwise we
            # read the convolution bias.
            if isinstance(layer.norm, nn.Identity):
                read(layer.conv.bias)
            else:
                read(layer.norm.bias)
                read(layer.norm.weight)
                read(layer.norm.running_mean)
                read(layer.norm.running_var)

            read_count = read(layer.conv.weight)
            if read_count == 0:
                return

    def _read_config(self, config_file: Iterable[str]) -> List[Dict[str, Any]]:
        """Reads a YOLOv4 network configuration file and returns a list of configuration sections.

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

        def convert(key, value):
            """Converts a value to the correct type based on key."""
            if key not in variable_types:
                warn("Unknown YOLO configuration variable: " + key)
                return key, value
            if key in list_variables:
                value = [variable_types[key](v) for v in value.split(",")]
            else:
                value = variable_types[key](value)
            return key, value

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
                key, value = line.split("=")
                key = key.rstrip()
                value = value.lstrip()
                key, value = convert(key, value)
                section[key] = value
        if section is not None:
            sections.append(section)

        return sections


def _create_layer(config: Dict[str, Any], num_inputs: List[int], **kwargs) -> Tuple[nn.Module, int]:
    """Calls one of the ``_create_<layertype>(config, num_inputs)`` functions to create a PyTorch module from the
    layer config.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the number of channels in
        its output.
    """
    create_func = {
        "convolutional": _create_convolutional,
        "maxpool": _create_maxpool,
        "route": _create_route,
        "shortcut": _create_shortcut,
        "upsample": _create_upsample,
        "yolo": _create_yolo,
    }
    return create_func[config["type"]](config, num_inputs, **kwargs)


def _create_convolutional(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    batch_normalize = config.get("batch_normalize", False)
    padding = (config["size"] - 1) // 2 if config["pad"] else 0

    layer = layers.Conv(
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


def _create_maxpool(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    """Creates a max pooling layer.

    Padding is added so that the output resolution will be the input resolution divided by stride, rounded upwards.
    """
    layer = MaxPool(config["size"], config["stride"])
    return layer, num_inputs[-1]


def _create_route(config, num_inputs: List[int], **kwargs):
    num_chunks = config.get("groups", 1)
    chunk_idx = config.get("group_id", 0)

    # 0 is the first layer, -1 is the previous layer
    last = len(num_inputs) - 1
    source_layers = [layer if layer >= 0 else last + layer for layer in config["layers"]]

    layer = layers.RouteLayer(source_layers, num_chunks, chunk_idx)

    # The number of outputs of a source layer is the number of inputs of the next layer.
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)

    return layer, num_outputs


def _create_shortcut(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    layer = layers.ShortcutLayer(config["from"])
    return layer, num_inputs[-1]


def _create_upsample(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    layer = nn.Upsample(scale_factor=config["stride"], mode="nearest")
    return layer, num_inputs[-1]


def _create_yolo(
    config: Dict[str, Any],
    num_inputs: List[int],
    prior_shapes: Optional[List[Tuple[int, int]]] = None,
    matching_algorithm: Optional[str] = None,
    matching_threshold: Optional[float] = None,
    ignore_bg_threshold: Optional[float] = None,
    overlap_func: Optional[Union[str, Callable]] = None,
    predict_overlap: Optional[float] = None,
    overlap_loss_multiplier: Optional[float] = None,
    confidence_loss_multiplier: Optional[float] = None,
    class_loss_multiplier: Optional[float] = None,
    **kwargs,
):
    if prior_shapes is None:
        # The "anchors" list alternates width and height.
        prior_shapes = config["anchors"]
        prior_shapes = [(prior_shapes[i], prior_shapes[i + 1]) for i in range(0, len(prior_shapes), 2)]
    if ignore_bg_threshold is None:
        ignore_bg_threshold = config.get("ignore_thresh", 1.0)
    if overlap_func is None:
        overlap_func = config.get("iou_loss", "iou")
    if overlap_loss_multiplier is None:
        overlap_loss_multiplier = config.get("iou_normalizer", 1.0)
    if confidence_loss_multiplier is None:
        confidence_loss_multiplier = config.get("obj_normalizer", 1.0)
    if class_loss_multiplier is None:
        class_loss_multiplier = config.get("cls_normalizer", 1.0)

    layer = layers.create_detection_layer(
        num_classes=config["classes"],
        prior_shapes=prior_shapes,
        prior_shape_idxs=config["mask"],
        matching_algorithm=matching_algorithm,
        matching_threshold=matching_threshold,
        ignore_bg_threshold=ignore_bg_threshold,
        overlap_func=overlap_func,
        predict_overlap=predict_overlap,
        overlap_loss_multiplier=overlap_loss_multiplier,
        confidence_loss_multiplier=confidence_loss_multiplier,
        class_loss_multiplier=class_loss_multiplier,
        xy_scale=config.get("scale_x_y", 1.0),
        input_is_normalized=config.get("new_coords", 0) > 0,
    )
    return layer, None
