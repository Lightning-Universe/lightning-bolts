import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from warnings import warn

import torch.nn as nn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pl_bolts.models.detection.yolo import yolo_layers
from pl_bolts.models.detection.yolo.target_matching import (
    HighestIoUMatching,
    IoUThresholdMatching,
    SimOTAMatching,
    SizeRatioMatching,
)
from pl_bolts.models.detection.yolo.yolo_loss import LossFunction


class DarknetConfiguration:
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation.

    The :func:`~pl_bolts.models.detection.yolo.yolo_config.YOLOConfiguration.get_network` method returns a PyTorch
    module list that can be used to construct a YOLO model.
    """

    def __init__(self, path: str) -> None:
        """Saves the variables from the first configuration section to attributes of this object, and the rest of
        the sections to the ``layer_configs`` list.

        Args:
            path: Path to a configuration file
        """
        with open(path) as config_file:
            sections = self._read_file(config_file)

        if len(sections) < 2:
            raise MisconfigurationException("The model configuration file should include at least two sections.")

        self.__dict__.update(sections[0])
        self.global_config = sections[0]
        self.layer_configs = sections[1:]

    def get_network(self, **kwargs) -> nn.ModuleList:
        """Iterates through the layers from the configuration and creates corresponding PyTorch modules. Returns
        the network structure that can be used to create a YOLO model.

        Returns:
            A :class:`~torch.nn.ModuleList` that defines the YOLO network.
        """
        result = nn.ModuleList()
        num_inputs = [3]  # Number of channels in the input of every layer up to the current layer
        for layer_config in self.layer_configs:
            config = {**self.global_config, **layer_config}
            module, num_outputs = _create_layer(config, num_inputs, **kwargs)
            result.append(module)
            num_inputs.append(num_outputs)
        return result

    def _read_file(self, config_file: Iterable[str]) -> List[Dict[str, Any]]:
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
    layer = nn.Sequential()

    batch_normalize = config.get("batch_normalize", False)
    padding = (config["size"] - 1) // 2 if config["pad"] else 0

    conv = nn.Conv2d(
        num_inputs[-1], config["filters"], config["size"], config["stride"], padding, bias=not batch_normalize
    )
    layer.add_module("conv", conv)

    if batch_normalize:
        bn = nn.BatchNorm2d(config["filters"])  # YOLOv5: eps=0.001, momentum=0.03
        layer.add_module("bn", bn)

    activation_name = config["activation"]
    if activation_name == "leaky":
        leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        layer.add_module("leakyrelu", leakyrelu)
    elif activation_name == "mish":
        mish = yolo_layers.Mish()
        layer.add_module("mish", mish)
    elif activation_name == "swish":
        swish = nn.SiLU(inplace=True)
        layer.add_module("swish", swish)
    elif activation_name == "logistic":
        logistic = nn.Sigmoid()
        layer.add_module("logistic", logistic)
    elif activation_name == "linear":
        pass
    else:
        raise MisconfigurationException("Unknown activation: " + activation_name)

    return layer, config["filters"]


def _create_maxpool(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    """Creates a max pooling layer.

    Padding is added so that the output resolution will be the input resolution divided by stride, rounded upwards.
    """
    kernel_size = config["size"]
    padding = (kernel_size - 1) // 2
    maxpool = nn.MaxPool2d(kernel_size, config["stride"], padding)
    if kernel_size % 2 == 1:
        return maxpool, num_inputs[-1]

    # If the kernel size is an even number, we need one cell of extra padding, on top of the padding added by MaxPool2d
    # on both sides.
    layer = nn.Sequential()
    layer.add_module("pad", nn.ZeroPad2d((0, 1, 0, 1)))
    layer.add_module("maxpool", maxpool)
    return layer, num_inputs[-1]


def _create_route(config, num_inputs: List[int], **kwargs):
    num_chunks = config.get("groups", 1)
    chunk_idx = config.get("group_id", 0)

    # 0 is the first layer, -1 is the previous layer
    last = len(num_inputs) - 1
    source_layers = [layer if layer >= 0 else last + layer for layer in config["layers"]]

    layer = yolo_layers.RouteLayer(source_layers, num_chunks, chunk_idx)

    # The number of outputs of a source layer is the number of inputs of the next layer.
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)

    return layer, num_outputs


def _create_shortcut(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    layer = yolo_layers.ShortcutLayer(config["from"])
    return layer, num_inputs[-1]


def _create_upsample(config: Dict[str, Any], num_inputs: List[int], **kwargs):
    layer = nn.Upsample(scale_factor=config["stride"], mode="nearest")
    return layer, num_inputs[-1]


def _create_yolo(
    config: Dict[str, Any],
    num_inputs: List[int],
    match_sim_ota: bool = False,
    match_size_ratio: Optional[float] = None,
    match_iou_threshold: Optional[float] = None,
    ignore_iou_threshold: Optional[float] = None,
    overlap_loss: Optional[Union[str, Callable]] = None,
    predict_overlap: Optional[float] = None,
    overlap_loss_multiplier: Optional[float] = None,
    class_loss_multiplier: Optional[float] = None,
    confidence_loss_multiplier: Optional[float] = None,
    **kwargs,
):
    # The "anchors" list alternates width and height.
    anchor_dims = config["anchors"]
    anchor_dims = [(anchor_dims[i], anchor_dims[i + 1]) for i in range(0, len(anchor_dims), 2)]
    anchor_ids = config["mask"]

    xy_scale = config.get("scale_x_y", 1.0)
    input_is_normalized = config.get("new_coords", 0) > 0
    ignore_iou_threshold = config.get("ignore_thresh", 1.0) if ignore_iou_threshold is None else ignore_iou_threshold

    overlap_loss = overlap_loss or config.get("iou_loss", "iou")
    if overlap_loss_multiplier is None:
        overlap_loss_multiplier = config.get("iou_normalizer", 1.0)
    if class_loss_multiplier is None:
        class_loss_multiplier = config.get("cls_normalizer", 1.0)
    if confidence_loss_multiplier is None:
        confidence_loss_multiplier = config.get("obj_normalizer", 1.0)
    loss_func = LossFunction(
        overlap_loss, predict_overlap, overlap_loss_multiplier, class_loss_multiplier, confidence_loss_multiplier
    )

    if sum(var is not None for var in (match_sim_ota, match_size_ratio, match_iou_threshold)) > 1:
        raise ValueError("More than one matching algorithm specified.")
    if match_sim_ota:
        sim_ota_loss_func = LossFunction(
            overlap_loss, None, overlap_loss_multiplier, class_loss_multiplier, confidence_loss_multiplier
        )
        matching_func = SimOTAMatching(sim_ota_loss_func)
    elif match_size_ratio is not None:
        matching_func = SizeRatioMatching(match_size_ratio, anchor_dims, anchor_ids, ignore_iou_threshold)
    elif match_iou_threshold is not None:
        matching_func = IoUThresholdMatching(match_iou_threshold, anchor_dims, anchor_ids, ignore_iou_threshold)
    else:
        matching_func = HighestIoUMatching(anchor_dims, anchor_ids, ignore_iou_threshold)

    layer = yolo_layers.DetectionLayer(
        num_classes=config["classes"],
        anchor_dims=[anchor_dims[i] for i in anchor_ids],
        matching_func=matching_func,
        loss_func=loss_func,
        xy_scale=xy_scale,
        input_is_normalized=input_is_normalized,
    )

    return layer, num_inputs[-1]
