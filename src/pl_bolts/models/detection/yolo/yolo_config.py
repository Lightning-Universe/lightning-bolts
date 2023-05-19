import re
from typing import Any, Dict, Iterable, List, Tuple
from warnings import warn

import torch.nn as nn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from pl_bolts.models.detection.yolo import yolo_layers
from pl_bolts.utils.stability import under_review


@under_review()
class YOLOConfiguration:
    """This class can be used to parse the configuration files of the Darknet YOLOv4 implementation.

    The :func:`~pl_bolts.models.detection.yolo.yolo_config.YOLOConfiguration.get_network` method
    returns a PyTorch module list that can be used to construct a YOLO model.
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

    def get_network(self) -> nn.ModuleList:
        """Iterates through the layers from the configuration and creates corresponding PyTorch modules. Returns
        the network structure that can be used to create a YOLO model.

        Returns:
            A :class:`~torch.nn.ModuleList` that defines the YOLO network.
        """
        result = nn.ModuleList()
        # Number of channels in the input of every layer up to the current layer,
        # use channels from configuration or default of 3
        num_inputs = [self.global_config.get("channels", 3)]
        for layer_config in self.layer_configs:
            config = {**self.global_config, **layer_config}
            module, num_outputs = _create_layer(config, num_inputs)
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


@under_review()
def _create_layer(config: dict, num_inputs: List[int]) -> Tuple[nn.Module, int]:
    """Calls one of the ``_create_<layertype>(config, num_inputs)`` functions to create a PyTorch module from the
    layer config.

    Args:
        config: Dictionary of configuration options for this layer.
        num_inputs: Number of channels in the input of every layer up to this layer.

    Returns:
        module (:class:`~torch.nn.Module`), num_outputs (int): The created PyTorch module and the
        number of channels in its output.
    """
    create_func = {
        "convolutional": _create_convolutional,
        "maxpool": _create_maxpool,
        "route": _create_route,
        "shortcut": _create_shortcut,
        "upsample": _create_upsample,
        "yolo": _create_yolo,
    }
    return create_func[config["type"]](config, num_inputs)


@under_review()
def _create_convolutional(config, num_inputs):
    module = nn.Sequential()

    batch_normalize = config.get("batch_normalize", False)
    padding = (config["size"] - 1) // 2 if config["pad"] else 0

    conv = nn.Conv2d(
        num_inputs[-1], config["filters"], config["size"], config["stride"], padding, bias=not batch_normalize
    )
    module.add_module("conv", conv)

    if batch_normalize:
        bn = nn.BatchNorm2d(config["filters"])
        module.add_module("bn", bn)

    activation_name = config["activation"]
    if activation_name == "leaky":
        leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        module.add_module("leakyrelu", leakyrelu)
    elif activation_name == "mish":
        mish = yolo_layers.Mish()
        module.add_module("mish", mish)
    elif activation_name == "swish":
        swish = nn.SiLU(inplace=True)
        module.add_module("swish", swish)
    elif activation_name == "logistic":
        logistic = nn.Sigmoid()
        module.add_module("logistic", logistic)
    elif activation_name == "linear":
        pass
    else:
        raise ValueError("Unknown activation: " + activation_name)

    return module, config["filters"]


@under_review()
def _create_maxpool(config, num_inputs):
    padding = (config["size"] - 1) // 2
    module = nn.MaxPool2d(config["size"], config["stride"], padding)
    return module, num_inputs[-1]


@under_review()
def _create_route(config, num_inputs):
    num_chunks = config.get("groups", 1)
    chunk_idx = config.get("group_id", 0)

    # 0 is the first layer, -1 is the previous layer
    last = len(num_inputs) - 1
    source_layers = [layer if layer >= 0 else last + layer for layer in config["layers"]]

    module = yolo_layers.RouteLayer(source_layers, num_chunks, chunk_idx)

    # The number of outputs of a source layer is the number of inputs of the next layer.
    num_outputs = sum(num_inputs[layer + 1] // num_chunks for layer in source_layers)

    return module, num_outputs


@under_review()
def _create_shortcut(config, num_inputs):
    module = yolo_layers.ShortcutLayer(config["from"])
    return module, num_inputs[-1]


@under_review()
def _create_upsample(config, num_inputs):
    module = nn.Upsample(scale_factor=config["stride"], mode="nearest")
    return module, num_inputs[-1]


@under_review()
def _create_yolo(config, num_inputs):
    # The "anchors" list alternates width and height.
    anchor_dims = config["anchors"]
    anchor_dims = [(anchor_dims[i], anchor_dims[i + 1]) for i in range(0, len(anchor_dims), 2)]

    xy_scale = config.get("scale_x_y", 1.0)
    input_is_normalized = config.get("new_coords", 0) > 0
    ignore_threshold = config.get("ignore_thresh", 1.0)
    overlap_loss_multiplier = config.get("iou_normalizer", 1.0)
    class_loss_multiplier = config.get("cls_normalizer", 1.0)
    confidence_loss_multiplier = config.get("obj_normalizer", 1.0)

    overlap_loss_name = config.get("iou_loss", "mse")
    if overlap_loss_name == "mse":
        overlap_loss_func = yolo_layers.SELoss()
    elif overlap_loss_name == "giou":
        overlap_loss_func = yolo_layers.GIoULoss()
    else:
        overlap_loss_func = yolo_layers.IoULoss()

    module = yolo_layers.DetectionLayer(
        num_classes=config["classes"],
        anchor_dims=anchor_dims,
        anchor_ids=config["mask"],
        xy_scale=xy_scale,
        input_is_normalized=input_is_normalized,
        ignore_threshold=ignore_threshold,
        overlap_loss_func=overlap_loss_func,
        image_space_loss=overlap_loss_name != "mse",
        overlap_loss_multiplier=overlap_loss_multiplier,
        class_loss_multiplier=class_loss_multiplier,
        confidence_loss_multiplier=confidence_loss_multiplier,
    )

    return module, num_inputs[-1]
