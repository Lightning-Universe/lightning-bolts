
import torch.nn as nn
from typing import Optional
from pl_bolts.utils.warnings import warn_missing_pkg

try:
    import torchvision

    from pl_bolts.models.detection.components._supported_models import TORCHVISION_MODEL_ZOO
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover


def _create_backbone_generic(model: nn.Module, out_channels: int):
    """
    Generic Backbone creater. It removes the last linear layer.

    Args:
        model: torch.nn model
        out_channels: Number of out_channels in last layer.
    """
    modules_total = list(model.children())
    modules = modules_total[:-1]
    ft_backbone = nn.Sequential(*modules)
    ft_backbone.out_channels = out_channels
    return ft_backbone


# Use this when you have Adaptive Pooling layer in End.
# When Model.features is not applicable.
def _create_backbone_adaptive(model, out_channels: Optional[int] = None):
    """
    Creates backbone by removing linear after Adaptive Pooling layer.

    Args:
        model: torch.nn model with adaptive pooling layer
        out_channels: Number of out_channels in last layer
    """
    if out_channels is None:
        modules_total = list(model.children())
        out_channels = modules_total[-1].in_features
    return _create_backbone_generic(model, out_channels=out_channels)


def _create_backbone_features(model, out_channels: int):
    """
    Creates backbone from feature sequential block.

    Args:
        model: torch.nn model with features as sequential block.
        out_channels: Number of out_channels in last layer.
    """
    ft_backbone = model.features
    ft_backbone.out_channels = out_channels
    return ft_backbone


def create_torchvision_backbone(model_name: str, pretrained: bool = True):
    """
    Creates CNN backbone from Torchvision.

    Args:
        model_name: Name of the model. E.g. resnet18
        pretrained: Pretrained weights dataset "imagenet", etc
    """

    if model_name == "mobilenet_v2":
        model_selected = TORCHVISION_MODEL_ZOO[model_name]
        net = model_selected(pretrained=pretrained)
        out_channels = 1280
        ft_backbone = _create_backbone_features(net, 1280)
        return ft_backbone, out_channels

    elif model_name in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        out_channels = 512
        model_selected = TORCHVISION_MODEL_ZOO[model_name]
        net = model_selected(pretrained=pretrained)
        ft_backbone = _create_backbone_features(net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet18", "resnet34"]:
        out_channels = 512
        model_selected = TORCHVISION_MODEL_ZOO[model_name]
        net = model_selected(pretrained=pretrained)
        ft_backbone = _create_backbone_adaptive(net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d"]:
        out_channels = 2048
        model_selected = TORCHVISION_MODEL_ZOO[model_name]
        net = model_selected(pretrained=pretrained)
        ft_backbone = _create_backbone_adaptive(net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        out_channels = 1280
        model_selected = TORCHVISION_MODEL_ZOO[model_name]
        net = model_selected(pretrained=pretrained)
        ft_backbone = _create_backbone_adaptive(net, out_channels)
        return ft_backbone, out_channels

    else:
        raise ValueError(f"Unsupported model: '{model_name}'")
