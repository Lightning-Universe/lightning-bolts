# Model should have AdaptiveAvgPool2d layer and then Linear Layer
# We will remove the Linear Layer and create a new Linear Layer with num_classes
from warnings import warn
try:
    from torchvision import models
except ModuleNotFoundError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')

import torch.nn as nn

__all__ = ["create_backbone_generic", "create_backbone_adaptive",
           "create_backbone_features", "create_torchvision_backbone"]


def create_backbone_generic(model, out_channels: int):
    # Here out_channels is a mandatory argument.
    modules_total = list(model.children())
    modules = modules_total[:-1]
    ft_backbone = nn.Sequential(*modules)
    ft_backbone.out_channels = out_channels
    return ft_backbone


# Use this when you have Adaptive Pooling layer in End.
# When Model.features is not applicable.
def create_backbone_adaptive(model, out_channels: int = None):
    # Out channels is optional can pass it if user knows it
    # print(list(model.children())[-1].in_features)
    if out_channels is None:
        # Tries to infer the out_channels from layer before adaptive avg pool.
        modules_total = list(model.children())
        out_channels = modules_total[-1].in_features
    return create_backbone_generic(model, out_channels=out_channels)


# Use this when model.features function is available as in mobilenet and vgg
# Again it is mandatory for out_channels here
def create_backbone_features(model, out_channels: int):
    ft_backbone = model.features
    ft_backbone.out_channels = out_channels
    return ft_backbone


def create_torchvision_backbone(model_name: str, num_classes: int, pretrained: bool = True):
    """
    Creates CNN backbone from Torchvision.
    Args:
        model_name (str) : Name of the model. E.g. resnet18
        num_classes (int) : Number of classes for classification.
        pretrained (bool) : If true uses modelwweights pretrained on ImageNet.
    """
    if model_name == "mobilenet":
        mobile_net = models.mobilenet_v2(pretrained=pretrained)
        out_channels = 1280
        ft_backbone = create_backbone_features(mobile_net, 1280)
        return ft_backbone, out_channels

    elif model_name in ["vgg11", "vgg13", "vgg16", "vgg19", ]:
        out_channels = 512
        if model_name == "vgg11":
            vgg_net = models.vgg11(pretrained=pretrained)
        elif model_name == "vgg13":
            vgg_net = models.vgg13(pretrained=pretrained)
        elif model_name == "vgg16":
            vgg_net = models.vgg16(pretrained=pretrained)
        elif model_name == "vgg19":
            vgg_net = models.vgg19(pretrained=pretrained)

        ft_backbone = create_backbone_features(vgg_net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet18", "resnet34"]:
        out_channels = 512
        if model_name == "resnet18":
            resnet_net = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            resnet_net = models.resnet34(pretrained=pretrained)

        ft_backbone = create_backbone_adaptive(resnet_net, out_channels)
        return ft_backbone, out_channels

    elif model_name in ["resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d", ]:
        out_channels = 2048
        if model_name == "resnet50":
            resnet_net = models.resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            resnet_net = models.resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            resnet_net = models.resnet152(pretrained=pretrained)
        elif model_name == "resnext50_32x4d":
            resnet_net = models.resnext50_32x4d(pretrained=pretrained)
        elif model_name == "resnext101_32x8d":
            resnet_net = models.resnext101_32x8d(pretrained=pretrained)

        ft_backbone = create_backbone_adaptive(resnet_net, 2048)
        return ft_backbone, out_channels

    elif model_name in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        out_channels = 1280
        if model_name == "mnasnet0_5":
            mnasnet_net = models.mnasnet0_5(pretrained=pretrained)
        elif model_name == "mnasnet0_75":
            mnasnet_net = models.mnasnet0_75(pretrained=pretrained)
        elif model_name == "mnasnet1_0":
            mnasnet_net = models.mnasnet1_0(pretrained=pretrained)
        elif model_name == "mnasnet1_3":
            mnasnet_net = models.mnasnet1_3(pretrained=pretrained)

        ft_backbone = create_backbone_adaptive(mnasnet_net, 1280)
        return ft_backbone, out_channels

    else:
        raise ValueError("No such model implemented in torchvision")
