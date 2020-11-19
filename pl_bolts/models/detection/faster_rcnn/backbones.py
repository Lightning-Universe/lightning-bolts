import torch.nn as nn
from pl_bolts.utils.warnings import warn_missing_pkg
from pl_bolts.models.detection.components import create_torchvision_backbone

try:
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
except ModuleNotFoundError:
    warn_missing_pkg('torchvision')  # pragma: no-cover


def create_fasterrcnn_backbone(backbone: str, fpn: bool = True, pretrained: str = None,
                               trainable_backbone_layers: int = 3, **kwargs) -> nn.Module:

    """
    Args:
        backbone (str):
            Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152",
            "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
            as resnets with fpn backbones.
            Without fpn backbones supported are: "resnet18", "resnet34", "resnet50","resnet101",
            "resnet152", "resnext101_32x8d", "mobilenet_v2", "vgg11", "vgg13", "vgg16", "vgg19",
        fpn (bool): If True then constructs fpn as well.
        pretrained (str): If None creates imagenet weights backbone.
    """

    if fpn:
        # Creates a torchvision resnet model with fpn added.
        print("Resnet FPN Backbones works only for imagenet weights")
        backbone = resnet_fpn_backbone(backbone, pretrained=True,
                                       trainable_layers=trainable_backbone_layers, **kwargs)
    else:
        # This does not create fpn backbone, it is supported for all models
        print("FPN is not supported for Non Resnet Backbones")
        backbone, _ = create_torchvision_backbone(backbone, pretrained)
    return backbone
