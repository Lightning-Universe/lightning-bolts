from typing import Any, Optional

import torch.nn as nn

from pl_bolts.models.detection.components import create_torchvision_backbone
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


@under_review()
def create_retinanet_backbone(
    backbone: str, fpn: bool = True, pretrained: Optional[str] = None, trainable_backbone_layers: int = 3, **kwargs: Any
) -> nn.Module:
    """
    Args:
        backbone:
            Supported backones are: "resnet18", "resnet34","resnet50", "resnet101", "resnet152",
            "resnext50_32x4d", "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2",
            as resnets with fpn backbones.
            Without fpn backbones supported are: "resnet18", "resnet34", "resnet50","resnet101",
            "resnet152", "resnext101_32x8d", "mobilenet_v2", "vgg11", "vgg13", "vgg16", "vgg19",
        fpn: If True then constructs fpn as well.
        pretrained: If None creates imagenet weights backbone.
        trainable_backbone_layers: number of trainable resnet layers starting from final block.
    """

    if fpn:
        # Creates a torchvision resnet model with fpn added.
        backbone = resnet_fpn_backbone(backbone, pretrained=True, trainable_layers=trainable_backbone_layers, **kwargs)
    else:
        # This does not create fpn backbone, it is supported for all models
        backbone, _ = create_torchvision_backbone(backbone, pretrained)
    return backbone
