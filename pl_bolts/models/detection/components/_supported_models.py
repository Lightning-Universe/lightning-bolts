from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    import torchvision

    TORCHVISION_MODEL_ZOO = {
        "vgg11": torchvision.models.vgg11,
        "vgg13": torchvision.models.vgg13,
        "vgg16": torchvision.models.vgg16,
        "vgg19": torchvision.models.vgg19,
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50_32x4d": torchvision.models.resnext50_32x4d,
        "resnext50_32x8d": torchvision.models.resnext101_32x8d,
        "mnasnet0_5": torchvision.models.mnasnet0_5,
        "mnasnet0_75": torchvision.models.mnasnet0_75,
        "mnasnet1_0": torchvision.models.mnasnet1_0,
        "mnasnet1_3": torchvision.models.mnasnet1_3,
        "mobilenet_v2": torchvision.models.mobilenet_v2,
    }

else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    TORCHVISION_MODEL_ZOO = {}
