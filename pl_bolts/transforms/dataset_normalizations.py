from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


def imagenet_normalization():
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            'You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`.'
        )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


def cifar10_normalization():
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            'You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`.'
        )

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    return normalize


def stl10_normalization():
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            'You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`.'
        )

    normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
    return normalize
