from typing import Callable

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


def imagenet_normalization() -> Callable:
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
        )

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return normalize


def cifar10_normalization() -> Callable:
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
        )

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    return normalize


def stl10_normalization() -> Callable:
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
        )

    normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
    return normalize


def emnist_normalization(split: str) -> Callable:
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            "You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`."
        )

    # `stats` contains mean and std for each `split`.
    stats = {
        "balanced": (0.175, 0.333),
        "byclass": (0.174, 0.332),
        "bymerge": (0.174, 0.332),
        "digits": (0.173, 0.332),
        "letters": (0.172, 0.331),
        "mnist": (0.173, 0.332),
    }

    return transforms.Normalize(mean=stats[split][0], std=stats[split][1])
