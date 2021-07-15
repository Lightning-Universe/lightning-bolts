from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


def _check_torchvision_avilable():
    if not _TORCHVISION_AVAILABLE:  # pragma: no cover
        raise ModuleNotFoundError(
            'You want to use `torchvision` which is not installed yet, install it with `pip install torchvision`.'
        )


def imagenet_normalization():
    _check_torchvision_avilable()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize


def cifar10_normalization():
    _check_torchvision_avilable()

    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    return normalize


def stl10_normalization():
    _check_torchvision_avilable()

    normalize = transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
    return normalize


def emnist_normalization(split: str):
    _check_torchvision_avilable()

    # `stats` contains mean and std for each `split`.
    stats = {
        'balanced': (0.17510417221708502, 0.3332070017067981),
        'byclass': (0.17359222670426913, 0.33162134741938604),
        'bymerge': (0.17359632632958918, 0.33161854660044826),
        'digits': (0.17325182375113168, 0.33163191505859535),
        'letters': (0.17222730561708793, 0.33091591285642147),
        'mnist': (0.17330445484320323, 0.33169403605816716),
    }

    return transforms.Normalize(mean=stats[split][0], std=stats[split][1])
