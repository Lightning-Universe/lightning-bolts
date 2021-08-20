import random

from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

if _PIL_AVAILABLE:
    from PIL import ImageFilter
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


class Moco2TrainCIFAR10Transforms:
    """Moco 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 32):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalCIFAR10Transforms:
    """Moco 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 32):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(height + 12),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class Moco2TrainSTL10Transforms:
    """Moco 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 64):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                stl10_normalization(),
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalSTL10Transforms:
    """Moco 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 64):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.test_augmentation = transforms.Compose(
            [
                transforms.Resize(height + 11),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
                stl10_normalization(),
            ]
        )

    def __call__(self, inp):
        q = self.test_augmentation(inp)
        k = self.test_augmentation(inp)
        return q, k


class Moco2TrainImagenetTransforms:
    """Moco 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 128):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

    def __call__(self, inp):
        q = self.train_transform(inp)
        k = self.train_transform(inp)
        return q, k


class Moco2EvalImagenetTransforms:
    """Moco 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, height: int = 128):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(height + 32),
                transforms.CenterCrop(height),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

    def __call__(self, inp):
        q = self.test_transform(inp)
        k = self.test_transform(inp)
        return q, k


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `Pillow` which is not installed yet, install it with `pip install Pillow`."
            )
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
