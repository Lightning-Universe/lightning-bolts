import random
from typing import Callable, List, Tuple, Union

from torch import Tensor

from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

if _PIL_AVAILABLE:
    from PIL import ImageFilter
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


@under_review()
class MoCoTrainTransforms:
    normalization: type
    """MoCo training transforms.

    Args:

    Example::

    """

    def __init__(self, size: int, normalize: Union[str, Callable]) -> None:
        if isinstance(normalize, str):
            self.normalize = normalize
        else:
            self.normalize = normalize

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.train_transform(x)
        k = self.train_transform(x)
        return q, k


@under_review()
class MoCo2TrainCIFAR10Transforms:
    """MoCo v2 transforms.

    Args:
        size (int, optional): input size. Defaults to 32.

    Transform::

        RandomResizedCrop(size=self.input_size)

    Example::

        from pl_bolts.transforms.self_supervised.MoCo_transforms import MoCo2TrainCIFAR10Transforms

        transform = MoCo2TrainCIFAR10Transforms(input_size=32)
        x = sample()
        (xi, xj) = transform(x)

    MoCo 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self, size: int = 32) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.train_transform(x)
        k = self.train_transform(x)
        return q, k


@under_review()
class MoCo2EvalCIFAR10Transforms:
    """MoCo 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, size: int = 32) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size + 12),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.test_transform(x)
        k = self.test_transform(x)
        return q, k


@under_review()
class MoCo2TrainImagenetTransforms:
    """MoCo 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, size: int = 224) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.train_transform(x)
        k = self.train_transform(x)
        return q, k


@under_review()
class MoCo2EvalImagenetTransforms:
    """Transforms for MoCo during training step.

    https://arxiv.org/pdf/2003.04297.pdf

    """

    def __init__(self, size: int = 128) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(size + 32),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                imagenet_normalization(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.test_transform(x)
        k = self.test_transform(x)
        return q, k


@under_review()
class MoCo2TrainSTL10Transforms:
    """MoCo 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, size: int = 64) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                stl10_normalization(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.train_transform(x)
        k = self.train_transform(x)
        return q, k


@under_review()
class MoCo2EvalSTL10Transforms:
    """MoCo 2 augmentation:

    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self, size: int = 64) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `transforms` from `torchvision` which is not installed yet.")

        self.test_augmentation = transforms.Compose(
            [
                transforms.Resize(size + 11),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                stl10_normalization(),
            ]
        )

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.test_augmentation(x)
        k = self.test_augmentation(x)
        return q, k


@under_review()
class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma: List[float] = [0.1, 2.0]) -> None:
        if not _PIL_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `Pillow` which is not installed yet, install it with `pip install Pillow`."
            )
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])  # noqa: S311
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))
