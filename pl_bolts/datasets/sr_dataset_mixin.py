"""Adapted from: https://github.com/https-deeplearning-ai/GANs-Public."""
from typing import Any, Tuple

import torch

from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class SRDatasetMixin:
    """Mixin for Super Resolution datasets.

    Scales range of high resolution images to [-1, 1] and range or low resolution images to [0, 1].
    """

    def __init__(self, hr_image_size: int, lr_image_size: int, image_channels: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.hr_transforms = transform_lib.Compose(
            [
                transform_lib.RandomCrop(hr_image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=(0.5,) * image_channels, std=(0.5,) * image_channels),
            ]
        )

        self.lr_transforms = transform_lib.Compose(
            [
                transform_lib.Normalize(mean=(-1.0,) * image_channels, std=(2.0,) * image_channels),
                transform_lib.ToPILImage(),
                transform_lib.Resize(lr_image_size, Image.BICUBIC),
                transform_lib.ToTensor(),
            ]
        )

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._get_image(index)

        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        return hr_image, lr_image
