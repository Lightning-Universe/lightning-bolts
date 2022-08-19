import os
from typing import Any

from pl_bolts.datasets.sr_dataset_mixin import SRDatasetMixin
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import CelebA
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    CelebA = object


@under_review()
class SRCelebA(SRDatasetMixin, CelebA):
    """CelebA dataset that can be used to train Super Resolution models.

    Function __getitem__ (implemented in SRDatasetMixin) returns tuple of high and low resolution image.
    """

    def __init__(self, scale_factor: int, *args: Any, **kwargs: Any) -> None:
        hr_image_size = 128
        lr_image_size = hr_image_size // scale_factor
        self.image_channels = 3
        super().__init__(hr_image_size, lr_image_size, self.image_channels, *args, **kwargs)

    def _get_image(self, index: int):
        return Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
