import os

from pl_bolts.datasets.sr_dataset_mixin import SRDatasetMixin
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    import PIL
else:
    warn_missing_pkg("PIL", pypi_name="Pillow")  # pragma: no-cover

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import CelebA
else:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    CelebA = object


class SRCelebADataset(SRDatasetMixin, CelebA):
    # TODO: add docs
    def __init__(self, hr_image_size: int, lr_image_size: int, image_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hr_image_size = hr_image_size
        self.lr_image_size = lr_image_size
        self.image_channels = image_channels

    def _get_image(self, index: int):
        return PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))