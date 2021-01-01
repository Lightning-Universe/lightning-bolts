import numpy as np

from pl_bolts.datasets.sr_dataset_mixin import SRDatasetMixin
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    import PIL
else:
    warn_missing_pkg("PIL", pypi_name="Pillow")  # pragma: no-cover

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import STL10
else:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    STL10 = object


class SRSTL10Dataset(SRDatasetMixin, STL10):
    # TODO: add docs
    def __init__(self, hr_image_size: int, lr_image_size: int, image_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hr_image_size = hr_image_size
        self.lr_image_size = lr_image_size
        self.image_channels = image_channels

    def _get_image(self, index: int):
        return PIL.Image.fromarray(np.transpose(self.data[index], (1, 2, 0)))
