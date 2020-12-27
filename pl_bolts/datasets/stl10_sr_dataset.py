import numpy as np

from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:
    warn_missing_pkg('PIL', pypi_name='Pillow')  # pragma: no-cover

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import STL10
else:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    STL10 = object


class STL10_SR(STL10):
    def __init__(self, root: str, *args, **kwargs) -> None:
        super().__init__(root, *args, **kwargs)

        # Scale range of HR images to [-1, 1]
        self.hr_transforms = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Scale range of LR images to [0, 1]
        self.lr_transforms = transform_lib.Compose(
            [
                transform_lib.Resize(24, Image.BICUBIC),
                transform_lib.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        data = self.data[index]
        image = Image.fromarray(np.transpose(data, (1, 2, 0)))

        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(image)

        return hr_image, lr_image
