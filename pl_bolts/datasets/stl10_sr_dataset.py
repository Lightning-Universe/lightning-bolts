import numpy as np
from PIL import Image

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import STL10
except ModuleNotFoundError:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class STL10_SR(STL10):
    def __init__(self, root, *args, **kwargs):
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
