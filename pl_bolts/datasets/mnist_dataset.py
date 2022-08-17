from typing import Any, Tuple

from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE, _TORCHVISION_LESS_THAN_0_9_1
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import MNIST
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    MNIST = object

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


class BinaryMNIST(MNIST):
    """Binarized MNIST Dataset.

    MNIST dataset binarized using a thresholding operation. Threshold is set to 127. Note that the images are binarized
    prior to the application of any transforms.
    """

    threshold = 127.0

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        img, target = self.data[idx], int(self.targets[idx])

        # Convert to PIL Image (8-bit BW)
        img = Image.fromarray(img.numpy(), mode="L")

        # Binarize image at threshold
        img = img.point(lambda p: 255 if p > self.threshold else 0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
