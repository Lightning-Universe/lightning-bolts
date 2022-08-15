from typing import Any, Tuple

from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import EMNIST
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    EMNIST = object

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


class BinaryEMNIST(EMNIST):
    threshold = 127
    
    """Binarizred EMNIST Dataset."""        
    def __getitem__(self, idx : int) -> Tuple[Any, Any]:
        """
        Args:
            index: Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        img, target = self.data[idx], int(self.targets[idx])

        # Binarize image according to threshold
        img[img < self.threshold] = 0.0
        img[img >= self.threshold] = 255.0

        # Convert to PIL Image (8-bit BW)
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
