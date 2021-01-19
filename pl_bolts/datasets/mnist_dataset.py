import os

from pl_bolts.datasets.sr_dataset_mixin import SRDatasetMixin
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import MNIST
else:  # pragma: no cover
    warn_missing_pkg('torchvision')
    MNIST = object

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg('PIL', pypi_name='Pillow')


class BinaryMNIST(MNIST):

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError('You want to use `torchvision` which is not installed yet.')

        img, target = self.data[idx], int(self.targets[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # binary
        img[img < 0.5] = 0.0
        img[img >= 0.5] = 1.0

        return img, target


class SRMNISTDataset(SRDatasetMixin, MNIST):
    """
    MNIST dataset that can be used to train Super Resolution models.

    Function __getitem__ (implemented in SRDatasetMixin) returns tuple of high and low resolution image.

    """

    def __init__(self, scale_factor: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hr_image_size = 28
        self.lr_image_size = self.hr_image_size // scale_factor
        self.image_channels = 1

    def _get_image(self, index: int):
        return Image.fromarray(self.data[index].numpy(), mode="L")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")
