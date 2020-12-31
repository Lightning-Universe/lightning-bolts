from pl_bolts.datasets.sr_dataset_mixin import SRDatasetMixin
from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    import PIL
else:
    warn_missing_pkg("PIL", pypi_name="Pillow")  # pragma: no-cover

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import MNIST
else:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    MNIST = object


class BinaryMNIST(MNIST):
    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
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
    # TODO: add docs
    def __init__(self, hr_image_size: int, lr_image_size: int, image_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hr_image_size = hr_image_size
        self.lr_image_size = lr_image_size
        self.image_channels = image_channels

    def _get_image(self, index: int):
        return PIL.Image.fromarray(self.data[index].numpy(), mode="L")