from typing import Any, Tuple, Union

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
    """Binarized EMNIST Dataset.

    EMNIST dataset binarized using a thresholding operation. Default threshold value is 127.
    Note that the images are binarized prior to the application of any transforms.

    Args:
        root (string): Root directory of dataset where ``EMNIST/raw/train-images-idx3-ubyte``
            and  ``EMNIST/raw/t10k-images-idx3-ubyte`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        threshold (Union[int, float], optional): Threshold value for binarizing image.
            Pixel value is set to 255 if value is greater than threshold, otherwise 0.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    Note:
        Documentation is based on https://pytorch.org/vision/main/generated/torchvision.datasets.EMNIST.html
    """

    def __init__(self, root: str, split: str, threshold: Union[int, float] = 127.0, **kwargs: Any) -> None:
        super().__init__(root, split, **kwargs)
        self.threshold = threshold

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            index: Index

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
