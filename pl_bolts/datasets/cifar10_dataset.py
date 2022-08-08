import os
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from pl_bolts.utils import _PIL_AVAILABLE, _TORCHVISION_AVAILABLE, _TORCHVISION_LESS_THAN_0_9_1
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets import CIFAR10
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    CIFAR10 = object

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg("PIL", pypi_name="Pillow")


@under_review()
class TrialCIFAR10(CIFAR10):
    """
    Customized `CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.
    Examples:

        >>> dataset = TrialCIFAR10(download=True, num_samples=150, labels=(1, 5, 8), data_dir="datasets")
        >>> len(dataset)
        450
        >>> sorted(set([d.item() for d in dataset.targets]))
        [1, 5, 8]
        >>> torch.bincount(dataset.targets)
        tensor([  0, 150,   0,   0,   0, 150,   0,   0, 150])
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([3, 32, 32])
    """

    def __init__(
        self,
        data_dir: str = ".",
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        num_samples: int = 100,
        labels: Optional[Sequence] = (1, 5, 8),
        relabel: bool = True,
    ):
        """
        Args:
            data_dir: Root directory of dataset where ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            train: If ``True``, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download: If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            num_samples: number of examples per selected class/digit
            labels: list selected CIFAR10 digits/classes
        """
        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of CIFAR dataset
        self.labels = labels if labels else list(range(10))
        self.relabel = relabel

        self.cache_folder_name = f'labels-{"-".join(str(d) for d in sorted(self.labels))}_nb-{self.num_samples}'

        super().__init__(data_dir, train=train, transform=transform, download=download)

    def prepare_data(self, download: bool) -> None:
        super().prepare_data(download)

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(super().cached_folder_path, fname)
            assert os.path.isfile(path_fname), "Missing cached file: %s" % path_fname
            data, targets = torch.load(path_fname)
            if self.num_samples or len(self.labels) < 10:
                data, targets = self._prepare_subset(data, targets, self.num_samples, self.labels)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))
