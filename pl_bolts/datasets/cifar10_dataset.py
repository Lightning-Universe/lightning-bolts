import os
import pickle
import tarfile
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from pl_bolts.datasets import LightDataset
from pl_bolts.utils import _PIL_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _PIL_AVAILABLE:
    from PIL import Image
else:  # pragma: no cover
    warn_missing_pkg('PIL', pypi_name='Pillow')


class CIFAR10(LightDataset):
    """
    Customized `CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/

    Args:
        data_dir: Root directory of dataset where ``CIFAR10/processed/training.pt``
            and  ``CIFAR10/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    Examples:

        >>> from torchvision import transforms
        >>> from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
        >>> cf10_transforms = transforms.Compose([transforms.ToTensor(), cifar10_normalization()])
        >>> dataset = CIFAR10(download=True, transform=cf10_transforms, data_dir="datasets")
        >>> len(dataset)
        50000
        >>> torch.bincount(dataset.targets)
        tensor([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([3, 32, 32])
        >>> label
        6

    Labels::

        airplane: 0
        automobile: 1
        bird: 2
        cat: 3
        deer: 4
        dog: 5
        frog: 6
        horse: 7
        ship: 8
        truck: 9
    """

    BASE_URL = "https://www.cs.toronto.edu/~kriz/"
    FILE_NAME = 'cifar-10-python.tar.gz'
    cache_folder_name = 'complete'
    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    DATASET_NAME = 'CIFAR10'
    labels = set(range(10))
    relabel = False

    def __init__(
        self, data_dir: str = '.', train: bool = True, transform: Optional[Callable] = None, download: bool = True
    ):
        super().__init__()
        self.dir_path = data_dir
        self.train = train  # training set or test set
        self.transform = transform

        if not _PIL_AVAILABLE:
            raise ImportError('You want to use PIL.Image for loading but it is not installed yet.')

        os.makedirs(self.cached_folder_path, exist_ok=True)
        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].reshape(3, 32, 32)
        target = int(self.targets[idx])

        if self.transform is not None:
            img = img.numpy().transpose((1, 2, 0))  # convert to HWC
            img = self.transform(Image.fromarray(img))
        if self.relabel:
            target = list(self.labels).index(target)
        return img, target

    @classmethod
    def _check_exists(cls, data_folder: str, file_names: Sequence[str]) -> bool:
        if isinstance(file_names, str):
            file_names = [file_names]
        return all(os.path.isfile(os.path.join(data_folder, fname)) for fname in file_names)

    def _unpickle(self, path_folder: str, file_name: str) -> Tuple[Tensor, Tensor]:
        with open(os.path.join(path_folder, file_name), 'rb') as fo:
            pkl = pickle.load(fo, encoding='bytes')
        return torch.tensor(pkl[b'data']), torch.tensor(pkl[b'labels'])

    def _extract_archive_save_torch(self, download_path):
        # extract achieve
        with tarfile.open(os.path.join(download_path, self.FILE_NAME), 'r:gz') as tar:
            tar.extractall(path=download_path)
        # this is internal path in the archive
        path_content = os.path.join(download_path, 'cifar-10-batches-py')

        # load Test and save as PT
        torch.save(
            self._unpickle(path_content, 'test_batch'), os.path.join(self.cached_folder_path, self.TEST_FILE_NAME)
        )
        # load Train and save as PT
        data, labels = [], []
        for i in range(5):
            fname = f'data_batch_{i + 1}'
            _data, _labels = self._unpickle(path_content, fname)
            data.append(_data)
            labels.append(_labels)
        # stash all to one
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        # and save as PT
        torch.save((data, labels), os.path.join(self.cached_folder_path, self.TRAIN_FILE_NAME))

    def prepare_data(self, download: bool):
        if self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            return

        base_path = os.path.join(self.dir_path, self.DATASET_NAME)
        if download:
            self.download(base_path)
        self._extract_archive_save_torch(base_path)

    def download(self, data_folder: str) -> None:
        """Download the data if it doesn't exist in cached_folder_path already."""
        if self._check_exists(data_folder, self.FILE_NAME):
            return
        self._download_from_url(self.BASE_URL, data_folder, self.FILE_NAME)


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
        data_dir: str = '.',
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
            assert os.path.isfile(path_fname), 'Missing cached file: %s' % path_fname
            data, targets = torch.load(path_fname)
            if self.num_samples or len(self.labels) < 10:
                data, targets = self._prepare_subset(data, targets, self.num_samples, self.labels)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))
