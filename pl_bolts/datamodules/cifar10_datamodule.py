import os
import pickle
import tarfile
from typing import Tuple, Optional, Sequence, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib

from pl_bolts.datamodules.base_dataset import LightDataset
from pl_bolts.datamodules.lightning_datamodule import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class CIFAR10(LightDataset):
    """
    Customized `CIFAR10 <http://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset for testing Pytorch Lightning
    without the torchvision dependency.

    Part of the code was copied from
    https://github.com/pytorch/vision/blob/build/v0.5.0/torchvision/datasets/

    Args:
        root: Root directory of dataset where ``CIFAR10/processed/training.pt``
            and  ``CIFAR10/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    Examples:

        >>> dataset = CIFAR10(download=True)
        >>> len(dataset)
        50000
        >>> torch.bincount(dataset.targets)
        tensor([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000])
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([3, 32, 32])
        >>> label
        6
    """

    BASE_URL = "https://www.cs.toronto.edu/~kriz/"
    FILE_NAME = 'cifar-10-python.tar.gz'
    cache_folder_name = 'complete'
    TRAIN_FILE_NAME = 'training.pt'
    TEST_FILE_NAME = 'test.pt'
    DATASET_NAME = 'CIFAR10'
    labels = set(range(10))

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Callable = None,
            download: bool = True
    ):
        super().__init__()
        self.root_path = root
        self.train = train  # training set or test set
        self.transform = transform

        os.makedirs(self.cached_folder_path, exist_ok=True)
        self.prepare_data(download)

        if not self._check_exists(self.cached_folder_path, (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME)):
            raise RuntimeError('Dataset not found.')

        data_file = self.TRAIN_FILE_NAME if self.train else self.TEST_FILE_NAME
        self.data, self.targets = torch.load(os.path.join(self.cached_folder_path, data_file))

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        img = self.data[idx].float().reshape(3, 32, 32)
        target = int(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    @classmethod
    def _check_exists(cls, data_folder: str, file_names: Sequence[str]) -> bool:
        if isinstance(file_names, str):
            file_names = [file_names]
        return all(os.path.isfile(os.path.join(data_folder, fname))
                   for fname in file_names)

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
        torch.save(self._unpickle(path_content, 'test_batch'),
                   os.path.join(self.cached_folder_path, self.TEST_FILE_NAME))
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

        base_path = os.path.join(self.root_path, self.DATASET_NAME)
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

    Args:
        root: Root directory of dataset where ``CIFAR10/processed/training.pt``
            and  ``CIFAR10/processed/test.pt`` exist.
        train: If ``True``, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        normalize: mean and std deviation of the MNIST dataset.
        download: If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        num_samples: number of examples per selected class/digit
        labels: list selected MNIST digits/classes

    Examples:

        >>> dataset = TrialCIFAR10(download=True, num_samples=150)
        >>> len(dataset)
        450
        >>> sorted(set([d.item() for d in dataset.targets]))
        [0, 1, 2]
        >>> torch.bincount(dataset.targets)
        tensor([150, 150, 150])
        >>> data, label = dataset[0]
        >>> data.shape
        torch.Size([3, 32, 32])
    """
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Callable = None,
            download: bool = False,
            num_samples: int = 100,
            labels: Optional[Sequence] = (0, 1, 2)
    ):
        # number of examples per class
        self.num_samples = num_samples
        # take just a subset of CIFAR dataset
        self.labels = labels if labels else list(range(10))

        self.cache_folder_name = f'labels-{"-".join(str(d) for d in sorted(self.labels))}_nb-{self.num_samples}'

        super().__init__(
            root,
            train=train,
            transform=transform,
            download=download
        )

    def prepare_data(self, download: bool) -> None:
        super().prepare_data(download)

        for fname in (self.TRAIN_FILE_NAME, self.TEST_FILE_NAME):
            path_fname = os.path.join(super().cached_folder_path, fname)
            assert os.path.isfile(path_fname), 'Missing cached file: %s' % path_fname
            data, targets = torch.load(path_fname)
            data, targets = self._prepare_subset(data, targets, self.num_samples, self.labels)
            torch.save((data, targets), os.path.join(self.cached_folder_path, fname))


class CIFAR10DataModule(LightningDataModule):

    def __init__(self, data_dir, val_split=5000, num_workers=16):
        """
        Standard CIFAR10, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])

        Example::

            from pl_bolts.datamodules import CIFAR10DataModule

            dm = CIFAR10DataModule()
            model = LitModel(datamodule=dm)

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
        """
        super().__init__()
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        if hasattr(self, '_train_dataset'):
            return len(self._train_dataset.labels)
        return 10

    @property
    def size(self) -> Tuple:
        """
        Return:

            (1, 32, 32)
        """
        if hasattr(self, '_train_dataset'):
            return self._train_dataset[0].shape
        return 3, 32, 32

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        self._train_dataset = CIFAR10(self.data_dir, train=True, download=True)
        self._test_dataset = CIFAR10(self.data_dir, train=False, download=True)

    def train_dataloader(self, batch_size, transforms=None):
        """
        CIFAR train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is None:
            transforms = self._default_transforms()

        dataset = CIFAR10(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(dataset, [train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size, transforms=None):
        """
        CIFAR10 val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """

        if transforms is None:
            transforms = self._default_transforms()

        dataset = CIFAR10(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size, transforms=None):
        """
        CIFAR10 test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is None:
            transforms = self._default_transforms()

        dataset = CIFAR10(self.data_dir, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        cf10_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return cf10_transforms
