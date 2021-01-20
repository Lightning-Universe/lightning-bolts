# type: ignore[override]
import os
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from pl_bolts.datasets import ConcatDataset
from pl_bolts.transforms.dataset_normalizations import stl10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import STL10
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


class STL10DataModule(LightningDataModule):  # pragma: no cover
    """
    .. figure:: https://samyzaf.com/ML/cifar10/cifar1.jpg
        :width: 400
        :alt: STL-10

    Specs:
        - 10 classes (1 per type)
        - Each image is (3 x 96 x 96)

    Standard STL-10, train, val, test splits and transforms.
    STL-10 has support for doing validation splits on the labeled or unlabeled splits

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=(0.43, 0.42, 0.39),
                std=(0.27, 0.26, 0.27)
            )
        ])

    Example::

        from pl_bolts.datamodules import STL10DataModule

        dm = STL10DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)
    """

    name = 'stl10'

    def __init__(
        self,
        data_dir: Optional[str] = None,
        unlabeled_val_split: int = 5000,
        train_val_split: int = 500,
        num_workers: int = 16,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to save/load the data
            unlabeled_val_split: how many images from the unlabeled training split to use for validation
            train_val_split: how many images from the labeled training split to use for validation
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                'You want to use STL10 dataset loaded from `torchvision` which is not installed yet.'
            )

        self.dims = (3, 96, 96)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.unlabeled_val_split = unlabeled_val_split
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_unlabeled_samples = 100000 - unlabeled_val_split

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        """
        Downloads the unlabeled, train and test split
        """
        STL10(self.data_dir, split='unlabeled', download=True, transform=transform_lib.ToTensor())
        STL10(self.data_dir, split='train', download=True, transform=transform_lib.ToTensor())
        STL10(self.data_dir, split='test', download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self) -> DataLoader:
        """
        Loads the 'unlabeled' split minus a portion set aside for validation via `unlabeled_val_split`.
        """
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = STL10(self.data_dir, split='unlabeled', download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.unlabeled_val_split, self.unlabeled_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def train_dataloader_mixed(self) -> DataLoader:
        """
        Loads a portion of the 'unlabeled' training data and 'train' (labeled) data.
        both portions have a subset removed for validation via `unlabeled_val_split` and `train_val_split`

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        unlabeled_dataset = STL10(self.data_dir, split='unlabeled', download=False, transform=transforms)
        unlabeled_length = len(unlabeled_dataset)
        unlabeled_dataset, _ = random_split(
            unlabeled_dataset, [unlabeled_length - self.unlabeled_val_split, self.unlabeled_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        labeled_dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        labeled_length = len(labeled_dataset)
        labeled_dataset, _ = random_split(
            labeled_dataset, [labeled_length - self.train_val_split, self.train_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        dataset = ConcatDataset(unlabeled_dataset, labeled_dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """
        Loads a portion of the 'unlabeled' training data set aside for validation
        The val dataset = (unlabeled - train_val_split)

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = STL10(self.data_dir, split='unlabeled', download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset, [train_length - self.unlabeled_val_split, self.unlabeled_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader_mixed(self) -> DataLoader:
        """
        Loads a portion of the 'unlabeled' training data set aside for validation along with
        the portion of the 'train' dataset to be used for validation

        unlabeled_val = (unlabeled - train_val_split)

        labeled_val = (train- train_val_split)

        full_val = unlabeled_val + labeled_val

        Args:

            batch_size: the batch size
            transforms: a sequence of transforms
        """
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        unlabeled_dataset = STL10(self.data_dir, split='unlabeled', download=False, transform=transforms)
        unlabeled_length = len(unlabeled_dataset)
        _, unlabeled_dataset = random_split(
            unlabeled_dataset, [unlabeled_length - self.unlabeled_val_split, self.unlabeled_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        labeled_dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        labeled_length = len(labeled_dataset)
        _, labeled_dataset = random_split(
            labeled_dataset, [labeled_length - self.train_val_split, self.train_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        dataset = ConcatDataset(unlabeled_dataset, labeled_dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """
        Loads the test split of STL10

        Args:
            batch_size: the batch size
            transforms: the transforms
        """
        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = STL10(self.data_dir, split='test', download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def train_dataloader_labeled(self) -> DataLoader:
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.train_val_split, self.train_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader_labeled(self) -> DataLoader:
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = STL10(self.data_dir, split='train', download=False, transform=transforms)
        labeled_length = len(dataset)
        _, labeled_val = random_split(
            dataset, [labeled_length - self.train_val_split, self.train_val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )

        loader = DataLoader(
            labeled_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def _default_transforms(self) -> Callable:
        data_transforms = transform_lib.Compose([transform_lib.ToTensor(), stl10_normalization()])
        return data_transforms
