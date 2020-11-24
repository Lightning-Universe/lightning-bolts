import os
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import VisionDataset
except ModuleNotFoundError:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class BaseDataModule(LightningDataModule):

    extra_args = {}

    def __init__(
        self,
        dataset_cls: VisionDataset,
        dims: Tuple[int, int, int],
        data_dir: Optional[str] = None,
        val_split: int = 5000,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
        """

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                "You want to use VisionDataset loaded from `torchvision` which is not installed yet."
            )

        super().__init__(*args, **kwargs)

        self.dataset_cls = dataset_cls
        self.dims = dims
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size

    def prepare_data(self):
        """
        Saves files to data_dir
        """
        self.dataset_cls(
            self.data_dir, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args
        )
        self.dataset_cls(
            self.data_dir, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args
        )

    def train_dataloader(self):
        """
        Train set removes a subset to use for validation
        """
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
        dataset = self.dataset_cls(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """
        Val set uses a subset of the training set for validation
        """
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
        dataset = self.dataset_cls(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset, [train_length - self.val_split, self.val_split], generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """
        Test set uses the test split
        """
        transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
        dataset = self.dataset_cls(self.data_dir, train=False, download=False, transform=transforms, **self.extra_args)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
