import os
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from pl_bolts.utils.warnings import warn_missing_pkg

try:
    from torchvision import transforms as transform_lib
except ModuleNotFoundError:
    warn_missing_pkg("torchvision")  # pragma: no-cover
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class BaseDataModule(LightningDataModule):

    extra_args = {}

    def __init__(
        self,
        dataset_cls,
        dims: Tuple[int, int, int],
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
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

        super().__init__(*args, **kwargs)

        self.dataset_cls = dataset_cls
        self.dims = dims
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def prepare_data(self):
        """
        Saves files to data_dir
        """
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset_dev = self.dataset_cls(self.data_dir, train=True, download=False, **self.extra_args)

            # Split
            self.dataset_train, self.dataset_val = self._split_dataset(dataset_dev)

            # Update transforms
            self.dataset_train.transform = (
                self.default_transforms() if self.train_transforms is None else self.train_transforms
            )
            self.dataset_val.transform = (
                self.default_transforms() if self.val_transforms is None else self.val_transforms
            )

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, download=False, transform=test_transforms, **self.extra_args
            )

    def _split_dataset(self, dataset_dev):
        dev_len = len(dataset_dev)
        splits = self._get_splits(dev_len)
        dataset_train, dataset_val = random_split(
            dataset_dev, splits, generator=torch.Generator().manual_seed(self.seed)
        )
        return dataset_train, dataset_val

    def _get_splits(self, dev_len):
        if isinstance(self.val_split, int):
            train_len = dev_len - self.val_split
            return [train_len, self.val_split]

        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * dev_len)
            train_len = dev_len - val_len
            return [train_len, val_len]

    def default_transforms(self):
        return transform_lib.ToTensor()

    def train_dataloader(self):
        """
        Train set removes a subset to use for validation
        """
        return self._data_loader(self.dataset_train, self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        """
        Val set uses a subset of the training set for validation
        """
        return self._data_loader(self.dataset_val, self.eval_batch_size)

    def test_dataloader(self):
        """
        Test set uses the test split
        """
        return self._data_loader(self.dataset_test, self.eval_batch_size)

    def _data_loader(self, dataset, batch_size, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
