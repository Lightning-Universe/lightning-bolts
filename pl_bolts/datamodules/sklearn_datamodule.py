import math
from typing import Any
import torch
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from torch.utils.data import Dataset, DataLoader

from pl_bolts.datamodules.lightning_datamodule import LightningDataModule


class SklearnDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, X_transform: Any = None, y_transform: Any = None):
        """
        Mapping between numpy (or sklearn) datasets to PyTorch datasets.

        Args:
            X: Numpy ndarray
            y: Numpy ndarray
            X_transform: Any transform that works with Numpy arrays
            y_transform: Any transform that works with Numpy arrays

        Example:
            >>> from sklearn.datasets import load_boston
            >>> from pl_bolts.datamodules import SklearnDataset
            ...
            >>> X, y = load_boston(return_X_y=True)
            >>> dataset = SklearnDataset(X, y)
            >>> len(dataset)
            506

        """
        super().__init__()
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        y = self.Y[idx]

        # Do not convert integer to float for classification data
        if not y.dtype == np.integer:
            y = y.astype(np.float32)

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y


class TensorDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, X_transform: Any = None, y_transform: Any = None):
        """
        Prepare PyTorch tensor dataset for data loaders.

        Args:
            X: PyTorch tensor
            y: PyTorch tensor
            X_transform: Any transform that works with PyTorch tensors
            y_transform: Any transform that works with PyTorch tensors

        Example:
            >>> from pl_bolts.datamodules import TensorDataset
            ...
            >>> X = torch.rand(10, 3)
            >>> y = torch.rand(10)
            >>> dataset = TensorDataset(X, y)
            >>> len(dataset)
            10

        """
        super().__init__()
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].float()
        y = self.Y[idx]

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y


class SklearnDataModule(LightningDataModule):

    name = 'sklearn'

    def __init__(
            self, X, y,
            x_val=None, y_val=None,
            x_test=None, y_test=None,
            val_split=0.2, test_split=0.1,
            num_workers=2,
            random_state=1234,
            shuffle=True,
            *args,
            **kwargs,
    ):
        """
        Automatically generates the train, validation and test splits for a Numpy dataset. They are set up as
        dataloaders for convenience. Optionally, you can pass in your own validation and test splits.

        Example:

            >>> from sklearn.datasets import load_boston
            >>> from pl_bolts.datamodules import SklearnDataModule
            ...
            >>> X, y = load_boston(return_X_y=True)
            >>> loaders = SklearnDataModule(X, y)
            ...
            >>> # train set
            >>> train_loader = loaders.train_dataloader(batch_size=32)
            >>> len(train_loader.dataset)
            355
            >>> len(train_loader)
            11
            >>> # validation set
            >>> val_loader = loaders.val_dataloader(batch_size=32)
            >>> len(val_loader.dataset)
            100
            >>> len(val_loader)
            3
            >>> # test set
            >>> test_loader = loaders.test_dataloader(batch_size=32)
            >>> len(test_loader.dataset)
            51
            >>> len(test_loader)
            1

        """

        super().__init__(*args, **kwargs)
        self.num_workers = num_workers

        # shuffle x and y
        if shuffle:
            X, y = sk_shuffle(X, y, random_state=random_state)

        val_split = 0 if x_val is not None or y_val is not None else val_split
        test_split = 0 if x_test is not None or y_test is not None else test_split

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            val_split = val_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_holdout, y_holdout = X[: hold_out_size], y[: hold_out_size]
            test_i_start = int(val_split * hold_out_size)
            x_val_hold_out, y_val_holdout = x_holdout[:test_i_start], y_holdout[:test_i_start]
            x_test_hold_out, y_test_holdout = x_holdout[test_i_start:], y_holdout[test_i_start:]
            X, y = X[hold_out_size:], y[hold_out_size:]

        # if don't have x_val and y_val create split from X
        if x_val is None and y_val is None and val_split > 0:
            x_val, y_val = x_val_hold_out, y_val_holdout

        # if don't have x_test, y_test create split from X
        if x_test is None and y_test is None and test_split > 0:
            x_test, y_test = x_test_hold_out, y_test_holdout

        self._init_datasets(X, y, x_val, y_val, x_test, y_test)

    def _init_datasets(self, X, y, x_val, y_val, x_test, y_test):
        self.train_dataset = SklearnDataset(X, y)
        self.val_dataset = SklearnDataset(x_val, y_val)
        self.test_dataset = SklearnDataset(x_test, y_test)

    def train_dataloader(self, batch_size: int = 16):
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size: int = 16):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size: int = 16):
        loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader


class TensorDataModule(SklearnDataModule):
    """
    Automatically generates the train, validation and test splits for a PyTorch tensor dataset. They are set up as
    dataloaders for convenience. Optionally, you can pass in your own validation and test splits.

    Example:

        >>> from pl_bolts.datamodules import TensorDataModule
        >>> import torch
        ...
        >>> # create dataset
        >>> X = torch.rand(100, 3)
        >>> y = torch.rand(100)
        >>> loaders = TensorDataModule(X, y)
        ...
        >>> # train set
        >>> train_loader = loaders.train_dataloader(batch_size=10)
        >>> len(train_loader.dataset)
        70
        >>> len(train_loader)
        7
        >>> # validation set
        >>> val_loader = loaders.val_dataloader(batch_size=10)
        >>> len(val_loader.dataset)
        20
        >>> len(val_loader)
        2
        >>> # test set
        >>> test_loader = loaders.test_dataloader(batch_size=10)
        >>> len(test_loader.dataset)
        10
        >>> len(test_loader)
        1
    """
