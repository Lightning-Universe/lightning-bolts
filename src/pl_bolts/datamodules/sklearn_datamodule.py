import math
from typing import Any, Tuple

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pl_bolts.utils import _SKLEARN_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _SKLEARN_AVAILABLE:
    from sklearn.utils import shuffle as sk_shuffle
else:  # pragma: no cover
    warn_missing_pkg("sklearn", pypi_name="scikit-learn")


@under_review()
class SklearnDataset(Dataset):
    """Mapping between numpy (or sklearn) datasets to PyTorch datasets.

    Args:
        X: Numpy ndarray
        y: Numpy ndarray
        x_transform: Any transform that works with Numpy arrays
        y_transform: Any transform that works with Numpy arrays

    Example:
        >>> from sklearn.datasets import load_diabetes
        >>> from pl_bolts.datamodules import SklearnDataset
        ...
        >>> X, y = load_diabetes(return_X_y=True)
        >>> dataset = SklearnDataset(X, y)
        >>> len(dataset)
        442

    """

    def __init__(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        x_transform: Any = None,
        y_transform: Any = None,
    ) -> None:
        super().__init__()
        self.data = X
        self.labels = y
        self.data_transform = x_transform
        self.labels_transform = y_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = self.data[idx].astype(np.float32)
        y = self.labels[idx]

        # Do not convert integer to float for classification data
        if not ((y.dtype == np.int32) or (y.dtype == np.int64)):
            y = y.astype(np.float32)

        if self.data_transform:
            x = self.data_transform(x)

        if self.labels_transform:
            y = self.labels_transform(y)

        return x, y


@under_review()
class SklearnDataModule(LightningDataModule):
    """Automatically generates the train, validation and test splits for a Numpy dataset. They are set up as dataloaders
    for convenience. Optionally, you can pass in your own validation and test splits.

    Example:

        >>> from sklearn.datasets import load_diabetes
        >>> from pl_bolts.datamodules import SklearnDataModule
        ...
        >>> X, y = load_diabetes(return_X_y=True)
        >>> loaders = SklearnDataModule(X, y, batch_size=32)
        ...
        >>> # train set
        >>> train_loader = loaders.train_dataloader()
        >>> len(train_loader.dataset)
        310
        >>> len(train_loader)
        10
        >>> # validation set
        >>> val_loader = loaders.val_dataloader()
        >>> len(val_loader.dataset)
        88
        >>> len(val_loader)
        3
        >>> # test set
        >>> test_loader = loaders.test_dataloader()
        >>> len(test_loader.dataset)
        44
        >>> len(test_loader)
        2

    """

    name = "sklearn"

    def __init__(
        self,
        X,  # noqa: N803
        y,
        x_val=None,
        y_val=None,
        x_test=None,
        y_test=None,
        val_split=0.2,
        test_split=0.1,
        num_workers=0,
        random_state=1234,
        shuffle=True,
        batch_size: int = 16,
        pin_memory=True,
        drop_last=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # shuffle x and y
        if shuffle and _SKLEARN_AVAILABLE:
            X, y = sk_shuffle(X, y, random_state=random_state)  # noqa: N806
        elif shuffle and not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use shuffle function from `scikit-learn` which is not installed yet."
            )

        val_split = 0 if x_val is not None or y_val is not None else val_split
        test_split = 0 if x_test is not None or y_test is not None else test_split

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            val_split = val_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_holdout, y_holdout = X[:hold_out_size], y[:hold_out_size]
            test_i_start = int(val_split * hold_out_size)
            x_val_hold_out, y_val_holdout = x_holdout[:test_i_start], y_holdout[:test_i_start]
            x_test_hold_out, y_test_holdout = x_holdout[test_i_start:], y_holdout[test_i_start:]
            X, y = X[hold_out_size:], y[hold_out_size:]  # noqa: N806

        # if don't have x_val and y_val create split from X
        if x_val is None and y_val is None and val_split > 0:
            x_val, y_val = x_val_hold_out, y_val_holdout

        # if don't have x_test, y_test create split from X
        if x_test is None and y_test is None and test_split > 0:
            x_test, y_test = x_test_hold_out, y_test_holdout

        self._init_datasets(X, y, x_val, y_val, x_test, y_test)

    def _init_datasets(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.train_dataset = SklearnDataset(x, y)
        self.val_dataset = SklearnDataset(x_val, y_val)
        self.test_dataset = SklearnDataset(x_test, y_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
