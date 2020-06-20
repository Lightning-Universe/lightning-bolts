from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as sk_shuffle
import math


class SklearnDataset(Dataset):
    def __init__(self, X, y, X_transform=None, y_transform=None):
        """
        Mapping between numpy (or sklearn) datasets to PyTorch datasets.


        Example:
            >>> from sklearn.datasets import load_boston
            >>> from pl_bolts.datamodules import SklearnDataset
            >>> from torch.utils.data import DataLoader
            >>>
            >>> X, y = load_boston(return_X_y=True)
            >>> dataset = SklearnDataset(X, y)
            >>> len(dataset)
            506

        """
        self.X = X
        self.Y = y
        self.X_transform = X_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        if self.X_transform:
            x = self.X_transform(x)

        if self.y_transform:
            y = self.y_transform(y)

        return x, y


class SklearnDataLoaders(BoltDataLoaders):

    def __init__(self, X, y, x_val=None, y_val=None, x_test=None, y_test=None, val_split=0.15, test_split=0.15, num_workers=2, random_state=1234, shuffle=True):
        super().__init__()

        self.num_workers = num_workers

        # shuffle x and y
        if shuffle:
            X, y = sk_shuffle(X, y, random_state=random_state)

        val_split = 0 if x_val is not None or y_val is not None else val_split
        test_split = 0 if x_test is not None or y_test is not None else test_split

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            val_split = val_split / hold_out_split
            test_split = test_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_split, y_split = X[: hold_out_size], y[: hold_out_size]
            X, y = X[hold_out_size:], y[hold_out_size:]

        # if don't have x_val and y_val create split from X
        if x_val is None and y_val is None:
            val_size = int(math.floor(val_split * len(x_split)))
            x_val, y_val = x_split[:val_size], y_split[:val_size]
            x_split, y_split = x_split[val_size:], y_split[val_size:]

        # if don't have x_test, y_test create split from X
        if x_test is None and y_test is None:
            test_size = int(math.floor(test_split * len(x_split)))
            x_test, y_test = x_split[test_size:], y_split[test_size:]

        self.train_dataset = SklearnDataset(X, y)
        self.val_dataset = SklearnDataset(x_val, y_val)
        self.test_dataset = SklearnDataset(x_test, y_test)

    def train_dataloader(self, batch_size: int = 32):
        loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size: int = 32):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size=32):
        loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader


