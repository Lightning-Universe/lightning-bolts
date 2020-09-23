import os
import math

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from pl_bolts.datamodules.kitti_dataset import KittiDataset

class KittiDataModule(LightningDataModule):

    name = 'kitti'

    def __init__(
            self,
            data_dir: str = '/Users/annikabrundyn/Documents/data_semantics',
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        Standard Kitti train, val, test splits and transforms.

        Example::

            from pl_bolts.datamodules import KittiDataModule

            dm = KittiDataModule(PATH)
            model = LitModel()

            Trainer().fit(model, dm)

        Args::
            data_dir: where to load the data from (note these needs to be downloaded in advance)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
        """
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size

        self.default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])

        hold_out_split = val_split + test_split
        if hold_out_split > 0:
            val_split = val_split / hold_out_split
            hold_out_size = math.floor(len(X) * hold_out_split)
            x_holdout, y_holdout = X[: hold_out_size], y[: hold_out_size]
            test_i_start = int(val_split * hold_out_size)
            x_val_hold_out, y_val_holdout = x_holdout[:test_i_start], y_holdout[:test_i_start]
            x_test_hold_out, y_test_holdout = x_holdout[test_i_start:], y_holdout[test_i_start:]
            X, y = X[hold_out_size:], y[hold_out_size:]

        # create validation split
        if val_split > 0:
            x_val, y_val = x_val_hold_out, y_val_holdout

        # create test split
        if test_split > 0:
            x_test, y_test = x_test_hold_out, y_test_holdout

        self._init_datasets(X, y, x_val, y_val, x_test, y_test)

    def _init_datasets(self, X, y, x_val, y_val, x_test, y_test):
        self.train_dataset = SklearnDataset(X, y)
        self.val_dataset = SklearnDataset(x_val, y_val)
        self.test_dataset = SklearnDataset(x_test, y_test)

        self.trainset = KittiDataset(self.data_dir, split='train', transform=self.default_transforms)
        self.validset = KittiDataset(self.data_dir, split='valid', transform=self.default_transforms)

    def train_dataloader(self):
        loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        return

