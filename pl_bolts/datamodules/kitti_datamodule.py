import os

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

