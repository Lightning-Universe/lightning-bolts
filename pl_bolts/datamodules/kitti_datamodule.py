import os

from pytorch_lightning import LightningDataModule
from pl_bolts.datamodules.kitti_dataset import KittiDataset

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split


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
        You need to have downloaded the Kitti dataset first and update the data directory.

        You can download the data here: http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

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
        self.num_workers = num_workers

        self.default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])

        # split into train, val, test
        kitti_dataset = KittiDataset(self.data_dir, transform=self.default_transforms)

        val_len = round(val_split * kitti_dataset.__len__())
        test_len = round(test_split * kitti_dataset.__len__())
        train_len = kitti_dataset.__len__() - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(kitti_dataset, lengths=[train_len, val_len, test_len])

    def train_dataloader(self):
        loader = DataLoader(self.trainset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.testset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers)

