from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST
import os

from pl_bolts.datamodules.bolts_dataloaders_base import LightningDataModule


class MNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = os.getcwd(), val_split: int = 5000, num_workers: int = 16):
        """
        Standard MNIST, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])

        Example::

            from pl_bolts.datamodules import MNISTDataModule

            dm = MNISTDataModule()
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
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def size(self):
        """
        Return:

            (1, 28, 28)
        """
        return 1, 28, 28

    def prepare_data(self):
        """
        Saves MNIST files to data_dir
        """
        MNIST(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        MNIST(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None):
        """
        MNIST train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is None:
            transforms = self._default_transforms()

        dataset = MNIST(self.data_dir, train=True, download=False, transform=transforms)
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
        MNIST val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is None:
            transforms = self._default_transforms()

        dataset = MNIST(self.data_dir, train=True, download=True, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size, transforms=None):
        """
        MNIST test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """

        if transforms is None:
            transforms = self._default_transforms()

        dataset = MNIST(self.data_dir, train=False, download=False, transform=transforms)
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
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])
        return mnist_transforms
