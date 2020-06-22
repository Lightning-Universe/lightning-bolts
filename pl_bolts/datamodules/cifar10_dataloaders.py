from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10

from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class MyCIFAR10(CIFAR10):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    TODO: proper dataset shrinking as TrialMNIST
    """
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
    ]


class CIFAR10DataModule(BoltDataModule):

    def __init__(self, save_path, val_split=5000, num_workers=16):
        super().__init__()
        self.save_path = save_path
        self.val_split = val_split
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        MyCIFAR10(self.save_path, train=True, download=True, transform=transform_lib.ToTensor())
        MyCIFAR10(self.save_path, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = MyCIFAR10(self.save_path, train=True, download=False, transform=transforms)
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

    def val_dataloader(self, batch_size, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = MyCIFAR10(self.save_path, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(dataset, [train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = MyCIFAR10(self.save_path, train=False, download=False, transform=transforms)
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
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return mnist_transforms
