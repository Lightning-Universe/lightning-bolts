from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders


class MNISTDataLoaders(BoltDataLoaders):

    def __init__(self, save_path, val_split=5000, num_workers=16):
        super().__init__()
        self.save_path = save_path
        self.val_split = val_split
        self.num_workers = num_workers

    @property
    def train_length(self):
        return 60000

    def prepare_data(self):
        MNIST(self.save_path, train=True, download=True, transform=transform_lib.ToTensor())
        MNIST(self.save_path, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None, use_default_normalize=True):
        if transforms is None:
            transforms = self.get_transforms()

        dataset = MNIST(self.save_path, train=True, download=False, transform=transforms)
        dataset_train, _ = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        return loader

    def val_dataloader(self, batch_size,transforms=None, use_default_normalize=True):
        if transforms is None:
            transforms = self.get_transforms()

        dataset = MNIST(self.save_path, train=True, download=True, transform=transforms)
        _, dataset_val = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
        )
        return loader

    def test_dataloader(self, batch_size, transforms=None, use_default_normalize=True):
        if transforms is None:
            transforms = self.get_transforms()

        dataset = MNIST(self.save_path, train=False, download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True
        )
        return loader

    def default_transforms(self):
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])
        return mnist_transforms
