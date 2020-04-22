import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders


class MNISTDataLoaders(BoltDataLoaders):
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self, batch_size, val_split=5000):
        dataset = MNIST(os.getcwd(), train=True, download=False, transform=self.get_transforms())
        mnist_train, _ = random_split(dataset, [60000-val_split, val_split])
        loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        return loader

    def val_dataloader(self, batch_size, val_split=5000):
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=self.get_transforms())
        _, mnist_val = random_split(dataset, [60000-val_split, val_split])
        loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
        return loader

    def test_dataloader(self, batch_size):
        dataset = MNIST(os.getcwd(), train=False, download=False, transform=self.get_transforms())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader

    def get_transforms(self):
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        )
        return mnist_transforms
