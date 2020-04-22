import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_datamodule import BoltDataModule


class MNISTDataModule(BoltDataModule):
    def prepare_data(self):
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        dataset = MNIST(os.getcwd(), train=True, download=False, transform=self.get_transforms())
        mnist_train, _ = random_split(dataset, [55000, 5000])
        loader = DataLoader(mnist_train, batch_size=self.batch_size)
        return loader

    def val_dataloader(self):
        dataset = MNIST(os.getcwd(), train=True, download=True, transform=self.get_transforms())
        _, mnist_val = random_split(dataset, [55000, 5000])
        loader = DataLoader(mnist_val, batch_size=self.batch_size)
        return loader

    def test_dataloader(self):
        dataset = MNIST(os.getcwd(), train=False, download=False, transform=self.get_transforms())
        loader = DataLoader(dataset, batch_size=self.batch_size)
        return loader

    def get_transforms(self):
        mnist_transforms = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        )
        return mnist_transforms
