from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders


class CIFAR10DataLoaders(BoltDataLoaders):

    def __init__(self, save_path, val_split=5000, num_workers=16):
        super().__init__()
        self.save_path = save_path
        self.val_split = val_split
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        CIFAR10(self.save_path, train=True, download=True, transform=transform_lib.ToTensor())
        CIFAR10(self.save_path, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = CIFAR10(self.save_path, train=True, download=False, transform=transforms)
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

        dataset = CIFAR10(self.save_path, train=True, download=False, transform=transforms)
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

        dataset = CIFAR10(self.save_path, train=False, download=False, transform=transforms)
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
            self.normalize_transform()
        ])
        return mnist_transforms

    def normalize_transform(self):
        normalize = transform_lib.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
        return normalize
