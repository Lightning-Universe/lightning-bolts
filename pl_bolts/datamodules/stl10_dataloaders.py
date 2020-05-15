from torchvision import transforms as transform_lib
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders


class STL10DataLoaders(BoltDataLoaders):

    def __init__(self, save_path, val_split=5000, num_workers=16):
        super().__init__()
        self.save_path = save_path
        self.val_split = val_split
        self.num_workers = num_workers

    @property
    def train_length(self):
        return 50000

    def prepare_data(self):
        STL10(self.save_path, split='unlabeled', download=True, transform=transform_lib.ToTensor())
        STL10(self.save_path, split='train', download=True, transform=transform_lib.ToTensor())
        STL10(self.save_path, split='test', download=True, transform=transform_lib.ToTensor())

    def train_dataloader_mixed(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='train+unlabeled', download=False, transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader_unlabeled(self, batch_size,transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='unlabeled', download=False, transform=transforms)
        _, dataset_val = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def train_dataloader_unlabeled(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='unlabeled', download=False, transform=transforms)
        dataset_train, _ = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader_unlabeled(self, batch_size,transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='unlabeled', download=False, transform=transforms)
        _, dataset_val = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='test', download=False, transform=transforms)
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
        normalize = transform_lib.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))
        return normalize
