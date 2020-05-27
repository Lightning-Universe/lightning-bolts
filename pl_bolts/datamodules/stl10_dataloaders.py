from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import STL10

from pl_bolts.datamodules.concat_dataset import ConcatDataset
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders
from pl_bolts.transforms.dataset_normalizations import stl10_normalization


class STL10DataLoaders(BoltDataLoaders):

    def __init__(self, save_path, unlabeled_val_split=5000, train_val_split=500, num_workers=16):
        super().__init__()
        self.save_path = save_path
        self.unlabeled_val_split = unlabeled_val_split
        self.train_val_split = train_val_split
        self.num_workers = num_workers

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        STL10(self.save_path, split='unlabeled', download=True, transform=transform_lib.ToTensor())
        STL10(self.save_path, split='train', download=True, transform=transform_lib.ToTensor())
        STL10(self.save_path, split='test', download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='unlabeled', download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(dataset,
                                        [train_length - self.unlabeled_val_split,
                                         self.unlabeled_val_split])
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def train_dataloader_mixed(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        unlabeled_dataset = STL10(self.save_path,
                                  split='unlabeled',
                                  download=False,
                                  transform=transforms)
        unlabeled_length = len(unlabeled_dataset)
        unlabeled_dataset, _ = random_split(unlabeled_dataset,
                                            [unlabeled_length - self.unlabeled_val_split,
                                             self.unlabeled_val_split])

        labeled_dataset = STL10(self.save_path, split='train', download=False, transform=transforms)
        labeled_length = len(labeled_dataset)
        labeled_dataset, _ = random_split(labeled_dataset,
                                          [labeled_length - self.train_val_split,
                                           self.train_val_split])

        dataset = ConcatDataset(unlabeled_dataset, labeled_dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = STL10(self.save_path, split='unlabeled', download=False, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(dataset,
                                      [train_length - self.unlabeled_val_split,
                                       self.unlabeled_val_split])
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def val_dataloader_mixed(self, batch_size, transforms=None):
        if transforms is None:
            transforms = self._default_transforms()

        unlabeled_dataset = STL10(self.save_path,
                                  split='unlabeled',
                                  download=False,
                                  transform=transforms)
        unlabeled_length = len(unlabeled_dataset)
        _, unlabeled_dataset = random_split(unlabeled_dataset,
                                            [unlabeled_length - self.unlabeled_val_split,
                                             self.unlabeled_val_split])

        labeled_dataset = STL10(self.save_path, split='train', download=False, transform=transforms)
        labeled_length = len(labeled_dataset)
        _, labeled_dataset = random_split(labeled_dataset,
                                          [labeled_length - self.train_val_split,
                                           self.train_val_split])

        dataset = ConcatDataset(unlabeled_dataset, labeled_dataset)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader_unlabeled(self, batch_size, transforms=None):
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
            stl10_normalization()
        ])
        return mnist_transforms
