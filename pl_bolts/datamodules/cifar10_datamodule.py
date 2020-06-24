from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10

from pl_bolts.datamodules.lightning_datamodule import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization


class CIFAR10DataModule(LightningDataModule):

    def __init__(self,
                 data_dir,
                 val_split=5000,
                 num_workers=16,
                 train_transforms=None,
                 val_transforms=None,
                 test_transforms=None):
        """
        Standard CIFAR10, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            ])

        Example::

            from pl_bolts.datamodules import CIFAR10DataModule

            dm = CIFAR10DataModule()
            model = LitModel(datamodule=dm)

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            train_transforms: Optional set of transforms to use for training
            val_transforms: Optional set of transforms to use for validation
            test_transforms: Optional set of transforms to use for testing
        """
        super().__init__()
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms

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

            (1, 32, 32)
        """
        return 3, 32, 32

    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        CIFAR10(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        CIFAR10(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None):
        """
        CIFAR train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is not None:
            self.train_transforms = transforms

        if self.train_transforms is None:
            self.train_transforms = self._default_transforms()

        dataset = CIFAR10(self.data_dir, train=True, download=False, transform=self.train_transforms)
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
        CIFAR10 val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is not None:
            self.val_transforms = transforms

        if self.val_transforms is None:
            self.val_transforms = self._default_transforms()

        dataset = CIFAR10(self.data_dir, train=True, download=False, transform=self.val_transforms)
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

    def test_dataloader(self, batch_size, transforms=None):
        """
        CIFAR10 test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        if transforms is not None:
            self.test_transforms = transforms

        if self.test_transforms is None:
            self.test_transforms = self._default_transforms()

        dataset = CIFAR10(self.data_dir, train=False, download=False, transform=self.test_transforms)
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
        cf10_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            cifar10_normalization()
        ])
        return cf10_transforms
