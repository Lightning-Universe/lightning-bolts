from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import FashionMNIST
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class FashionMNISTDataModule(LightningDataModule):

    name = 'fashion_mnist'

    def __init__(
            self,
            data_dir: str,
            val_split: int = 5000,
            num_workers: int = 16,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        """
        .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/
            wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset.png
            :width: 400
            :alt: Fashion MNIST

        Specs:
            - 10 classes (1 per type)
            - Each image is (1 x 28 x 28)

        Standard FashionMNIST, train, val, test splits and transforms

        Transforms::

            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor()
            ])

        Example::

            from pl_bolts.datamodules import FashionMNISTDataModule

            dm = FashionMNISTDataModule('.')
            model = LitModel()

            Trainer().fit(model, dm)

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                'You want to use fashion MNIST dataset loaded from `torchvision` which is not installed yet.'
            )

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def prepare_data(self):
        """
        Saves FashionMNIST files to data_dir
        """
        FashionMNIST(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor())
        FashionMNIST(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size=32, transforms=None):
        """
        FashionMNIST train set removes a subset to use for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.train_transforms or self._default_transforms()

        dataset = FashionMNIST(self.data_dir, train=True, download=False, transform=transforms)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):
        """
        FashionMNIST val set uses a subset of the training set for validation

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()

        dataset = FashionMNIST(self.data_dir, train=True, download=True, transform=transforms)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size=32, transforms=None):
        """
        FashionMNIST test set uses the test split

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.test_transforms or self._default_transforms()

        dataset = FashionMNIST(self.data_dir, train=False, download=False, transform=transforms)
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
