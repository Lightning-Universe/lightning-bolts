from warnings import warn

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

try:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import Cityscapes
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')
    _TORCHVISION_AVAILABLE = False
else:
    _TORCHVISION_AVAILABLE = True


class CityscapesDataModule(LightningDataModule):

    name = 'Cityscapes'
    extra_args = {}

    def __init__(
            self,
            data_dir,
            val_split=5000,
            num_workers=16,
            batch_size=32,
            seed=42,
            *args,
            **kwargs,
    ):
        """
        .. figure:: https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/muenster00-1024x510.png
            :width: 400
            :alt: Cityscape

        Standard Cityscapes, train, val, test splits and transforms

        Specs:
            - 30 classes (road, person, sidewalk, etc...)
            - (image, target) - image dims: (3 x 32 x 32), target dims: (3 x 32 x 32)

        Transforms::

            transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.28689554, 0.32513303, 0.28389177],
                    std=[0.18696375, 0.19017339, 0.18720214]
                )
            ])

        Example::

            from pl_bolts.datamodules import CityscapesDataModule

            dm = CityscapesDataModule(PATH)
            model = LitModel()

            Trainer().fit(model, dm)

        Or you can set your own transforms

        Example::

            dm.train_transforms = ...
            dm.test_transforms = ...
            dm.val_transforms  = ...

        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:
            raise ImportError(
                'You want to use CityScapes dataset loaded from `torchvision` which is not installed yet.'
            )

        self.dims = (3, 32, 32)
        self.DATASET = Cityscapes
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed

    @property
    def num_classes(self):
        """
        Return:
            30
        """
        return 30

    def prepare_data(self):
        """
        Saves Cityscapes files to data_dir
        """
        self.DATASET(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
        self.DATASET(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args)

    def train_dataloader(self):
        """
        Cityscapes train set with removed subset to use for validation
        """
        transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        dataset_train, _ = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self):
        """
        Cityscapes val set uses a subset of the training set for validation
        """
        transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
        train_length = len(dataset)
        _, dataset_val = random_split(
            dataset,
            [train_length - self.val_split, self.val_split],
            generator=torch.Generator().manual_seed(self.seed)
        )
        loader = DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader

    def test_dataloader(self):
        """
        Cityscapes test set uses the test split
        """
        transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms, **self.extra_args)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def default_transforms(self):
        cityscapes_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=[0.28689554, 0.32513303, 0.28389177],
                std=[0.18696375, 0.19017339, 0.18720214]
            )
        ])
        return cityscapes_transforms
