import os
from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from pl_bolts.datasets import KittiDataset
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.stability import under_review
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


@under_review()
class KittiDataModule(LightningDataModule):

    name = "kitti"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: float = 0.2,
        test_split: float = 0.1,
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Kitti train, validation and test dataloaders.

        Note:
            You need to have downloaded the Kitti dataset first and provide the path to where it is saved.
            You can download the dataset here:
            http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015

        Specs:
            - 200 samples
            - Each image is (3 x 1242 x 376)

        In total there are 34 classes but some of these are not useful so by default we use only 19 of the classes
        specified by the `valid_labels` parameter.

        Example::

            from pl_bolts.datamodules import KittiDataModule

            dm = KittiDataModule(PATH)
            model = LitModel()

            Trainer().fit(model, datamodule=dm)

        Args:
            data_dir: where to load the data from path, i.e. '/path/to/folder/with/data_semantics/'
            val_split: size of validation test (default 0.2)
            test_split: size of test set (default 0.1)
            num_workers: how many workers to use for loading data
            batch_size: the batch size
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use `torchvision` which is not installed yet.")

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # split into train, val, test
        kitti_dataset = KittiDataset(self.data_dir, transform=self._default_transforms())

        val_len = round(val_split * len(kitti_dataset))
        test_len = round(test_split * len(kitti_dataset))
        train_len = len(kitti_dataset) - val_len - test_len

        self.trainset, self.valset, self.testset = random_split(
            kitti_dataset, lengths=[train_len, val_len, test_len], generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        kitti_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.35675976, 0.37380189, 0.3764753], std=[0.32064945, 0.32098866, 0.32325324]
                ),
            ]
        )
        return kitti_transforms
