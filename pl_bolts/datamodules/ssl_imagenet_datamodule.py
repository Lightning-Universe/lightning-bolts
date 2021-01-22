# type: ignore[override]
import os
from typing import Any, Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


class SSLImagenetDataModule(LightningDataModule):  # pragma: no cover

    name = 'imagenet'

    def __init__(
        self,
        data_dir: str,
        meta_dir: Optional[str] = None,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                'You want to use ImageNet dataset loaded from `torchvision` which is not installed yet.'
            )

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 1000

    def _verify_splits(self, data_dir: str, split: str) -> None:
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(
                f'a {split} Imagenet split was not found in {data_dir}, make sure the'
                f' folder contains a subfolder named {split}'
            )

    def prepare_data(self) -> None:
        # imagenet cannot be downloaded... must provide path to folder with the train/val splits
        self._verify_splits(self.data_dir, 'train')
        self._verify_splits(self.data_dir, 'val')

        for split in ['train', 'val']:
            files = os.listdir(os.path.join(self.data_dir, split))
            if 'meta.bin' not in files:
                raise FileNotFoundError(
                    """
                no meta.bin present. Imagenet is no longer automatically downloaded by PyTorch.
                To get imagenet:
                1. download yourself from http://www.image-net.org/challenges/LSVRC/2012/downloads
                2. download the devkit (ILSVRC2012_devkit_t12.tar.gz)
                3. generate the meta.bin file using the devkit
                4. copy the meta.bin file into both train and val split folders

                To generate the meta.bin do the following:

                from pl_bolts.datamodules.imagenet_dataset import UnlabeledImagenet
                path = '/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/'
                UnlabeledImagenet.generate_meta_bins(path)
                """
                )

    def train_dataloader(self, num_images_per_class: int = -1, add_normalize: bool = False) -> DataLoader:
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class=num_images_per_class,
            meta_dir=self.meta_dir,
            split='train',
            transform=transforms
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def val_dataloader(self, num_images_per_class: int = 50, add_normalize: bool = False) -> DataLoader:
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class_val_split=num_images_per_class,
            meta_dir=self.meta_dir,
            split='val',
            transform=transforms
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def test_dataloader(self, num_images_per_class: int, add_normalize: bool = False) -> DataLoader:
        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = UnlabeledImagenet(
            self.data_dir,
            num_imgs_per_class=num_images_per_class,
            meta_dir=self.meta_dir,
            split='test',
            transform=transforms
        )
        loader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )
        return loader

    def _default_transforms(self) -> Callable:
        mnist_transforms = transform_lib.Compose([transform_lib.ToTensor(), imagenet_normalization()])
        return mnist_transforms
