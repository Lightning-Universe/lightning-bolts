import os
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib

from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataModule
from pl_bolts.datamodules.imagenet_dataset import UnlabeledImagenet
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization


class ImagenetDataModule(BoltDataModule):

    def __init__(self,
                 data_dir: str,
                 meta_root: str = None,
                 num_imgs_per_val_class: int = 50,
                 image_size: int = 224,
                 num_workers: int = 16):
        """
        Imagenet train, val and test dataloaders.

        The train set is the imagenet train.

        The val set is taken from the train set with `num_imgs_per_val_class` images per class.
        For example if `num_imgs_per_val_class=2` then there will be 2,000 images in the validation set.

        The test set is the official imagenet validation set.

        The images are normalized using: Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

         Example::

            from pl_bolts.datamodules import ImagenetDataModule

            datamodule = ImagenetDataModule(IMAGENET_PATH)
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            test_loader = datamodule.test_dataloader()

        Args:

            data_dir: path to the imagenet dataset file
            meta_root: path to meta.bin file
            num_imgs_per_val_class: how many images per class for the validation set
            image_size: final image size
            num_workers: how many data workers
        """

        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_root = meta_root
        self.num_imgs_per_val_class = num_imgs_per_val_class
        self.image_size = image_size

    @property
    def num_classes(self):
        return 1000

    def size(self):
        return 3, self.image_size, self.image_size

    def _verify_splits(self, data_dir, split):
        dirs = os.listdir(data_dir)

        if split not in dirs:
            raise FileNotFoundError(f'a {split} Imagenet split was not found in {data_dir}, make sure the'
                                    f'folder contains a subfolder named {split}')

    def prepare_data(self):
        # imagenet cannot be downloaded... must provide path to folder with the train/val splits
        self._verify_splits(self.data_dir, 'train')
        self._verify_splits(self.data_dir, 'val')

        for split in ['train', 'val']:
            files = os.listdir(os.path.join(self.data_dir, split))
            if 'meta.bin' not in files:
                raise FileNotFoundError("""
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
                """)

    def train_dataloader(self, batch_size, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self.train_transform()

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class=-1,
                                    meta_root=self.meta_root,
                                    split='train',
                                    transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def val_dataloader(self, batch_size, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self.val_transform()

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class_val_split=self.num_imgs_per_val_class,
                                    meta_root=self.meta_root,
                                    split='val',
                                    transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return loader

    def test_dataloader(self, batch_size, num_images_per_class, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self.val_transform()

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class=-1,
                                    meta_root=self.meta_root,
                                    split='test',
                                    transform=transforms)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def train_transform(self):
        """
        The standard imagenet transforms

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        """
        preprocessing = transform_lib.Compose([
            transform_lib.RandomResizedCrop(self.image_size),
            transform_lib.RandomHorizontalFlip(),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])

        return preprocessing

    def val_transform(self):
        """
        The standard imagenet transforms for validation

        .. code-block:: python

            transform_lib.Compose([
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        """

        preprocessing = transform_lib.Compose([
            transform_lib.Resize(self.image_size + 32),
            transform_lib.CenterCrop(self.image_size),
            transform_lib.ToTensor(),
            imagenet_normalization(),
        ])
        return preprocessing
