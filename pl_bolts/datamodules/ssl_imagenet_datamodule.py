import os

from torch.utils.data import DataLoader
from torchvision import transforms as transform_lib

from pl_bolts.datamodules.imagenet_dataset import UnlabeledImagenet
from pl_bolts.datamodules.lightning_datamodule import LightningDataModule
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization


class SSLImagenetDataModule(LightningDataModule):  # pragma: no cover

    name = 'imagenet'

    def __init__(
            self,
            data_dir,
            meta_dir=None,
            num_workers=16,
            *args,
            **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.meta_dir = meta_dir

    @property
    def num_classes(self):
        return 1000

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

    def train_dataloader(self, batch_size, num_images_per_class=-1, add_normalize=False):
        transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class=num_images_per_class,
                                    meta_dir=self.meta_dir,
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

    def val_dataloader(self, batch_size, num_images_per_class=50, add_normalize=False):
        transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class_val_split=num_images_per_class,
                                    meta_dir=self.meta_dir,
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

    def test_dataloader(self, batch_size, num_images_per_class, add_normalize=False):
        transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms

        dataset = UnlabeledImagenet(self.data_dir,
                                    num_imgs_per_class=num_images_per_class,
                                    meta_dir=self.meta_dir,
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

    def _default_transforms(self):
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            imagenet_normalization()
        ])
        return mnist_transforms
