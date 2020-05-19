import os
from torchvision import transforms as transform_lib
from torch.utils.data import DataLoader
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders
from pl_bolts.models.self_supervised.amdim.ssl_datasets import UnlabeledImagenet


class SSLImagenetDataLoaders(BoltDataLoaders):

    def __init__(self,
                 data_dir,
                 num_workers=16):

        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers

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
                m = f"""
                no meta.bin present. Imagenet is no longer automatically downloaded by PyTorch.
                To get imagenet:
                1. download yourself from http://www.image-net.org/challenges/LSVRC/2012/downloads
                2. download the devkit (ILSVRC2012_devkit_t12.tar.gz)
                3. generate the meta.bin file using the devkit
                4. copy the meta.bin file into both train and val split folders
                
                To generate the meta.bin do the following:
                
                from pl_bolts.models.self_supervised.amdim.ssl_datasets import UnlabeledImagenet
                path = '/path/to/folder/with/ILSVRC2012_devkit_t12.tar.gz/'
                UnlabeledImagenet.generate_meta_bins(path)    
                """
                raise FileNotFoundError(m)

    def train_dataloader(self, batch_size, num_images_per_class=-1, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        data_dir = self._resolve_data_subfolder(self.data_dir, 'train')

        dataset = UnlabeledImagenet(data_dir,
                                    num_imgs_per_class=num_images_per_class,
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

    def val_dataloader(self, batch_size, num_images_per_class=50, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        data_dir = self._resolve_data_subfolder(self.data_dir, 'train')
        dataset = UnlabeledImagenet(data_dir,
                                    num_imgs_per_class_val_split=num_images_per_class,
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
            transforms = self._default_transforms()
        data_dir = self._resolve_data_subfolder(self.data_dir, 'val')

        dataset = UnlabeledImagenet(data_dir,
                                    num_imgs_per_class=num_images_per_class,
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
            self.normalize_transform()
        ])
        return mnist_transforms

    def normalize_transform(self):
        normalize = transform_lib.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
        return normalize
