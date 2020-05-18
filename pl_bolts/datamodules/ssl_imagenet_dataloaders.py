from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders
from pl_bolts.models.self_supervised.amdim.ssl_datasets import UnlabeledImagenet


class SSLImagenetDataLoaders(BoltDataLoaders):

    def __init__(self,
                 save_path,
                 nb_imgs_per_train_class=-1,
                 num_workers=16):

        super().__init__()
        self.save_path = save_path
        self.val_split = val_split
        self.num_workers = num_workers
        self.nb_imgs_per_train_class = nb_imgs_per_train_class
        self.nb_imgs_per_val_class = nb_imgs_per_val_class

    @property
    def num_classes(self):
        return 1000

    def prepare_data(self):
        UnlabeledImagenet(self.save_path, split='train', download=True, transform=transform_lib.ToTensor())
        UnlabeledImagenet(self.save_path, split='test', download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, num_images_per_class=-1, transforms=None, add_normalize=False):
        if transforms is None:
            transforms = self._default_transforms()

        dataset = UnlabeledImagenet(self.save_path,
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

        dataset = UnlabeledImagenet(self.save_path,
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

        dataset = UnlabeledImagenet(self.save_path,
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
        normalize = transform_lib.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
        return normalize
