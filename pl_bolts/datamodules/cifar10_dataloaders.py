from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from pl_bolts.datamodules.bolts_dataloaders_base import BoltDataLoaders


class CIFAR10DataLoaders(BoltDataLoaders):

    def __init__(self, save_path, val_split=5000, num_workers=16):
        super().__init__()
        self.save_path = save_path
        self.val_split = val_split
        self.num_workers = num_workers

    @property
    def train_length(self):
        return 50000

    def prepare_data(self):
        CIFAR10(self.save_path, train=True, download=True, transform=transform_lib.ToTensor())
        CIFAR10(self.save_path, train=False, download=True, transform=transform_lib.ToTensor())

    def train_dataloader(self, batch_size, transforms=None, add_normalize=True):
        if transforms is None:
            transforms = self.get_transforms()

        if add_normalize:
            self.add_default_normalize(transforms, train=True)

        dataset = CIFAR10(self.save_path, train=True, download=False, transform=transforms)
        dataset_train, _ = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        return loader

    def val_dataloader(self, batch_size,transforms=None, add_normalize=True):
        if transforms is None:
            transforms = self.get_transforms()

        if add_normalize:
            self.add_default_normalize(transforms)

        dataset = CIFAR10(self.save_path, train=True, download=True, transform=transforms)
        _, dataset_val = random_split(dataset, [self.train_length - self.val_split, self.val_split])
        loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def test_dataloader(self, batch_size, transforms=None, add_normalize=True):
        if transforms is None:
            transforms = self.get_transforms()

        if add_normalize:
            self.add_default_normalize(transforms)

        dataset = CIFAR10(self.save_path, train=False, download=False, transform=transforms)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

    def default_transforms(self):
        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])
        return mnist_transforms

    def add_default_normalize(self, user_transforms):
        normalize = transform_lib.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        user_transforms.transforms.append(normalize)
