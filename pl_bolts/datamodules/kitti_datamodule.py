import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from pl_bolts.datamodules.kitti_dataset import KittiDataset

class KittiDataModule(LightningDataModule):

    name = 'kitti'

    def __init__(
            self,
            data_dir: str = '/Users/annikabrundyn/Documents/data_semantics',
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.default_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                 std=[0.32064945, 0.32098866, 0.32325324])
        ])
        self.trainset = KittiDataset(self.data_dir, split='train', transform=self.default_transforms)
        self.validset = KittiDataset(self.data_dir, split='valid', transform=self.default_transforms)

    def train_dataloader(self):
        loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        return

