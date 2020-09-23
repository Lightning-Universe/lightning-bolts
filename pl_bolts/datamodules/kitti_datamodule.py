import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class KittiDataModule(LightningDataModule):

    name = 'kitti'

    def __init__(
            self,
            data_dir: str = None,
            val_split: float = 0.2,
            test_split: float = 0.1,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.trainset = KittiDataset(self.data_path, split='train', transform=self.transform)
        self.validset = KittiDataset(self.data_path, split='valid', transform=self.transform)

        def train_dataloader(self):
            loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
            return loader

        def val_dataloader(self):
            loader = DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)
            return loader

        def test_dataloader(self):
            return

        def default_transforms(self):
             kitti_transforms = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.35675976, 0.37380189, 0.3764753],
                                      std=[0.32064945, 0.32098866, 0.32325324])
            ])
             return kitti_transforms


