import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

class KittiDataModule(LightningDataModule):

    name = 'kitti'

    def __init__(
            self,
            data_dir: str = None,
            num_workers: int = 16,
            batch_size: int = 32,
            seed: int = 42,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        def train_dataloader(self):
            return

        def val_dataloader(self):
            return

        def test_dataloader(self):
            return


