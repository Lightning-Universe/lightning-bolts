import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pl_bolts.datasets.stl10_sr_dataset import STL10_SR


class STL10_SR_DataModule(LightningDataModule):

    name = "stl10_sr"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        train_split: str = "train+unlabeled",
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.dataset_cls = STL10_SR
        self.dims = (3, 96, 96)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.train_split = train_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        self.dataset_cls(self.data_dir, split=self.train_split, download=True)
        self.dataset_cls(self.data_dir, split="test", download=True)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            self.dataset_train = self.dataset_cls(self.data_dir, split=self.train_split)

        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(self.data_dir, split="test")

    def train_dataloader(self):
        return self._dataloader(self.dataset_train, shuffle=self.shuffle)

    def test_dataloader(self):
        return self._dataloader(self.dataset_test, shuffle=False)

    def _dataloader(self, dataset: Dataset, shuffle: bool = True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
