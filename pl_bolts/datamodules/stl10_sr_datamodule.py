import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from pl_bolts.datasets.stl10_sr_dataset import STL10_SR


class STL10_SR_DataModule(LightningDataModule):

    name = 'stl10_sr'

    # TODO: update with shuffle, etc.
    def __init__(
            self,
            data_dir: Optional[str] = None,
            num_workers: int = 12,
            batch_size: int = 32,
            *args,
            **kwargs,
    ) -> None:
        super().__init__()

        self.dataset_cls = STL10_SR
        self.dims = (3, 96, 96)
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.batch_size = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        # TODO self.dataset_cls(self.data_dir, split='train+unlabeled', download=True)
        self.dataset_cls(self.data_dir, split='train', download=True)
        self.dataset_cls(self.data_dir, split='test', download=True)

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            # TODO self.dataset_train = self.dataset_cls(self.data_dir, split='train+unlabeled')
            self.dataset_train = self.dataset_cls(self.data_dir, split='train')

        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(self.data_dir, split='test')

    def train_dataloader(self):
        return self._dataloader(self.dataset_train)

    def test_dataloader(self):
        return self._dataloader(self.dataset_test, shuffle=False)

    def _dataloader(self, dataset: Dataset, shuffle: bool = True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
