from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SRDataModule(LightningDataModule):
    # TODO: add docs
    def __init__(
        self,
        dataset_train: Dataset,
        dataset_val: Dataset,
        dataset_test: Dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.dataset_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.dataset_test, shuffle=False)

    def _dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
