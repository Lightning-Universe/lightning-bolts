from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class TVTDataModule(LightningDataModule):
    """Simple DataModule creating train, val, and test dataloaders from given train, val, and test dataset.

    Example::
        from pl_bolts.datamodules import TVTDataModule
        from pl_bolts.datasets.sr_mnist_dataset import SRMNIST

        dataset_dev = SRMNIST(scale_factor=4, root=".", train=True)
        dataset_train, dataset_val = random_split(dataset_dev, lengths=[55_000, 5_000])
        dataset_test = SRMNIST(scale_factor=4, root=".", train=True)
        dm = TVTDataModule(dataset_train, dataset_val, dataset_test)
    """

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
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            dataset_train: Train dataset
            dataset_val: Val dataset
            dataset_test: Test dataset
            batch_size: How many samples per batch to load
            num_workers: How many workers to use for loading data
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                returning them
            drop_last: If true drops the last incomplete batch
        """
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
