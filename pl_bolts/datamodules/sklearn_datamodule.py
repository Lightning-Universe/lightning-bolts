import warnings
from typing import Any, Optional, Union

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import exceptions
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset

from pl_bolts.datasets import ArrayDataset


class SklearnDataModule(LightningDataModule):
    """Split arrays into train, val, and test dataloaders with `scikit-learn`.

    Args:
        data: Feature variables.
        target: Target variables.
        test_dataset: Test `Array Dataset`. If None, training and testing samples are created with `test_size`.
            Default is None.
        val_size: Size of validation sample created from training sample. If float, should be between 0.0 and 1.0
            and represent the proportion of the dataset to include in the validation split. If int, represents the
            absolute number of validation samples.
        test_size: Size of test sample. Splits data and target into training and testing samples. If float, should
            be between 0.0 and 1.0  and represent the proportion of the dataset to include in the testing split.
                If int, represents the absolute number of testing samples. Default is None.
        random_state: Controls the shuffling applied to the data before applying the split. Default is `None`
        shuffle: Whether to shuffle the data before splitting. If shuffle=False then stratify must be None.
        stratify: If not None, data is split in a stratified fashion, using this as the class labels.
        num_workers: Number of subprocesses to use for data loading. 0 means that the data will be loaded in the
            main process. Number of CPUs available.
        batch_size: Batch size to use for each dataloader. Default is 1.
        pin_memory: If `True`, the data loader will copy Tensors into device/CUDA pinned memory before returning
            them.
        drop_last: set to `True` to drop the last incomplete batch, if the dataset size is not divisible by the
            batch size. If `False` and the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. Default is `False`.
        persistent_workers: If `True`, the data loader will not shutdown the worker processes after a dataset has
            been consumed once. This allows to maintain the workers `Dataset` instances alive. Default is `False`.

    Raises:
        MisconfigurationException: On initialisation, `test_dataset` and `test_size` cannot both be `None`. If a
            `test_dataset` is provided, it will be used in to create a `test_dataloader`. If a `test_size` is
            provided then a test `ArrayDataset` will be created from the split of `x` and `y`.
        MisconfigurationException: On initialisation, if the sum of `val_size` and `test_size` are greater or equal
            to 1.

    Warnings:
        `test_dataset` will be used to create the  `test_dataloader` if arguments are provided to `test_dataset` and
            `test_size`.
        Training sample will be smaller or equal to the addition of the validation and testing samples if the sum of
            `val_size` and `test_size` is greater or equal to 0.5.
    """

    def __init__(
        self,
        data: Any,
        target: Any,
        test_dataset: Optional[ArrayDataset] = None,
        val_size: Union[float, int] = 0.2,
        test_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: bool = False,
        num_workers: int = 0,
        batch_size: int = 1,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()

        if test_size is None and test_dataset is None:
            raise exceptions.MisconfigurationException("test_dataset is not provided, a value for test_size is needed.")

        if test_dataset is not None and not isinstance(test_dataset, ArrayDataset):
            raise exceptions.MisconfigurationException("Not a valid type for `test_dataset`.")

        if test_dataset and test_size:
            raise warnings.warn(
                "Arguments were provided for both `test_dataset` and test_size. The test_dataset will be used."
            )

        if (val_size + test_size) >= 1.0:
            raise exceptions.MisconfigurationException("The sum of val_size and test_size is too large.")
        elif (val_size + test_size) >= 0.5:
            warnings.warn("We strongly recommend you decrease val_size or test_size for more training data.")

        self.data = data
        self.target = target
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._test_dataset = test_dataset
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers and num_workers > 0

        if isinstance(self._test_dataset, ArrayDataset):
            self.x_train, self.x_val, self.y_train, self.y_val = self._sklearn_train_test_split(
                self.data, self.target, split=self.val_size
            )
        else:
            _x, self.x_test, _y, self.y_test = self._sklearn_train_test_split(
                self.data, self.target, split=self.test_size
            )
            self.x_train, self.x_val, self.y_train, self.y_val = self._sklearn_train_test_split(
                _x, _y, split=self.val_size
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ArrayDataset(self.x_train, self.y_train)

        if stage in ("fit", "validate") or stage is None:
            self.val_dataset = ArrayDataset(self.x_val, self.y_val)

        if stage == "test" or stage is None:
            self.test_dataset = (
                ArrayDataset(self.x_test, self.y_test)
                if self.x_test is not None and self.y_test is not None
                else self._test_dataset
            )

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.test_dataset)

    def _sklearn_train_test_split(self, x, y, split: Optional[Union[float, int]] = None):
        """Split arrays with  `scikit-learn` `train_test_split`.

        Args:
            x: x data.
            y: y_data.
            split: split size.

        Returns:
            list, length=2 * len(arrays). List containing train-test split of inputs.
        """
        return model_selection.train_test_split(
            x,
            y,
            shuffle=self.shuffle,
            test_size=split,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create dataloader.

        Args:
            dataset: `torch` Dataset.
            shuffle: Whether to shuffle the data.

        Returns:
            Dataloader.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
