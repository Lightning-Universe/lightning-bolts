from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import EMNIST
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


class EMNISTDataModule(VisionDataModule):
    """
    .. figure:: https://user-images.githubusercontent.com/4632336/123210742-4d6b3380-d477-11eb-80da-3e9a74a18a07.png
        :width: 400
        :alt: EMNIST
    Specs:
        - 6 splits: ``byclass``, ``bymerge``,
                    ``balanced``, ``letters``,
                    ``digits`` and ``mnist``.

        - Table:

          | Split Name   | Classes | No. Training | No. Testing | Validation | Total   |
          |--------------|---------|--------------|-------------|------------|---------|
          | ``byclass``  | 62      | 697,932      | 116,323     | No         | 814,255 |
          | ``bymerge``  | 47      | 697,932      | 116,323     | No         | 814,255 |
          | ``balanced`` | 47      | 112,800      | 18,800      | Yes        | 131,600 |
          | ``digits``   | 10      | 240,000      | 40,000      | Yes        | 280,000 |
          | ``letters``  | 37      | 88,800       | 14,800      | Yes        | 103,600 |
          | ``mnist``    | 10      | 60,000       | 10,000      | Yes        | 70,000  |

          source: https://arxiv.org/pdf/1702.05373.pdf [Table-II]

        - For each split:
          - Each image is (1 x 28 x 28)
    Standard EMNIST, train, val, test-splits and transforms
    Transforms::
        emnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])
    Example::
        from pl_bolts.datamodules import EMNISTDataModule
        dm = EMNISTDataModule('.')
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """
    name = "emnist"
    dataset_cls = EMNIST
    dims = (1, 28, 28)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = 'mnist',
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir (string, optional): Where to save/load the data.
            split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
                            ``balanced``, ``letters``, ``digits`` and ``mnist``.
                            This argument specifies which one to use.
            val_split (int, float): Percent (float) or number (int) of samples
                                    to use for the validation split.
            num_workers (int): How many workers to use for loading data
            normalize (bool): If ``True``, applies image normalize.
            batch_size (int): How many samples per batch to load.
            seed (int): Random seed to be used for train/val/test splits.
            shuffle (bool): If ``True``, shuffles the train data every epoch.
            pin_memory (bool): If ``True``, the data loader will copy Tensors into
                               CUDA pinned memory before returning them.
            drop_last (bool): If ``True``, drops the last incomplete batch.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                'You want to use MNIST dataset loaded from `torchvision` which is not installed yet.'
            )

        super(EMNISTDataModule, self).__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )
        self.split = split

    @property
    def num_classes(self) -> int:
        """
        Return:
           - for ``byclass``: 62
           - for ``bymerge``: 47
           - for ``balanced``: 47
           - for ``digits``: 10
           - for ``letters``: 47
           - for ``mnist``: 10
        """
        # The _metadata is only added to EMNIST dataset
        # to get split-specific metadata.
        nc = (self.dataset_cls
                  ._metadata.get('splits')
                  .get(self.split)
                  .get('num_classes'))
        return nc

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        def _prepare_with_splits(split: str):
            self.dataset_cls(self.data_dir, split=split,
                             train=True, download=True)
            self.dataset_cls(self.data_dir, split=split,
                             train=False, download=True)
        
        _prepare_with_splits(self.split)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """
        # TODO: change type: Any to something like torch
        def _setup_with_splits(split: str, train: bool, transform: Any):  # type: ignore[misc]
            return self.dataset_cls(
                self.data_dir,
                split=split,
                train=train,
                transform=transform,
                **self.EXTRA_ARGS
            )
        
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
            
            dataset_train = _setup_with_splits(split=self.split, train=True, transform=train_transforms) 
            
            dataset_val = _setup_with_splits(split=self.split, train=True, transform=val_transforms)

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = _setup_with_splits(
                split=self.split, train=False, transform=test_transforms)            

    def default_transforms(self) -> Callable:
        if self.normalize:
            emnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(), 
                transform_lib.Normalize(mean=(0.5, ), std=(0.5, )),
                # TODO: check that EMNIST also uses mean=0.5 and std=0.5
            ])
        else:
            emnist_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return emnist_transforms
