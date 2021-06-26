from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets import BinaryEMNIST
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no cover
    warn_missing_pkg('torchvision')


class BinaryEMNISTDataModule(VisionDataModule):
    """
    .. figure:: https://user-images.githubusercontent.com/4632336/123210742-4d6b3380-d477-11eb-80da-3e9a74a18a07.png
        :width: 400
        :alt: EMNIST
    Specs:
        - 6 splits: ``byclass``, ``bymerge``,
                    ``balanced``, ``letters``, 
                    ``digits`` and ``mnist``.
        - For each split:
          - 10 classes (1 per digit) 
          - Each image is (1 x 28 x 28)
    Binary EMNIST, train, val, test splits and transforms
    Transforms::
        emnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])
    Example::
        from pl_bolts.datamodules import BinaryEMNISTDataModule
        dm = BinaryEMNISTDataModule('.')
        model = LitModel()
        Trainer().fit(model, datamodule=dm)
    """

    name = "binary_emnist"
    dataset_cls = BinaryEMNIST
    dims = (1, 28, 28)

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = 'digits',
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
                "You want to use transforms loaded from `torchvision` which is not installed yet."
            )

        super().__init__(  # type: ignore[misc]
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
            10
        """
        return 10 # TODO: check and return correct num_classes


    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        def _prepare_splits(split: str):    
            self.dataset_cls(self.data_dir, split=split, 
                             train=True, download=True)
            self.dataset_cls(self.data_dir, split=split, 
                             train=False, download=True)
        
        # TODO: expose split = 'all' option to the api later
        # If you choose ``all`` for split, that will use **all** splits.
        if self.split == 'all':
            splits = self.dataset_cls.splits
            for split in splits:
                _prepare_splits(split)
        else:
            _prepare_splits(self.split)
       

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
