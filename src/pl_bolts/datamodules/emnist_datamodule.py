from typing import Any, Callable, Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.transforms.dataset_normalizations import emnist_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import EMNIST
else:  # pragma: no cover
    warn_missing_pkg("torchvision")
    EMNIST = object


class EMNISTDataModule(VisionDataModule):
    """
    .. figure:: https://user-images.githubusercontent.com/4632336/123210742-4d6b3380-d477-11eb-80da-3e9a74a18a07.png
        :width: 400
        :alt: EMNIST

    .. list-table:: Dataset information (source: `EMNIST: an extension of MNIST to handwritten
        letters <https://arxiv.org/abs/1702.05373>`_ [Table-II])
       :header-rows: 1

       * - Split Name
         - No. classes
         - Train set size
         - Test set size
         - Validation set
         - Total size
       * - ``"byclass"``
         - 62
         - 697,932
         - 116,323
         - No
         - 814,255
       * - ``"byclass"``
         - 62
         - 697,932
         - 116,323
         - No
         - 814,255
       * - ``"bymerge"``
         - 47
         - 697,932
         - 116,323
         - No
         - 814,255
       * - ``"balanced"``
         - 47
         - 112,800
         - 18,800
         - Yes
         - 131,600
       * - ``"digits"``
         - 10
         - 240,000
         - 40,000
         - Yes
         - 280,000
       * - ``"letters"``
         - 37
         - 88,800
         - 14,800
         - Yes
         - 103,600
       * - ``"mnist"``
         - 10
         - 60,000
         - 10,000
         - Yes
         - 70,000

    |

    Args:
        data_dir: Root directory of dataset.
        split: The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``.
            This argument is passed to :class:`torchvision.datasets.EMNIST`.
        val_split: Percent (float) or number (int) of samples to use for the validation split.
        num_workers: How many workers to use for loading data
        normalize: If ``True``, applies image normalize.
        batch_size: How many samples per batch to load.
        seed: Random seed to be used for train/val/test splits.
        shuffle: If ``True``, shuffles the train data every epoch.
        pin_memory: If ``True``, the data loader will copy Tensors into
            CUDA pinned memory before returning them.
        drop_last: If ``True``, drops the last incomplete batch.
        strict_val_split: If ``True``, uses the validation split defined in the paper and ignores ``val_split``.
            Note that it only works with ``"balanced"``, ``"digits"``, ``"letters"``, ``"mnist"`` splits.

    Here is the default EMNIST, train, val, test-splits and transforms.

    Transforms::

        emnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
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

    _official_val_split = {
        "balanced": 18_800,
        "digits": 40_000,
        "letters": 14_800,
        "mnist": 10_000,
    }

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "mnist",
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        strict_val_split: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use MNIST dataset loaded from `torchvision` which is not installed yet."
            )

        if split not in self.dataset_cls.splits:
            raise ValueError(
                f"Unknown value '{split}' for argument `split`. Valid values are {self.dataset_cls.splits}."
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

        if strict_val_split:
            # replaces the value in `val_split` with the one defined in the paper
            if self.split in self._official_val_split:
                self.val_split = self._official_val_split[self.split]
            else:
                raise ValueError(
                    f"Invalid value '{self.split}' for argument `split` with `strict_val_split=True`. "
                    f"Valid values are {set(self._official_val_split)}."
                )

    @property
    def num_classes(self) -> int:
        """Returns the number of classes.

        See the table above.
        """
        return len(self.dataset_cls.classes_split_dict[self.split])

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to ``data_dir``."""
        self.dataset_cls(self.data_dir, split=self.split, train=True, download=True)
        self.dataset_cls(self.data_dir, split=self.split, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms

            dataset_train = self.dataset_cls(
                self.data_dir, split=self.split, train=True, transform=train_transforms, **self.EXTRA_ARGS
            )
            dataset_val = self.dataset_cls(
                self.data_dir, split=self.split, train=True, transform=val_transforms, **self.EXTRA_ARGS
            )

            # Split
            self.dataset_train = self._split_dataset(dataset_train)
            self.dataset_val = self._split_dataset(dataset_val, train=False)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            self.dataset_test = self.dataset_cls(
                self.data_dir, split=self.split, train=False, transform=test_transforms, **self.EXTRA_ARGS
            )

    def default_transforms(self) -> Callable:
        if self.normalize:
            emnist_transforms = transform_lib.Compose([transform_lib.ToTensor(), emnist_normalization(self.split)])
        else:
            emnist_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return emnist_transforms
