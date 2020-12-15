from typing import Optional, Sequence, Union

from pl_bolts.datamodules.base_datamodule import BaseDataModule
from pl_bolts.datasets.cifar10_dataset import TrialCIFAR10
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import CIFAR10
else:
    warn_missing_pkg('torchvision')  # pragma: no-cover


class CIFAR10DataModule(BaseDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
        Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
        :width: 400
        :alt: CIFAR-10

    Specs:
        - 10 classes (1 per class)
        - Each image is (3 x 32 x 32)

    Standard CIFAR10, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])

    Example::

        from pl_bolts.datamodules import CIFAR10DataModule

        dm = CIFAR10DataModule(PATH)
        model = LitModel()

        Trainer().fit(model, dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
    """

    name = "cifar10"
    extra_args = {}

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            seed: Seed to fix the validation split
            batch_size: How many samples per batch to load
        """

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                "You want to use CIFAR10 dataset loaded from `torchvision` which is not installed yet."
            )

        dataset_cls = CIFAR10
        dims = (3, 32, 32)

        super().__init__(
            dataset_cls=dataset_cls,
            dims=dims,
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            seed=seed,
            batch_size=batch_size,
            *args,
            **kwargs,
        )

    @property
    def num_samples(self) -> int:
        len_dataset = 50_000
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len

        return train_len

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def default_transforms(self) -> transform_lib.Compose:
        if self.normalize:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor(), cifar10_normalization()])
        else:
            cf10_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return cf10_transforms


class TinyCIFAR10DataModule(CIFAR10DataModule):
    """
    Standard CIFAR10, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        ])

    Example::

        from pl_bolts.datamodules import CIFAR10DataModule

        dm = CIFAR10DataModule(PATH)
        model = LitModel(datamodule=dm)
    """

    def __init__(
        self,
        data_dir: str,
        val_split: int = 50,
        num_workers: int = 16,
        num_samples: int = 100,
        labels: Optional[Sequence] = (1, 5, 8),
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            num_samples: number of examples per selected class/label
            labels: list selected CIFAR10 classes/labels
        """
        super().__init__(data_dir, val_split, num_workers, *args, **kwargs)
        self.dims = (3, 32, 32)
        self.DATASET = TrialCIFAR10
        self.num_samples = num_samples
        self.labels = sorted(labels) if labels is not None else set(range(10))
        self.extra_args = dict(num_samples=self.num_samples, labels=self.labels)

    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.labels)
