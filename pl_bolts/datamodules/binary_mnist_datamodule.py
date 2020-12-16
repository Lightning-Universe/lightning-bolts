from typing import Optional, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pl_bolts.datasets.mnist_dataset import BinaryMNIST
from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no-cover
    warn_missing_pkg('torchvision')


class BinaryMNISTDataModule(VisionDataModule):
    """
    .. figure:: https://miro.medium.com/max/744/1*AO2rIhzRYzFVQlFLx9DM9A.png
        :width: 400
        :alt: MNIST

    Specs:
        - 10 classes (1 per digit)
        - Each image is (1 x 28 x 28)

    Binary MNIST, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])

    Example::

        from pl_bolts.datamodules import BinaryMNISTDataModule

        dm = BinaryMNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, dm)
    """

    name = "binary_mnist"
    DATASET_CLS = BinaryMNIST
    DIMS = (1, 28, 28)

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
                "You want to use transforms loaded from `torchvision` which is not installed yet."
            )

        super().__init__(
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
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def default_transforms(self):
        if self.normalize:
            mnist_transforms = transform_lib.Compose(
                [transform_lib.ToTensor(), transform_lib.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transform_lib.Compose([transform_lib.ToTensor()])

        return mnist_transforms
