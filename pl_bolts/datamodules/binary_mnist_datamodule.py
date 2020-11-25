from typing import Optional, Union

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.datamodules.base_datamodule import BaseDataModule
from pl_bolts.datasets.mnist_dataset import BinaryMNIST
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
else:  # pragma: no-cover
    warn_missing_pkg('torchvision')


class BinaryMNISTDataModule(BaseDataModule):
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

    def __init__(
        self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        seed: int = 42,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        *args,
        **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: size of batch
        """

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                "You want to use transforms loaded from `torchvision` which is not installed yet."
            )

        dataset_cls = BinaryMNIST
        dims = (1, 28, 28)

        super().__init__(
            dataset_cls=dataset_cls,
            dims=dims,
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            seed=seed,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            *args,
            **kwargs,
        )

    @property
    def num_classes(self):
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
            mnist_transforms = transform_lib.ToTensor()

        return mnist_transforms
