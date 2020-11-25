from typing import Optional, Union

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.datamodules.base_datamodule import BaseDataModule
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import FashionMNIST
else:
    warn_missing_pkg('torchvision')  # pragma: no-cover


class FashionMNISTDataModule(BaseDataModule):
    """
    .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/
        wp-content/uploads/2019/02/Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset.png
        :width: 400
        :alt: Fashion MNIST

    Specs:
        - 10 classes (1 per type)
        - Each image is (1 x 28 x 28)

    Standard FashionMNIST, train, val, test splits and transforms

    Transforms::

        mnist_transforms = transform_lib.Compose([
            transform_lib.ToTensor()
        ])

    Example::

        from pl_bolts.datamodules import FashionMNISTDataModule

        dm = FashionMNISTDataModule('.')
        model = LitModel()

        Trainer().fit(model, dm)
    """

    name = "fashion_mnist"

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
            batch_size: size of batch
        """

        if not _TORCHVISION_AVAILABLE:
            raise ModuleNotFoundError(  # pragma: no-cover
                "You want to use FashionMNIST dataset loaded from `torchvision` which is not installed yet."
            )

        dataset_cls = FashionMNIST
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
