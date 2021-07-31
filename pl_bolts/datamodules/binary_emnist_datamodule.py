from typing import Any, Optional, Union

from pl_bolts.datamodules.emnist_datamodule import EMNISTDataModule
from pl_bolts.datasets import BinaryEMNIST
from pl_bolts.utils import _TORCHVISION_AVAILABLE


class BinaryEMNISTDataModule(EMNISTDataModule):
    """
    .. figure:: https://user-images.githubusercontent.com/4632336/123210742-4d6b3380-d477-11eb-80da-3e9a74a18a07.png
        :width: 400
        :alt: EMNIST

    Please see :class:`~pl_bolts.datamodules.emnist_datamodule.EMNISTDataModule` for more details.

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
        split: str = 'mnist',
        val_split: Optional[Union[int, float]] = 0.2,
        num_workers: int = 16,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = False,
        pin_memory: bool = False,
        drop_last: bool = False,
        strict_val_split: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: Where to save/load the data.
            split: The dataset has 6 different splits: ``byclass``, ``bymerge``,
                ``balanced``, ``letters``, ``digits`` and ``mnist``.
                This argument is passed to :class:`torchvision.datasets.EMNIST`.
            val_split: Percent (float) or number (int) of samples
                to use for the validation split.
                If ``val_split = None``, the default size
                (as in the EMNIST documentation) of
                validation dataset is disbursed based on the ``split``.
            num_workers: How many workers to use for loading data
            normalize: If ``True``, applies image normalize.
            batch_size: How many samples per batch to load.
            seed: Random seed to be used for train/val/test splits.
            shuffle: If ``True``, shuffles the train data every epoch.
            pin_memory: If ``True``, the data loader will copy Tensors into
                CUDA pinned memory before returning them.
            drop_last: If ``True``, drops the last incomplete batch.
            strict_val_split: If ``True``, uses the validation split defined in `the paper
                <https://arxiv.org/abs/1702.05373>`_ and only works with ``balanced``, ``digits``, ``letters``,
                ``mnist`` splits.
        """
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                'You want to use EMNIST dataset loaded from `torchvision` which is not installed yet.'
            )

        super(BinaryEMNISTDataModule, self).__init__(  # type: ignore[misc]
            data_dir=data_dir,
            split=split,
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
