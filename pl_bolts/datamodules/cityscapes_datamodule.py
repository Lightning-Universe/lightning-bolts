# type: ignore[override]
from typing import Any, Callable

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import Cityscapes
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class CityscapesDataModule(LightningDataModule):
    """
    .. figure:: https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/muenster00-1024x510.png
        :width: 400
        :alt: Cityscape

    Standard Cityscapes, train, val, test splits and transforms

    Note: You need to have downloaded the Cityscapes dataset first and provide the path to where it is saved.
        You can download the dataset here: https://www.cityscapes-dataset.com/

    Specs:
        - 30 classes (road, person, sidewalk, etc...)
        - (image, target) - image dims: (3 x 1024 x 2048), target dims: (1024 x 2048)

    Transforms::

        transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            transform_lib.Normalize(
                mean=[0.28689554, 0.32513303, 0.28389177],
                std=[0.18696375, 0.19017339, 0.18720214]
            )
        ])

    Example::

        from pl_bolts.datamodules import CityscapesDataModule

        dm = CityscapesDataModule(PATH)
        model = LitModel()

        Trainer().fit(model, datamodule=dm)

    Or you can set your own transforms

    Example::

        dm.train_transforms = ...
        dm.test_transforms = ...
        dm.val_transforms  = ...
        dm.target_transforms = ...
    """

    name = "Cityscapes"
    extra_args: dict = {}

    def __init__(
        self,
        data_dir: str,
        quality_mode: str = "fine",
        target_type: str = "instance",
        num_workers: int = 0,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            data_dir: where to load the data from path, i.e. where directory leftImg8bit and gtFine or gtCoarse
                are located
            quality_mode: the quality mode to use, either 'fine' or 'coarse'
            target_type: targets to use, either 'instance' or 'semantic'
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
            seed: random seed to be used for train/val/test splits
            shuffle: If true shuffles the data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """
        super().__init__(*args, **kwargs)

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use CityScapes dataset loaded from `torchvision` which is not installed yet."
            )

        if target_type not in ["instance", "semantic"]:
            raise ValueError(f'Only "semantic" and "instance" target types are supported. Got {target_type}.')

        self.dims = (3, 1024, 2048)
        self.data_dir = data_dir
        self.quality_mode = quality_mode
        self.target_type = target_type
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.target_transforms = None

    @property
    def num_classes(self) -> int:
        """
        Return:
            30
        """
        return 30

    def train_dataloader(self) -> DataLoader:
        """Cityscapes train set."""
        transforms = self.train_transforms or self._default_transforms()
        target_transforms = self.target_transforms or self._default_target_transforms()

        dataset = Cityscapes(
            self.data_dir,
            split="train",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
            **self.extra_args,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self) -> DataLoader:
        """Cityscapes val set."""
        transforms = self.val_transforms or self._default_transforms()
        target_transforms = self.target_transforms or self._default_target_transforms()

        dataset = Cityscapes(
            self.data_dir,
            split="val",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
            **self.extra_args,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        """Cityscapes test set."""
        transforms = self.test_transforms or self._default_transforms()
        target_transforms = self.target_transforms or self._default_target_transforms()

        dataset = Cityscapes(
            self.data_dir,
            split="test",
            target_type=self.target_type,
            mode=self.quality_mode,
            transform=transforms,
            target_transform=target_transforms,
            **self.extra_args,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        return loader

    def _default_transforms(self) -> Callable:
        cityscapes_transforms = transform_lib.Compose(
            [
                transform_lib.ToTensor(),
                transform_lib.Normalize(
                    mean=[0.28689554, 0.32513303, 0.28389177], std=[0.18696375, 0.19017339, 0.18720214]
                ),
            ]
        )
        return cityscapes_transforms

    def _default_target_transforms(self) -> Callable:
        cityscapes_target_transforms = transform_lib.Compose(
            [transform_lib.ToTensor(), transform_lib.Lambda(lambda t: t.squeeze())]
        )
        return cityscapes_target_transforms
