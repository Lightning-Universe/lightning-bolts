import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision import transforms as transform_lib


class VOCDetectionDataModule(LightningDataModule):
    name = "vocdetection"

    def __init__(
        self,
        data_dir: str,
        year: str = "2012",
        num_workers: int = 16,
        normalize: bool = False,
        *args,
        **kwargs,
    ):
        """
        TODO(teddykoker) docstring
        """

        super().__init__(*args, **kwargs)
        self.year = year
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.normalize = normalize

    @property
    def num_classes(self):
        """
        Return:
            20
        """
        return 20

    def prepare_data(self):
        """
        Saves VOCDetection files to data_dir
        """
        VOCDetection(self.data_dir, year=self.year, image_set="train", download=True)
        VOCDetection(self.data_dir, year=self.year, image_set="val", download=True)

    def train_dataloader(self, batch_size=8, transforms=None):
        """
        VOCDetection train set uses the `train` subset

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.train_transforms or self._default_transforms()
        dataset = VOCDetection(
            self.data_dir, year=self.year, image_set="train", transform=transforms
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=8, transforms=None):
        """
        VOCDetection val set uses the `val` subset

        Args:
            batch_size: size of batch
            transforms: custom transforms
        """
        transforms = transforms or self.val_transforms or self._default_transforms()
        dataset = VOCDetection(
            self.data_dir, year=self.year, image_set="trainval", transform=transforms
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader

    def _default_transforms(self):
        if self.normalize:
            transforms = transform_lib.Compose(
                [
                    transform_lib.ToTensor(),
                    transform_lib.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            transforms = transform_lib.ToTensor()
        return transforms


dm = VOCDetectionDataModule("tmp")
dm.prepare_data()
print(next(iter(dm.train_dataloader())))
