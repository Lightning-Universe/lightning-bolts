import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pl_bolts.utils import _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as transform_lib
    from torchvision.datasets import VOCDetection
else:  # pragma: no cover
    warn_missing_pkg("torchvision")


class Compose:
    """Like `torchvision.transforms.compose` but works for (image, target)"""

    def __init__(self, transforms: List[Callable], image_transforms: Optional[Callable] = None) -> None:
        self.transforms = transforms
        self.image_transforms = image_transforms

    def __call__(self, image: Any, target: Any) -> Tuple[Tensor, Tensor]:
        for t in self.transforms:
            image, target = t(image, target)
        if self.image_transforms:
            image = self.image_transforms(image)
        return image, target


def _collate_fn(batch: List[Tensor]) -> tuple:
    return tuple(zip(*batch))


CLASSES = (
    "__background__ ",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def _prepare_voc_instance(image: Any, target: Dict[str, Any]):
    """Prepares VOC dataset into appropriate target for fasterrcnn.

    https://github.com/pytorch/vision/issues/1097#issuecomment-508917489
    """
    anno = target["annotation"]
    boxes = []
    classes = []
    area = []
    iscrowd = []
    objects = anno["object"]
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        bbox = obj["bndbox"]
        bbox = [int(bbox[n]) - 1 for n in ["xmin", "ymin", "xmax", "ymax"]]
        boxes.append(bbox)
        classes.append(CLASSES.index(obj["name"]))
        iscrowd.append(int(obj["difficult"]))
        area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    classes = torch.as_tensor(classes)
    area = torch.as_tensor(area)
    iscrowd = torch.as_tensor(iscrowd)

    image_id = anno["filename"][5:-4]
    image_id = torch.as_tensor([int(image_id)])

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["image_id"] = image_id

    # for conversion to coco api
    target["area"] = area
    target["iscrowd"] = iscrowd

    return image, target


class VOCDetectionDataModule(LightningDataModule):
    """TODO(teddykoker) docstring."""

    name = "vocdetection"

    def __init__(
        self,
        data_dir: Optional[str] = None,
        year: str = "2012",
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 16,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use VOC dataset loaded from `torchvision` which is not installed yet."
            )

        super().__init__(*args, **kwargs)

        self.year = year
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.num_workers = num_workers
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        """
        Return:
            21
        """
        return 21

    def prepare_data(self) -> None:
        """Saves VOCDetection files to data_dir."""
        VOCDetection(self.data_dir, year=self.year, image_set="train", download=True)
        VOCDetection(self.data_dir, year=self.year, image_set="val", download=True)

    def train_dataloader(self, image_transforms: Optional[Callable] = None) -> DataLoader:
        """VOCDetection train set uses the `train` subset.

        Args:
            image_transforms: custom image-only transforms
        """
        transforms = [
            _prepare_voc_instance,
            self.default_transforms() if self.train_transforms is None else self.train_transforms,
        ]
        transforms = Compose(transforms, image_transforms)

        dataset = VOCDetection(self.data_dir, year=self.year, image_set="train", transforms=transforms)
        return self._data_loader(dataset, shuffle=self.shuffle)

    def val_dataloader(self, image_transforms: Optional[Callable] = None) -> DataLoader:
        """VOCDetection val set uses the `val` subset.

        Args:
            image_transforms: custom image-only transforms
        """
        transforms = [
            _prepare_voc_instance,
            self.default_transforms() if self.val_transforms is None else self.val_transforms,
        ]
        transforms = Compose(transforms, image_transforms)

        dataset = VOCDetection(self.data_dir, year=self.year, image_set="val", transforms=transforms)
        return self._data_loader(dataset, shuffle=False)

    def default_transforms(self) -> Callable:
        voc_transforms = [transform_lib.ToTensor()]
        if self.normalize:
            voc_transforms += [transform_lib.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        voc_transforms = transform_lib.Compose(voc_transforms)
        return lambda image, target: (voc_transforms(image), target)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            collate_fn=_collate_fn,
        )
