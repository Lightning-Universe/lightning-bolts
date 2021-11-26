import uuid
from pathlib import Path

import pytest
import torch
from PIL import Image

from pl_bolts.datamodules import (
    BinaryEMNISTDataModule,
    BinaryMNISTDataModule,
    CIFAR10DataModule,
    CityscapesDataModule,
    EMNISTDataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from pl_bolts.datamodules.sr_datamodule import TVTDataModule
from pl_bolts.datasets.cifar10_dataset import CIFAR10
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST


def test_dev_datasets(datadir):

    ds = CIFAR10(data_dir=datadir)
    for _ in ds:
        pass


def _create_synth_Cityscapes_dataset(path_dir):
    """Create synthetic dataset with random images, just to simulate that the dataset have been already
    downloaded."""
    non_existing_citites = ["dummy_city_1", "dummy_city_2"]
    fine_labels_dir = Path(path_dir) / "gtFine"
    images_dir = Path(path_dir) / "leftImg8bit"
    dataset_splits = ["train", "val", "test"]

    for split in dataset_splits:
        for city in non_existing_citites:
            (images_dir / split / city).mkdir(parents=True, exist_ok=True)
            (fine_labels_dir / split / city).mkdir(parents=True, exist_ok=True)
            base_name = str(uuid.uuid4())
            image_name = f"{base_name}_leftImg8bit.png"
            instance_target_name = f"{base_name}_gtFine_instanceIds.png"
            semantic_target_name = f"{base_name}_gtFine_labelIds.png"
            Image.new("RGB", (2048, 1024)).save(images_dir / split / city / image_name)
            Image.new("L", (2048, 1024)).save(fine_labels_dir / split / city / instance_target_name)
            Image.new("L", (2048, 1024)).save(fine_labels_dir / split / city / semantic_target_name)


def test_cityscapes_datamodule(datadir):

    _create_synth_Cityscapes_dataset(datadir)

    batch_size = 1
    target_types = ["semantic", "instance"]
    for target_type in target_types:
        dm = CityscapesDataModule(datadir, num_workers=0, batch_size=batch_size, target_type=target_type)
    loader = dm.train_dataloader()
    img, mask = next(iter(loader))
    assert img.size() == torch.Size([batch_size, 3, 1024, 2048])
    assert mask.size() == torch.Size([batch_size, 1024, 2048])

    loader = dm.val_dataloader()
    img, mask = next(iter(loader))
    assert img.size() == torch.Size([batch_size, 3, 1024, 2048])
    assert mask.size() == torch.Size([batch_size, 1024, 2048])

    loader = dm.test_dataloader()
    img, mask = next(iter(loader))
    assert img.size() == torch.Size([batch_size, 3, 1024, 2048])
    assert mask.size() == torch.Size([batch_size, 1024, 2048])


@pytest.mark.parametrize("val_split, train_len", [(0.2, 48_000), (5_000, 55_000)])
def test_vision_data_module(datadir, val_split, train_len):
    dm = _create_dm(MNISTDataModule, datadir, val_split=val_split)
    assert len(dm.dataset_train) == train_len


@pytest.mark.parametrize("dm_cls", [BinaryMNISTDataModule, CIFAR10DataModule, FashionMNISTDataModule, MNISTDataModule])
def test_data_modules(datadir, dm_cls):
    dm = _create_dm(dm_cls, datadir)
    loader = dm.train_dataloader()
    img, _ = next(iter(loader))
    assert img.size() == torch.Size([2, *dm.size()])


def _create_dm(dm_cls, datadir, **kwargs):
    dm = dm_cls(data_dir=datadir, num_workers=1, batch_size=2, **kwargs)
    dm.prepare_data()
    dm.setup()
    return dm


def test_sr_datamodule(datadir):
    dataset = SRMNIST(scale_factor=4, root=datadir, download=True)
    dm = TVTDataModule(dataset_train=dataset, dataset_val=dataset, dataset_test=dataset, batch_size=2)

    next(iter(dm.train_dataloader()))
    next(iter(dm.val_dataloader()))
    next(iter(dm.test_dataloader()))


@pytest.mark.parametrize("split", ["byclass", "bymerge", "balanced", "letters", "digits", "mnist"])
@pytest.mark.parametrize("dm_cls", [BinaryEMNISTDataModule, EMNISTDataModule])
def test_emnist_datamodules(datadir, dm_cls, split):
    """Test EMNIST datamodules download data and have the correct shape."""

    dm = _create_dm(dm_cls, datadir, split=split)
    loader = dm.train_dataloader()
    img, _ = next(iter(loader))
    assert img.size() == torch.Size([2, 1, 28, 28])


@pytest.mark.parametrize("dm_cls", [BinaryEMNISTDataModule, EMNISTDataModule])
def test_emnist_datamodules_with_invalid_split(datadir, dm_cls):
    """Test EMNIST datamodules raise an exception if the provided `split` doesn't exist."""

    with pytest.raises(ValueError, match="Unknown value"):
        dm_cls(data_dir=datadir, split="this_split_doesnt_exist")


@pytest.mark.parametrize("dm_cls", [BinaryEMNISTDataModule, EMNISTDataModule])
@pytest.mark.parametrize(
    "split, expected_val_split",
    [
        ("byclass", None),
        ("bymerge", None),
        ("balanced", 18_800),
        ("digits", 40_000),
        ("letters", 14_800),
        ("mnist", 10_000),
    ],
)
def test_emnist_datamodules_with_strict_val_split(datadir, dm_cls, split, expected_val_split):
    """Test EMNIST datamodules when strict_val_split is specified to use the validation set defined in the paper.

    Refer to https://arxiv.org/abs/1702.05373 for `expected_val_split` values.
    """

    if expected_val_split is None:
        with pytest.raises(ValueError, match="Invalid value"):
            dm = _create_dm(dm_cls, datadir, split=split, strict_val_split=True)

    else:
        dm = _create_dm(dm_cls, datadir, split=split, strict_val_split=True)
        assert dm.val_split == expected_val_split
        assert len(dm.dataset_val) == expected_val_split
