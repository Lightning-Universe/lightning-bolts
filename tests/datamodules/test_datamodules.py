import uuid
from pathlib import Path

import pytest
import torch
from PIL import Image

from pl_bolts.datamodules import (
    BinaryMNISTDataModule,
    CIFAR10DataModule,
    CityscapesDataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from pl_bolts.datasets.cifar10_dataset import CIFAR10


def test_dev_datasets(datadir):

    ds = CIFAR10(data_dir=datadir)
    for _ in ds:
        pass


def _create_synth_Cityscapes_dataset(path_dir):
    """Create synthetic dataset with random images, just to simulate that the dataset have been already downloaded."""
    non_existing_citites = ['dummy_city_1', 'dummy_city_2']
    fine_labels_dir = Path(path_dir) / 'gtFine'
    images_dir = Path(path_dir) / 'leftImg8bit'
    dataset_splits = ['train', 'val', 'test']

    for split in dataset_splits:
        for city in non_existing_citites:
            (images_dir / split / city).mkdir(parents=True, exist_ok=True)
            (fine_labels_dir / split / city).mkdir(parents=True, exist_ok=True)
            base_name = str(uuid.uuid4())
            image_name = f'{base_name}_leftImg8bit.png'
            instance_target_name = f'{base_name}_gtFine_instanceIds.png'
            semantic_target_name = f'{base_name}_gtFine_labelIds.png'
            Image.new('RGB', (2048, 1024)).save(images_dir / split / city / image_name)
            Image.new('L', (2048, 1024)).save(fine_labels_dir / split / city / instance_target_name)
            Image.new('L', (2048, 1024)).save(fine_labels_dir / split / city / semantic_target_name)


def test_cityscapes_datamodule(datadir):

    _create_synth_Cityscapes_dataset(datadir)

    batch_size = 1
    target_types = ['semantic', 'instance']
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


def _create_dm(dm_cls, datadir, val_split=0.2):
    dm = dm_cls(data_dir=datadir, val_split=val_split, num_workers=1, batch_size=2)
    dm.prepare_data()
    dm.setup()
    return dm
