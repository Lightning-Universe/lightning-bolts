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
from pl_bolts.datasets.cifar10_dataset import CIFAR10
from pl_bolts.datasets.emnist_dataset import EMNIST, EMNIST_METADATA


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


@pytest.mark.parametrize("split", EMNIST.splits)
@pytest.mark.parametrize("dm_cls", [BinaryEMNISTDataModule, EMNISTDataModule])
def test_emnist_datamodules(datadir, dm_cls, split):
    dm = _create_dm_emnistlike(dm_cls, datadir, split)
    loader = dm.train_dataloader()
    img, _ = next(iter(loader))
    assert img.size() == torch.Size([2, *dm.size()])


@pytest.mark.parametrize("val_split", [None, 0, 0., 0.2, 10_000])
@pytest.mark.parametrize("split", EMNIST.splits)
@pytest.mark.parametrize("dm_cls", [BinaryEMNISTDataModule, EMNISTDataModule])
def test_emnist_datamodules_val_split(dm_cls, datadir, split, val_split):
    dm = _create_dm_emnistlike(dm_cls, datadir, split, val_split)
    assert dm.dataset_cls._metadata == EMNIST_METADATA, \
        "ERROR!!!... `EMNIST_METADATA` mismatch detected!"
    assert dm.split_metadata == EMNIST_METADATA.get('splits').get(split), \
        "ERROR!!!... `split_metadata` mismatch detected."
    if val_split is None:
        if dm.split_metadata.get('validation'):
            assert dm.val_split == dm.split_metadata.get('num_test'), \
                "ERROR!!!... `val_split` was NOT mapped to default " + \
                f"'num_test' value: {dm.split_metadata.get('num_test')}"
        else:
            assert dm.val_split == dm._DEFAULT_NO_VALIDATION_VAL_SPLIT, \
                f"ERROR!!!... expected val_split = {dm._DEFAULT_NO_VALIDATION_VAL_SPLIT}, " + \
                f"assigned val_split = {dm.val_split}"
    else:
        if isinstance(val_split, (int, float)):
            assert dm.val_split == val_split, \
                f"ERROR!!!... `val_split` = {val_split} was NOT assigned."
        else:
            raise TypeError(
                'For `val_split`, ACCEPTED dtypes: `int`, `float`. ' +
                f'RECEIVED dtype: {type(val_split)}'
            )


def _create_dm_emnistlike(dm_cls, datadir, split='digits', val_split=0.2):
    dm = dm_cls(data_dir=datadir, split=split, val_split=val_split, num_workers=1, batch_size=2)
    dm.prepare_data()
    dm.setup()
    return dm
