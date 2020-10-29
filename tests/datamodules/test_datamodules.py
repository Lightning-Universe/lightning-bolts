import torch
import uuid
from PIL import Image
from pathlib import Path
from pl_bolts.datasets.cifar10_dataset import CIFAR10
from pl_bolts.datamodules import CityscapesDataModule


def test_dev_datasets(data_dir):

    ds = CIFAR10(data_dir=data_dir)
    for b in ds:
        pass


def _create_synth_Cityscapes_dataset(path_dir):
    """Create synthetic dataset with random images, just to simulate that the dataset have been already downloaded."""
    non_existing_citites = ['dummy_city_1', 'dummy_city_2']
    fine_labels_dir = Path(path_dir) / 'gtFine'
    images_dir = Path(path_dir) / 'leftImg8bit'
    dataset_splits = ['train', 'val', 'test']

    for split in dataset_splits:
        for city in non_existing_citites:
            (images_dir / split / city).mkdir(parents=True)
            (fine_labels_dir / split / city).mkdir(parents=True)
            base_name = str(uuid.uuid4())
            image_name = f'{base_name}_leftImg8bit.png'
            instance_target_name = f'{base_name}_gtFine_instanceIds.png'
            semantic_target_name = f'{base_name}_gtFine_labelIds.png'
            Image.new('RGB', (2048, 1024)).save(
                images_dir / split / city / image_name)
            Image.new('L', (2048, 1024)).save(
                fine_labels_dir / split / city / instance_target_name)
            Image.new('L', (2048, 1024)).save(
                fine_labels_dir / split / city / semantic_target_name)


def test_cityscapes_datamodule(tmpdir):

    _create_synth_Cityscapes_dataset(tmpdir)

    batch_size = 1
    target_types = ['semantic', 'instance']
    for target_type in target_types:
        dm = CityscapesDataModule(tmpdir,
                                  num_workers=0,
                                  batch_size=batch_size,
                                  target_type=target_type)
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
