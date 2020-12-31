import pytest
import torch
from torch.utils.data import DataLoader

from pl_bolts.datasets import DummyDataset, RandomDataset, RandomDictDataset, RandomDictStringDataset
from pl_bolts.datasets.mnist_dataset import SRMNISTDataset


def test_dummy_ds():
    ds = DummyDataset((1, 2), num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_ds():
    ds = RandomDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_dict_ds():
    ds = RandomDictDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


def test_rand_str_dict_ds():
    ds = RandomDictStringDataset(32, num_samples=100)
    dl = DataLoader(ds)

    for b in dl:
        pass


@pytest.mark.parametrize("scale_factor", [2, 4])
def test_sr_datasets(datadir, scale_factor):
    hr_image_size = 28
    lr_image_size = hr_image_size // scale_factor
    image_channels = 1
    dataset = SRMNISTDataset(hr_image_size, lr_image_size, image_channels, root=datadir)

    hr_image, lr_image = next(iter(DataLoader(dataset)))

    assert hr_image.size() == torch.Size([1, image_channels, hr_image_size, hr_image_size])
    assert lr_image.size() == torch.Size([1, image_channels, lr_image_size, lr_image_size])

    atol = 0.3
    assert torch.allclose(hr_image.min(), torch.tensor(-1.0), atol=atol)
    assert torch.allclose(hr_image.max(), torch.tensor(1.0), atol=atol)
    assert torch.allclose(lr_image.min(), torch.tensor(0.0), atol=atol)
    assert torch.allclose(lr_image.max(), torch.tensor(1.0), atol=atol)