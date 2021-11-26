import pytest
import torch
from torch.utils.data import DataLoader

from pl_bolts.datasets import DummyDataset, RandomDataset, RandomDictDataset, RandomDictStringDataset
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST


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
    dl = DataLoader(SRMNIST(scale_factor, root=datadir, download=True))
    hr_image, lr_image = next(iter(dl))

    hr_image_size = 28
    assert hr_image.size() == torch.Size([1, 1, hr_image_size, hr_image_size])
    assert lr_image.size() == torch.Size([1, 1, hr_image_size // scale_factor, hr_image_size // scale_factor])

    atol = 0.3
    assert torch.allclose(hr_image.min(), torch.tensor(-1.0), atol=atol)
    assert torch.allclose(hr_image.max(), torch.tensor(1.0), atol=atol)
    assert torch.allclose(lr_image.min(), torch.tensor(0.0), atol=atol)
    assert torch.allclose(lr_image.max(), torch.tensor(1.0), atol=atol)
