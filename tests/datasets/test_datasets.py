import pytest
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from pl_bolts.datasets import DummyDataset, RandomDataset, RandomDictDataset, RandomDictStringDataset
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST
from pl_bolts.datasets.cifar10_dataset import CIFAR10


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

def test_cifar10_datasets(datadir):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dl = DataLoader(CIFAR10(root=datadir, download=True, transform=transform))
    hr_image, lr_image = next(iter(dl))
    print("==============================", lr_image.size())

    hr_image_size = 32
    assert hr_image.size() == torch.Size([1, 3, hr_image_size, hr_image_size])
    assert lr_image.size() == torch.Size([1])

    atol = 0.3
    assert torch.allclose(hr_image.min(), torch.tensor(-1.0), atol=atol)
    assert torch.allclose(hr_image.max(), torch.tensor(1.0), atol=atol)
    assert torch.greater_equal(lr_image.min(), torch.tensor(0))
    assert torch.less_equal(lr_image.max(), torch.tensor(9))