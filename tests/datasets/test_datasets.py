import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transform_lib

from pl_bolts.datasets import (
    BinaryEMNIST,
    BinaryMNIST,
    DummyDataset,
    RandomDataset,
    RandomDictDataset,
    RandomDictStringDataset,
)
from pl_bolts.datasets.dummy_dataset import DummyDetectionDataset
from pl_bolts.datasets.sr_mnist_dataset import SRMNIST


@pytest.mark.parametrize("batch_size,num_samples", [(16, 100), (1, 0)])
def test_dummy_ds(catch_warnings, batch_size, num_samples):

    if num_samples > 0:

        ds = DummyDataset((1, 28, 28), (1,), num_samples=num_samples)
        dl = DataLoader(ds, batch_size=batch_size)

        assert isinstance(ds, Dataset)
        assert num_samples == len(ds)

        x = next(iter(ds))
        assert x[0].shape == torch.Size([1, 28, 28])
        assert x[1].shape == torch.Size([1])

        batch = next(iter(dl))
        assert batch[0].shape == torch.Size([batch_size, 1, 28, 28])
        assert batch[1].shape == torch.Size([batch_size, 1])

    else:
        with pytest.raises(ValueError, match="Provide an argument greater than 0"):
            ds = DummyDataset((1, 28, 28), (1,), num_samples=num_samples)


@pytest.mark.parametrize("batch_size,size,num_samples", [(16, 32, 100), (1, 0, 0)])
def test_rand_dict_ds(catch_warnings, batch_size, size, num_samples):

    if num_samples > 0 or size > 0:
        ds = RandomDictDataset(size, num_samples=num_samples)
        dl = DataLoader(ds, batch_size=batch_size)

        assert isinstance(ds, Dataset)
        assert num_samples == len(ds)

        x = next(iter(ds))
        assert x["a"].shape == torch.Size([size])
        assert x["b"].shape == torch.Size([size])

        batch = next(iter(dl))
        assert len(batch["a"]), len(batch["a"][0]) == (batch_size, size)
        assert len(batch["b"]), len(batch["b"][0]) == (batch_size, size)
    else:
        with pytest.raises(ValueError, match="Provide an argument greater than 0"):
            ds = RandomDictDataset(size, num_samples=num_samples)


@pytest.mark.parametrize("batch_size,size,num_samples", [(16, 32, 100), (1, 0, 0)])
def test_rand_ds(catch_warnings, batch_size, size, num_samples):
    if num_samples > 0 and size > 0:
        ds = RandomDataset(size=size, num_samples=num_samples)
        dl = DataLoader(ds, batch_size=batch_size)

        assert isinstance(ds, Dataset)
        assert num_samples == len(ds)

        x = next(iter(ds))
        assert x.shape == torch.Size([size])

        batch = next(iter(dl))
        assert len(batch), len(batch[0]) == (batch_size, size)

    else:
        with pytest.raises(ValueError, match="Provide an argument greater than 0"):
            ds = RandomDataset(size, num_samples=num_samples)


@pytest.mark.parametrize("batch_size,size,num_samples", [(16, 32, 100), (1, 0, 0)])
def test_rand_str_dict_ds(catch_warnings, batch_size, size, num_samples):

    if num_samples > 0 and size > 0:
        ds = RandomDictStringDataset(size=size, num_samples=100)
        dl = DataLoader(ds, batch_size=batch_size)

        assert isinstance(ds, Dataset)
        assert num_samples == len(ds)

        x = next(iter(ds))
        assert isinstance(x["id"], str)
        assert x["x"].shape == torch.Size([size])

        batch = next(iter(dl))
        assert len(batch["x"]) == batch_size
        assert len(batch["id"]) == batch_size
    else:
        with pytest.raises(ValueError, match="Provide an argument greater than 0"):
            ds = RandomDictStringDataset(size, num_samples=num_samples)


@pytest.mark.parametrize("batch_size,img_shape,num_samples", [(16, (3, 256, 256), 100), (1, (256, 256), 0)])
def test_dummy_detection_ds(catch_warnings, batch_size, img_shape, num_samples):
    if num_samples > 0:
        ds = DummyDetectionDataset(img_shape=img_shape, num_boxes=3, num_classes=3, num_samples=num_samples)
        dl = DataLoader(ds, batch_size=batch_size)

        assert isinstance(ds, Dataset)
        assert num_samples == len(ds)

        batch = next(iter(dl))
        x, y = batch
        assert x.size() == torch.Size([batch_size, *img_shape])
        assert y["boxes"].size() == torch.Size([batch_size, 3, 4])
        assert y["labels"].size() == torch.Size([batch_size, 3])

    else:
        with pytest.raises(ValueError, match="Provide an argument greater than 0"):
            ds = DummyDetectionDataset(img_shape=img_shape, num_boxes=3, num_classes=3, num_samples=num_samples)


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


def test_binary_mnist_dataset(datadir):
    """Check BinaryMNIST image and target dimensions and value range."""
    dl = DataLoader(BinaryMNIST(root=datadir, download=True, transform=transform_lib.ToTensor()))
    img, target = next(iter(dl))

    assert img.size() == torch.Size([1, 1, 28, 28])
    assert target.size() == torch.Size([1])

    assert torch.allclose(img.min(), torch.tensor(0.0))
    assert torch.allclose(img.max(), torch.tensor(1.0))
    assert torch.equal(torch.unique(img), torch.tensor([0.0, 1.0]))


@pytest.mark.parametrize("split", ["byclass", "bymerge", "balanced", "letters", "digits", "mnist"])
def test_binary_emnist_dataset(datadir, split):
    """Check BinaryEMNIST image and target dimensions and value range for each split."""
    dl = DataLoader(BinaryEMNIST(root=datadir, split=split, download=True, transform=transform_lib.ToTensor()))
    img, target = next(iter(dl))

    assert img.size() == torch.Size([1, 1, 28, 28])
    assert target.size() == torch.Size([1])

    assert torch.allclose(img.min(), torch.tensor(0.0))
    assert torch.allclose(img.max(), torch.tensor(1.0))
    assert torch.equal(torch.unique(img), torch.tensor([0.0, 1.0]))