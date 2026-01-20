import pytest
import torch
from pytorch_lightning import seed_everything

from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    emnist_normalization,
    imagenet_normalization,
    stl10_normalization,
)


@pytest.mark.parametrize(
    "normalization",
    [cifar10_normalization, imagenet_normalization, stl10_normalization],
)
def test_normalizations(normalization, catch_warnings):
    """Test normalizations for CIFAR10, ImageNet, STL10."""
    seed_everything(1234)
    x = torch.rand(3, 32, 32)
    assert normalization()(x).shape == (3, 32, 32)
    assert x.min() >= 0.0
    assert x.max() <= 1.0


@pytest.mark.parametrize(
    "split",
    ["balanced", "byclass", "bymerge", "digits", "letters", "mnist"],
)
def test_emnist_normalizations(split, catch_warnings):
    """Test normalizations for each EMNIST dataset split."""
    x = torch.rand(1, 28, 28)
    assert emnist_normalization(split)(x).shape == (1, 28, 28)
    assert x.min() >= 0.0
    assert x.max() <= 1.0
