import numpy as np
import pytest
import torch
from PIL import Image

from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform,
    SimCLRFinetuneTransform,
    SimCLRTrainDataTransform,
)
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVEvalDataTransform,
    SwAVFinetuneTransform,
    SwAVTrainDataTransform,
)


@pytest.mark.parametrize(
    "transform_cls",
    [pytest.param(SwAVTrainDataTransform, id="train-data"), pytest.param(SwAVEvalDataTransform, id="eval-data")],
)
def test_swav_train_data_transform(catch_warnings, transform_cls):
    # dummy image
    img = np.random.randint(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    crop_sizes = (96, 36)

    # size of the generated views
    transform = transform_cls(size_crops=crop_sizes)
    views = transform(img)

    # the transform must output a list or a tuple of images
    assert isinstance(views, (list, tuple))

    # the transform must output three images
    # (2 Global Crops, 4 Local Crops, online evaluation view)
    assert len(views) == 7

    # all views are tensors
    assert all(torch.is_tensor(v) for v in views)

    # Global Views have equal size
    assert all(v.size(1) == v.size(2) == crop_sizes[0] for v in views[:2])
    # Check local views have same size
    assert all(v.size(1) == v.size(2) == crop_sizes[1] for v in views[2 : len(views) - 1])  # Ignore online transform


def test_swav_finetune_transform(catch_warnings):
    # dummy image
    img = np.random.randint(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)

    # size of the generated views
    input_height = 96
    transform = SwAVFinetuneTransform(input_height=input_height)
    view = transform(img)

    # the view generator is a tensor
    assert torch.is_tensor(view)

    # view has expected size
    assert view.size(1) == view.size(2) == input_height


@pytest.mark.parametrize(
    "transform_cls",
    [pytest.param(SimCLRTrainDataTransform, id="train-data"), pytest.param(SimCLREvalDataTransform, id="eval-data")],
)
def test_simclr_train_data_transform(catch_warnings, transform_cls):
    # dummy image
    img = np.random.randint(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)

    # size of the generated views
    input_height = 96
    transform = transform_cls(input_height=input_height)
    views = transform(img)

    # the transform must output a list or a tuple of images
    assert isinstance(views, (list, tuple))

    # the transform must output three images
    # (1st view, 2nd view, online evaluation view)
    assert len(views) == 3

    # all views are tensors
    assert all(torch.is_tensor(v) for v in views)

    # all views have expected sizes
    assert all(v.size(1) == v.size(2) == input_height for v in views)


def test_simclr_finetune_transform(catch_warnings):
    # dummy image
    img = np.random.randint(low=0, high=255, size=(32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img)

    # size of the generated views
    input_height = 96
    transform = SimCLRFinetuneTransform(input_height=input_height)
    view = transform(img)

    # the view generator is a tensor
    assert torch.is_tensor(view)

    # view has expected size
    assert view.size(1) == view.size(2) == input_height
