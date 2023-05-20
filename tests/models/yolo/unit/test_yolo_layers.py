import warnings

import pytest
import torch
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from pl_bolts.models.detection.yolo.yolo_layers import GIoULoss, IoULoss, SELoss, _corner_coordinates


@pytest.mark.parametrize(
    ("xy", "wh", "expected"),
    [
        ([0.0, 0.0], [1.0, 1.0], [-0.5, -0.5, 0.5, 0.5]),
        ([5.0, 5.0], [2.0, 2.0], [4.0, 4.0, 6.0, 6.0]),
    ],
)
def test_corner_coordinates(xy, wh, expected, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    xy = torch.tensor(xy)
    wh = torch.tensor(wh)
    corners = _corner_coordinates(xy, wh)
    assert torch.allclose(corners, torch.tensor(expected))


@pytest.mark.parametrize(
    ("loss_func", "bbox1", "bbox2", "expected"),
    [
        (GIoULoss, [[0.0, 0.0, 120.0, 200.0]], [[189.0, 93.0, 242.0, 215.0]], 1.4144532680511475),
        (IoULoss, [[0.0, 0.0, 120.0, 200.0]], [[189.0, 93.0, 242.0, 215.0]], 1.0),
        (SELoss, [[0.0, 0.0, 120.0, 200.0]], [[189.0, 93.0, 242.0, 215.0]], 59479.0),
    ],
)
def test_loss_functions(loss_func, bbox1, bbox2, expected, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    loss_func = loss_func()
    tensor1 = torch.tensor(bbox1, dtype=torch.float32)
    tensor2 = torch.tensor(bbox2, dtype=torch.float32)

    loss = loss_func(tensor1, tensor2)
    assert loss.item() > 0.0
    assert loss.item() == expected
