import warnings
from pathlib import Path

import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

from pl_bolts.datasets import DummyDetectionDataset
from pl_bolts.models.detection import (
    YOLO,
    DarknetNetwork,
    FasterRCNN,
    RetinaNet,
    YOLOV4Network,
    YOLOV4P6Network,
    YOLOV4TinyNetwork,
    YOLOV5Network,
    YOLOXNetwork,
)
from pl_bolts.models.detection.faster_rcnn import create_fasterrcnn_backbone
from pl_bolts.models.detection.yolo.target_matching import _sim_ota_match
from pl_bolts.models.detection.yolo.utils import (
    aligned_iou,
    global_xy,
    grid_centers,
    grid_offsets,
    iou_below,
    is_inside_box,
)
from tests import TEST_ROOT


def _collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def test_fasterrcnn():
    model = FasterRCNN(pretrained=False, pretrained_backbone=False)

    image = torch.rand(1, 3, 224, 224)
    model(image)


def test_fasterrcnn_train(tmpdir):
    model = FasterRCNN(pretrained=False, pretrained_backbone=False)

    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_fasterrcnn_bbone_train(tmpdir):
    torch.manual_seed(123)
    model = FasterRCNN(backbone="resnet18", fpn=True, pretrained_backbone=False)
    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, valid_dl)


@torch.no_grad()
def test_retinanet():
    model = RetinaNet(pretrained=False)

    image = torch.rand(1, 3, 400, 400)
    model(image)


def test_retinanet_train(tmpdir):
    model = RetinaNet(pretrained=False)
    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_retinanet_backbone_train(tmpdir):
    model = FasterRCNN(backbone="resnet18", fpn=True, pretrained_backbone=False, pretrained=False)
    trainer = Trainer(fast_dev_run=True, logger=False, enable_checkpointing=False, default_root_dir=tmpdir)
    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    trainer.fit(model, train_dl, valid_dl)


def test_fasterrcnn_pyt_module_bbone_train(tmpdir):
    backbone = create_fasterrcnn_backbone(backbone="resnet18")
    model = FasterRCNN(backbone=backbone, fpn=True, pretrained_backbone=False, pretrained=False)
    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, valid_dl)


@pytest.mark.parametrize("width,height", [(10, 5)])
def test_grid_offsets(width: int, height: int):
    size = torch.tensor([width, height])
    offsets = grid_offsets(size)
    assert offsets.shape == (height, width, 2)
    assert torch.equal(offsets[0, :, 0], torch.arange(width, dtype=offsets.dtype))
    assert torch.equal(offsets[0, :, 1], torch.zeros(width, dtype=offsets.dtype))
    assert torch.equal(offsets[:, 0, 0], torch.zeros(height, dtype=offsets.dtype))
    assert torch.equal(offsets[:, 0, 1], torch.arange(height, dtype=offsets.dtype))


@pytest.mark.parametrize("width,height", [(10, 5)])
def test_grid_centers(width: int, height: int):
    size = torch.tensor([width, height])
    centers = grid_centers(size)
    assert centers.shape == (height, width, 2)
    assert torch.equal(centers[0, :, 0], 0.5 + torch.arange(width, dtype=torch.float))
    assert torch.equal(centers[0, :, 1], 0.5 * torch.ones(width))
    assert torch.equal(centers[:, 0, 0], 0.5 * torch.ones(height))
    assert torch.equal(centers[:, 0, 1], 0.5 + torch.arange(height, dtype=torch.float))


def test_global_xy():
    xy = torch.ones((2, 4, 4, 3, 2)) * 0.5  # 4x4 grid of coordinates to the center of the cell.
    image_size = torch.tensor([400, 200])
    xy = global_xy(xy, image_size)
    assert xy.shape == (2, 4, 4, 3, 2)
    assert torch.all(xy[:, :, 0, :, 0] == 50)
    assert torch.all(xy[:, 0, :, :, 1] == 25)
    assert torch.all(xy[:, :, 1, :, 0] == 150)
    assert torch.all(xy[:, 1, :, :, 1] == 75)
    assert torch.all(xy[:, :, 2, :, 0] == 250)
    assert torch.all(xy[:, 2, :, :, 1] == 125)
    assert torch.all(xy[:, :, 3, :, 0] == 350)
    assert torch.all(xy[:, 3, :, :, 1] == 175)


def test_is_inside_box():
    """
    centers:
        [[1,1; 3,1; 5,1; 7,1; 9,1; 11,1; 13,1; 15,1; 17,1; 19,1]
         [1,3; 3,3; 5,3; 7,3; 9,3; 11,3; 13,3; 15,3; 17,3; 19,3]
         [1,5; 3,5; 5,5; 7,5; 9,5; 11,5; 13,5; 15,5; 17,5; 19,5]
         [1,7; 3,7; 5,7; 7,7; 9,7; 11,7; 13,7; 15,7; 17,7; 19,7]
         [1,9; 3,9; 5,9; 7,9; 9,9; 11,9; 13,9; 15,9; 17,9; 19,9]]

    is_inside[0]:
        [[F, F, F, F, F, F, F, F, F, F]
         [F, T, T, F, F, F, F, F, F, F]
         [F, T, T, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]]

    is_inside[1]:
        [[F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, F, F, F]
         [F, F, F, F, F, F, F, T, T, F]]
    """
    size = torch.tensor([10, 5])
    centers = grid_centers(size) * 2.0
    centers = centers.view(-1, 2)
    boxes = torch.tensor([[2, 2, 6, 6], [14, 8, 18, 10]])
    is_inside = is_inside_box(centers, boxes).view(2, 5, 10)
    assert torch.count_nonzero(is_inside) == 6
    assert torch.all(is_inside[0, 1:3, 1:3])
    assert torch.all(is_inside[1, 4, 7:9])


def test_sim_ota_match():
    # IoUs will determined that 2 and 1 predictions will be selected for the first and the second target.
    ious = torch.tensor([[0.1, 0.1, 0.9, 0.9], [0.2, 0.3, 0.4, 0.1]])
    # Costs will determine that the first and the last prediction will be selected for the first target, and the first
    # prediction will be selected for the second target. Since the first prediction was selected for both targets, it
    # will be matched to the best target only (the second one).
    costs = torch.tensor([[0.3, 0.5, 0.4, 0.3], [0.1, 0.2, 0.5, 0.3]])
    matched_preds, matched_targets = _sim_ota_match(costs, ious)
    assert len(matched_preds) == 4
    assert matched_preds[0]
    assert not matched_preds[1]
    assert not matched_preds[2]
    assert matched_preds[3]
    assert len(matched_targets) == 2  # Two predictions were matched.
    assert matched_targets[0] == 1  # Which target was matched to the first prediction.
    assert matched_targets[1] == 0  # Which target was matched to the last prediction.


@pytest.mark.parametrize(
    "dims1, dims2, expected_ious",
    [
        (
            torch.tensor([[1.0, 1.0], [10.0, 1.0], [100.0, 10.0]]),
            torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            torch.tensor([[1.0 / 10.0, 1.0 / 40.0], [1.0 / 19.0, 2.0 / 48.0], [10.0 / 1000.0, 20.0 / 1020.0]]),
        )
    ],
)
def test_aligned_iou(dims1, dims2, expected_ious, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    torch.testing.assert_close(aligned_iou(dims1, dims2), expected_ious)


def test_iou_below():
    tl = torch.rand((10, 10, 3, 2)) * 100
    br = tl + 10
    pred_boxes = torch.cat((tl, br), -1)
    target_boxes = torch.stack((pred_boxes[1, 1, 0], pred_boxes[3, 5, 1]))
    result = iou_below(pred_boxes, target_boxes, 0.9)
    assert result.shape == (10, 10, 3)
    assert not result[1, 1, 0]
    assert not result[3, 5, 1]


@pytest.mark.parametrize("config", [("yolo"), ("yolo_giou")])
def test_darknet(config, catch_warnings):
    config_path = Path(TEST_ROOT) / "data" / f"{config}.cfg"
    network = DarknetNetwork(config_path)
    model = YOLO(network)

    image = torch.rand(1, 3, 256, 256)
    model(image)


@pytest.mark.parametrize(
    "cfg_name",
    [
        ("yolo"),
        ("yolo_giou"),
    ],
)
def test_darknet_train(tmpdir, cfg_name, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    config_path = Path(TEST_ROOT) / "data" / f"{cfg_name}.cfg"
    network = DarknetNetwork(config_path)
    model = YOLO(network)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov4_tiny(tmpdir):
    network = YOLOV4TinyNetwork(num_classes=2, width=4)
    model = YOLO(network)

    image = torch.rand(1, 3, 256, 256)
    model(image)


def test_yolov4_tiny_train(tmpdir):
    network = YOLOV4TinyNetwork(num_classes=2, width=4)
    model = YOLO(network)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov4(tmpdir):
    network = YOLOV4Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128))
    model = YOLO(network)

    image = torch.rand(1, 3, 256, 256)
    model(image)


def test_yolov4_train(tmpdir):
    network = YOLOV4Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128))
    model = YOLO(network)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov4p6(tmpdir):
    network = YOLOV4P6Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128, 128))
    model = YOLO(network)

    image = torch.rand(1, 3, 256, 256)
    model(image)


def test_yolov4p6_train(tmpdir):
    network = YOLOV4P6Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128, 128))
    model = YOLO(network)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov5(tmpdir):
    network = YOLOV5Network(num_classes=2, depth=1, width=4)
    model = YOLO(network)

    image = torch.rand(1, 3, 256, 256)
    model(image)


def test_yolov5_train(tmpdir):
    network = YOLOV5Network(num_classes=2, depth=1, width=4)
    model = YOLO(network)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolox(tmpdir):
    network = YOLOXNetwork(num_classes=2, depth=1, width=4)
    model = YOLO(network)

    image = torch.rand(1, 3, 256, 256)
    model(image)


def test_yolox_train(tmpdir):
    network = YOLOXNetwork(num_classes=2, depth=1, width=4)
    model = YOLO(network)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
