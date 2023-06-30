import warnings
from pathlib import Path

import pytest
import torch
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
    YOLOV7Network,
    YOLOXNetwork,
)
from pl_bolts.models.detection.faster_rcnn import create_fasterrcnn_backbone
from pl_bolts.utils import _IS_WINDOWS
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

from tests import TEST_ROOT


def _collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def test_fasterrcnn():
    model = FasterRCNN(pretrained=False, pretrained_backbone=False)

    image = torch.rand(1, 3, 224, 224)
    model(image)


@pytest.mark.flaky(reruns=3)
@pytest.mark.skipif(_IS_WINDOWS, reason="failing...")  # todo
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


@pytest.mark.parametrize("config", ["yolo", "yolo_giou"])
def test_darknet(config, catch_warnings):
    config_path = Path(TEST_ROOT) / "_data_configs" / f"{config}.cfg"
    network = DarknetNetwork(config_path)
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


@pytest.mark.parametrize("cfg_name", ["yolo", "yolo_giou"])
def test_darknet_train(tmpdir, cfg_name, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    config_path = Path(TEST_ROOT) / "_data_configs" / f"{cfg_name}.cfg"
    network = DarknetNetwork(config_path)
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov4_tiny(catch_warnings):
    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV4TinyNetwork(num_classes=2, width=4, overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


def test_yolov4_tiny_train(tmpdir):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV4TinyNetwork(num_classes=2, width=4, overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov4(catch_warnings):
    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV4Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128), overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


def test_yolov4_train(tmpdir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV4Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128), overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov4p6(catch_warnings):
    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV4P6Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128, 128), overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


def test_yolov4p6_train(tmpdir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV4P6Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128, 128), overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov5(catch_warnings):
    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV5Network(num_classes=2, depth=1, width=4, overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


def test_yolov5_train(tmpdir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV5Network(num_classes=2, depth=1, width=4, overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolov7(catch_warnings):
    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV7Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128), overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


def test_yolov7_train(tmpdir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOV7Network(num_classes=2, widths=(4, 8, 16, 32, 64, 128), overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


def test_yolox(catch_warnings):
    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOXNetwork(num_classes=2, depth=1, width=4, overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    image = torch.rand(3, 256, 256)
    detections = model.infer(image)
    assert "boxes" in detections
    assert "scores" in detections
    assert "labels" in detections


def test_yolox_train(tmpdir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message=".*does not have many workers which may be a bottleneck.*",
        category=PossibleUserWarning,
    )

    # Using giou allows the tests to pass also with older versions of Torchvision.
    network = YOLOXNetwork(num_classes=2, depth=1, width=4, overlap_func="giou")
    model = YOLO(network, confidence_threshold=0.5)

    train_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(num_classes=2), collate_fn=_collate_fn)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir, logger=False, max_epochs=10, accelerator="auto")
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)
