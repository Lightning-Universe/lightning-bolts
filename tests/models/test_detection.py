from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pl_bolts.datasets import DummyDetectionDataset
from pl_bolts.models.detection import FasterRCNN, Yolo, YoloConfiguration
from pl_bolts.models.detection.yolo.yolo_layers import _aligned_iou


def _collate_fn(batch):
    return tuple(zip(*batch))


def test_fasterrcnn():
    model = FasterRCNN()

    image = torch.rand(1, 3, 400, 400)
    model(image)


def test_fasterrcnn_train(tmpdir):
    model = FasterRCNN()

    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=valid_dl)


def test_fasterrcnn_bbone_train(tmpdir):
    model = FasterRCNN(backbone="resnet18", fpn=True, pretrained_backbone=True)
    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, valid_dl)


def _create_yolo_config_file(config_path):
    config_file = open(config_path, 'w')
    config_file.write(
        '''[net]
width=256
height=256
channels=3

[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=2
size=1
stride=1
pad=1
activation=mish

[convolutional]
batch_normalize=1
filters=4
size=3
stride=1
pad=1
activation=mish

[shortcut]
from=-3
activation=linear

[convolutional]
size=1
stride=1
pad=1
filters=14
activation=linear

[yolo]
mask=2,3
anchors=1,2, 3,4, 5,6, 9,10
classes=2
scale_x_y=1.05
cls_normalizer=1.0
iou_normalizer=0.07
ignore_thresh=0.7

[route]
layers = -4

[upsample]
stride=2

[convolutional]
size=1
stride=1
pad=1
filters=14
activation=linear

[yolo]
mask=0,1
anchors=1,2, 3,4, 5,6, 9,10
classes=2
scale_x_y=1.05
cls_normalizer=1.0
iou_normalizer=0.07
ignore_thresh=0.7'''
    )
    config_file.close()


def test_yolo(tmpdir):
    config_path = Path(tmpdir) / 'yolo.cfg'
    _create_yolo_config_file(config_path)
    config = YoloConfiguration(config_path)
    model = Yolo(config.get_network())

    image = torch.rand(1, 3, 256, 256)
    model(image)


def test_yolo_train(tmpdir):
    config_path = Path(tmpdir) / 'yolo.cfg'
    _create_yolo_config_file(config_path)
    config = YoloConfiguration(config_path)
    model = Yolo(config.get_network())

    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dataloader=train_dl, val_dataloaders=valid_dl)


@pytest.mark.parametrize(
    "dims1, dims2, expected_ious", [(
        torch.tensor([[1.0, 1.0], [10.0, 1.0], [100.0, 10.0]]), torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        torch.tensor([[1.0 / 10.0, 1.0 / 40.0], [1.0 / 19.0, 2.0 / 48.0], [10.0 / 1000.0, 20.0 / 1020.0]])
    )]
)
def test_aligned_iou(dims1, dims2, expected_ious):
    torch.testing.assert_allclose(_aligned_iou(dims1, dims2), expected_ious)
