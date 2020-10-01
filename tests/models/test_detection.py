import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from pl_bolts.datamodules import DummyDetectionDataset
from pl_bolts.models.detection import FasterRCNN
from pl_bolts.models.detection.detr import detr_model


def _collate_fn(batch):
    return tuple(zip(*batch))


def test_fasterrcnn(tmpdir):
    model = FasterRCNN()

    image = torch.rand(1, 3, 400, 400)
    model(image)


def test_fasterrcnn_train(tmpdir):
    model = FasterRCNN()

    train_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(), collate_fn=_collate_fn)

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, valid_dl)


def test_detr(tmpdir):
    model = detr_model.Detr()
    image = torch.rand(1, 3, 400, 400)
    model(image)
    return 1


def test_detr_train(tmpdir):
    model = detr_model.Detr(num_classes=91)

    train_dl = DataLoader(DummyDetectionDataset(img_shape=(3, 256, 256), num_boxes=1, num_classes=91, num_samples=10),
                          collate_fn=_collate_fn)
    valid_dl = DataLoader(DummyDetectionDataset(img_shape=(3, 256, 256), num_boxes=1, num_classes=91, num_samples=10),
                          collate_fn=_collate_fn)

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, valid_dl)
