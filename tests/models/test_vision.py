import warnings

import pytest
import torch
from packaging import version
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning import __version__ as pl_version
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch.utils.data import DataLoader

from pl_bolts.datamodules import FashionMNISTDataModule, MNISTDataModule
from pl_bolts.datasets import DummyDataset
from pl_bolts.models.vision import GPT2, ImageGPT, SemSegment, UNet
from pl_bolts.models.vision.unet import DoubleConv, Down, Up


class DummyDataModule(LightningDataModule):
    def train_dataloader(self):
        train_ds = DummyDataset((3, 35, 120), (35, 120), num_samples=100)
        return DataLoader(train_ds, batch_size=1)

    def val_dataloader(self):
        valid_ds = DummyDataset((3, 35, 120), (35, 120), num_samples=100)
        return DataLoader(valid_ds, batch_size=1)


@pytest.mark.skipif(
    version.parse(pl_version) > version.parse("1.1.0"), reason="igpt code not updated for latest lightning"
)
def test_igpt(tmpdir, datadir):
    seed_everything(0)
    dm = MNISTDataModule(data_dir=datadir, normalize=False)
    model = ImageGPT()

    trainer = Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    assert trainer.callback_metrics["test_loss"] < 1.7

    dm = FashionMNISTDataModule(data_dir=datadir, num_workers=1)
    model = ImageGPT(classify=True)
    trainer = Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, datamodule=dm)


@torch.no_grad()
def test_gpt2():
    seed_everything(0)
    seq_len = 17
    batch_size = 32
    vocab_size = 16
    x = torch.randint(0, vocab_size, (seq_len, batch_size))

    model = GPT2(
        embed_dim=16,
        heads=2,
        layers=2,
        num_positions=seq_len,
        vocab_size=vocab_size,
        num_classes=10,
    )
    model(x)


def test_unet_component(catch_warnings):
    x1 = torch.rand(1, 3, 28, 28)
    x2 = torch.rand(1, 64, 28, 33)
    x3 = torch.rand(1, 32, 64, 69)

    doubleConvLayer = DoubleConv(3, 64)
    y = doubleConvLayer(x1)
    assert y.shape == torch.Size([1, 64, 28, 28])

    downLayer = Down(3, 6)
    y = downLayer(x1)
    assert y.shape == torch.Size([1, 6, 14, 14])

    upLayer1 = Up(64, 32, False)
    upLayer2 = Up(64, 32, True)
    y1 = upLayer1(x2, x3)
    y2 = upLayer2(x2, x3)
    assert y1.shape == torch.Size([1, 32, 64, 69])
    assert y2.shape == torch.Size([1, 32, 64, 69])


@torch.no_grad()
def test_unet(catch_warnings):
    x = torch.rand(10, 3, 28, 28)
    model = UNet(num_classes=2)
    y = model(x)
    assert y.shape == torch.Size([10, 2, 28, 28])


def test_semantic_segmentation(tmpdir, catch_warnings):
    warnings.filterwarnings(
        "ignore",
        message="The dataloader, val_dataloader 0, does not have many workers which may be a bottleneck",
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The dataloader, train_dataloader, does not have many workers which may be a bottleneck",
        category=PossibleUserWarning,
    )
    dm = DummyDataModule()

    model = SemSegment(num_classes=19)
    progress_bar = TQDMProgressBar()

    trainer = Trainer(
        fast_dev_run=True,
        max_epochs=-1,
        default_root_dir=tmpdir,
        logger=False,
        accelerator="auto",
        callbacks=[progress_bar],
    )
    trainer.fit(model, datamodule=dm)
