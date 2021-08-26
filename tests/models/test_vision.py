import pytest
import torch
from packaging import version
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning import __version__ as pl_version
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from pl_bolts.datamodules import FashionMNISTDataModule, MNISTDataModule
from pl_bolts.datasets import DummyDataset
from pl_bolts.models.vision import GPT2, ImageGPT, SemSegment, UNet


class DummyDataModule(LightningDataModule):
    def train_dataloader(self):
        train_ds = DummyDataset((3, 35, 120), (35, 120), num_samples=100)
        return DataLoader(train_ds, batch_size=1)


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
        checkpoint_callback=False,
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


@torch.no_grad()
def test_unet():
    x = torch.rand(10, 3, 28, 28)
    model = UNet(num_classes=2)
    y = model(x)
    assert y.shape == torch.Size([10, 2, 28, 28])


def test_semantic_segmentation(tmpdir):
    dm = DummyDataModule()

    model = SemSegment(num_classes=19)

    trainer = Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, datamodule=dm)
    loss = trainer.progress_bar_dict["loss"]

    assert float(loss) > 0
