import pytorch_lightning as pl
import torch

from pl_bolts.datamodules import MNISTDataModule, FashionMNISTDataModule, DummyDataset
from pl_bolts.models import GPT2, ImageGPT, UNet, SemSegment
from torch.utils.data import DataLoader


def test_igpt(tmpdir):
    pl.seed_everything(0)
    dm = MNISTDataModule(tmpdir, normalize=False)
    model = ImageGPT()

    trainer = pl.Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
    )
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)
    assert trainer.callback_metrics["test_loss"] < 1.7

    dm = FashionMNISTDataModule(tmpdir, num_workers=1)
    model = ImageGPT(classify=True)
    trainer = pl.Trainer(
        limit_train_batches=2,
        limit_val_batches=2,
        limit_test_batches=2,
        max_epochs=1,
    )
    trainer.fit(model, datamodule=dm)


def test_gpt2(tmpdir):

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


def test_unet(tmpdir):
    x = torch.rand(10, 3, 28, 28)
    model = UNet(num_classes=2)
    y = model(x)
    assert y.shape == torch.Size([10, 2, 28, 28])


def test_semantic_segmentation(tmpdir):

    class DummyDataModule(pl.LightningDataModule):
        def train_dataloader(self):
            train_ds = DummyDataset((3, 35, 120), (35, 120), num_samples=100)
            return DataLoader(train_ds, batch_size=1)

    dm = DummyDataModule()

    model = SemSegment(datamodule=dm, num_classes=19)

    trainer = pl.Trainer(fast_dev_run=True, max_epochs=1)
    trainer.fit(model)
    loss = trainer.progress_bar_dict['loss']

    assert float(loss) > 0
