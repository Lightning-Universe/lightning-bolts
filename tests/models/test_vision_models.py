import torch
from pl_bolts.models import GPT2, ImageGPT
from pl_bolts.datamodules import MNISTDataModule
import pytorch_lightning as pl


def test_igpt(tmpdir):
    pl.seed_everything(0)
    dm = MNISTDataModule(tmpdir, normalize=False)
    model = ImageGPT(datamodule=dm)

    trainer = pl.Trainer(limit_train_batches=2, limit_val_batches=2, limit_test_batches=2, max_epochs=1)
    trainer.fit(model)
    trainer.test()
    assert trainer.callback_metrics['test_loss'] < 1.7

    model = ImageGPT(classify=True)
    trainer = pl.Trainer(limit_train_batches=2, limit_val_batches=2, limit_test_batches=2, max_epochs=1)
    trainer.fit(model)


def test_gpt2(tmpdir):

    seq_len = 17
    batch_size = 32
    classes = 10
    x = torch.randint(0, 10, (seq_len, batch_size))

    model = GPT2(embed_dim=16, heads=2, layers=2, num_positions=28 * 28, vocab_size=16, num_classes=classes)
    model(x)
