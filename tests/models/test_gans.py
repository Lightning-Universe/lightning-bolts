import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.datasets.cifar10_dataset import CIFAR10_SR
from pl_bolts.models.gans import GAN, SRGAN, SRResNet


@pytest.mark.parametrize(
    "dm_cls",
    [
        pytest.param(MNISTDataModule, id="mnist"),
        pytest.param(CIFAR10DataModule, id="cifar10"),
    ],
)
def test_gan(tmpdir, datadir, dm_cls):
    seed_everything()

    dm = dm_cls(data_dir=datadir)
    model = GAN(*dm.size())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


def test_srresnet(tmpdir, datadir):
    seed_everything()

    dl = DataLoader(CIFAR10_SR(datadir), batch_size=2, num_workers=1)
    model = SRResNet()
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dl)
    trainer.test()


def test_srgan(tmpdir, datadir):
    seed_everything()

    dl = DataLoader(CIFAR10_SR(datadir), batch_size=2, num_workers=1)
    model = SRGAN()
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dl)
    trainer.test()
