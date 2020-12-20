import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule, STL10_SR_DataModule
from pl_bolts.models.gans import GAN, SRGAN, SRResNet


@pytest.mark.parametrize(
    "dm_cls", [pytest.param(MNISTDataModule, id="mnist"), pytest.param(CIFAR10DataModule, id="cifar10")]
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

    dm = STL10_SR_DataModule(datadir, num_workers=1, batch_size=2)
    model = SRResNet()
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


def test_srgan(tmpdir, datadir):
    seed_everything()

    dm = STL10_SR_DataModule(datadir, num_workers=1, batch_size=2)
    model = SRGAN()
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)