import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.models.gans import GAN


@pytest.mark.parametrize(
    "dm_cls", [pytest.param(MNISTDataModule, id="mnist"), pytest.param(CIFAR10DataModule, id="cifar10")]
)
def test_gan(tmpdir, datadir, dm_cls):
    seed_everything()

    dm = dm_cls(data_dir=datadir)
    model = GAN(*dm.size())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm, ckpt_path=None)
