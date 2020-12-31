import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.datasets.mnist_dataset import SRMNISTDataset
from pl_bolts.models.gans import GAN, SRGAN, SRResNet


@pytest.mark.parametrize(
    "dm_cls",
    [
        pytest.param(MNISTDataModule, id="mnist"),
        pytest.param(CIFAR10DataModule, id="cifar10"),
    ],
)
def test_gan(tmpdir, datadir, dm_cls):
    seed_everything(42)

    dm = dm_cls(data_dir=datadir)
    model = GAN(*dm.size())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)


@pytest.mark.parametrize("sr_module_cls", [SRResNet, SRGAN])
@pytest.mark.parametrize("scale_factor", [2, 4])
def test_sr_modules(tmpdir, datadir, sr_module_cls, scale_factor):
    seed_everything(42)

    dl = _sr_mnist_dataloader(datadir, scale_factor)
    model = sr_module_cls(image_channels=1, scale_factor=scale_factor)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dl)
    trainer.test()


def _sr_mnist_dataloader(datadir, scale_factor):
    hr_image_size = 28
    lr_image_size = hr_image_size // scale_factor
    image_channels = 1
    dl = DataLoader(SRMNISTDataset(hr_image_size, lr_image_size, image_channels, root=datadir), batch_size=2)
    return dl
