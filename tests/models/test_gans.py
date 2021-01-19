import pytest
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torchvision import transforms as transform_lib

from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule
from pl_bolts.models.gans import DCGAN, GAN


@pytest.mark.parametrize(
    "dm_cls", [
        pytest.param(MNISTDataModule, id="mnist"),
        pytest.param(CIFAR10DataModule, id="cifar10"),
    ]
)
def test_gan(tmpdir, datadir, dm_cls):
    seed_everything()

    dm = dm_cls(data_dir=datadir, num_workers=0)
    model = GAN(*dm.size())
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm, ckpt_path=None)


@pytest.mark.parametrize(
    "dm_cls", [pytest.param(MNISTDataModule, id="mnist"),
               pytest.param(CIFAR10DataModule, id="cifar10")]
)
def test_dcgan(tmpdir, datadir, dm_cls):
    seed_everything()

    transforms = transform_lib.Compose([transform_lib.Resize(64), transform_lib.ToTensor()])
    dm = dm_cls(data_dir=datadir, train_transforms=transforms, val_transforms=transforms, test_transforms=transforms)

    model = DCGAN(image_channels=dm.dims[0])
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm, ckpt_path=None)
